"""The thinking tree: one block's search, from fan-out to winner.

run_block is the whole search. It fans out on the thought's first token into
up to BEAM_WIDTH conversations (fan_out), then runs checkpoints until the
beams close the block. Each checkpoint is a fixed sequence of steps, all but
two of them pure functions of the frontier here: advance (ONE streamed
request moves every live beam by a chunk), _expand (every candidate),
prune and churn with _apply_churn, _assign_split_uids, the judge
(scoring.score_axes, ONE request) with _update_leader and _award_medals, and
the two early ends, _should_stop and _drop_ramblers. finalise_block records
the winner (_pick_winner) once the loop is over. rank is the labelled Choice,
which the search does not call at present.

The pure steps read config at call time and, where they change anything,
change only the beams handed to them. Beams are compared by IDENTITY
throughout, never by value: Beam is a dataclass, so `==` would compare every
field, and the candidates this tree produces differ by a single token.
"""

from dataclasses import dataclass

from . import config, prompts
from .events import _beam_json, _block_open_event, _standings_event, _think_step_event
from .llm import UsageMeter
from .scoring import _in_contention, score_axes
from .state import Beam
from .text import cut_at_marker


@dataclass
class _Advanced:
    """The chunks one checkpoint's request produced, one per live beam.

    Filled in by advance, which is an async generator and so cannot
    return a value. Stays None if the stream never delivered them.
    """

    chunks: list = None


def _by_overall(b):
    """Sort key: best `overall` first, an unjudged thought (None) as -9."""
    return -(b.overall if b.overall is not None else -9)


def _child(b, text, score, tokens, penalty, done, last):
    """A candidate that continues beam `b` - a finished thought or a branch.

    It stays in b's lineage (same uid, from_uid = b.uid, parent = b's slot)
    and carries b's judgement down with it: that was earned by THIS
    conversation's last chunk, and carrying it is what lets the next
    checkpoint retire a laggard without re-judging anything. `axes` is a COPY,
    so rescoring one child can never rewrite a sibling's.
    """
    return Beam(text=text, score=score, parent=b.index, uid=b.uid, from_uid=b.uid,
                tokens=tokens, penalty=penalty, done=done, last=last,
                axes=dict(b.axes), overall=b.overall, raw=b.raw)


def _expand(live, chunks, ctx):
    """Every candidate one checkpoint produces, finished thoughts included.

    `chunks` holds (chunk, chunk_lp, top, stopped) per live beam, in `live`
    order, as advance collected them. A beam whose chunk closed its
    thought stands as ONE finished candidate; any other branches into one
    candidate per alternative in `top`. `ctx` is only read, for the copy
    penalty. Nothing passed in is modified: every candidate is a new Beam.
    """
    # Each beam advances by its whole chunk, then branches at the end of
    # it. width x k candidates, exactly as before - the tree has the
    # same shape, it just checkpoints every THINK_CHUNK tokens instead
    # of every one, because a round trip costs the same either way.
    cands = []
    for b, (chunk, chunk_lp, top, stopped) in zip(live, chunks):
        body, chunk_marker = cut_at_marker(b.text + chunk)
        b_tokens = b.tokens + max(1, len(chunk) // 4)
        # A beam that closed its own think block inside the chunk is
        # finished: it does not branch, it just stands as a candidate.
        if chunk_marker is not None or stopped or not top:
            cands.append(
                _child(b, text=body, score=b.score + chunk_lp, tokens=b_tokens,
                       penalty=b.penalty, done=True, last=chunk[-12:])
            )
            continue
        # THE BRANCH POINT IS THE LAST TOKEN'S POSITION, NOT THE ONE
        # AFTER IT. `top` is the distribution vLLM reported AT the
        # chunk's final token - the alternatives to the word it just
        # wrote - so appending one to the whole chunk duplicated that
        # word. Every beam of an earlier run ended "...looking at at",
        # "...person. IfIf", "...Carla is married married".
        #
        # So the stem is the chunk WITHOUT its last token, and each
        # alternative replaces it. At temperature 0 the token actually
        # generated is the top-ranked one, which is how it can be
        # identified and cut, and which makes candidate 0 "carry on
        # exactly as you were".
        best_tok, best_lp = top[0]
        stem = (body[: -len(best_tok)]
                if best_tok and body.endswith(best_tok) else body)
        seen_tokens = set()
        for tok, lp in top:
            clean, marker = cut_at_marker(stem + tok)
            # The tree used to burn its whole budget reproducing the
            # closing line of an earlier block, which sits in its
            # context. Retire that beam when it starts, rather than
            # truncating it afterwards in _inline_note.
            if prompts.NUDGE_PREFIX[:12] in clean:
                clean = clean[: clean.find(prompts.NUDGE_PREFIX[:12])]
                marker = marker or "nudge"
            pen = 0.0
            # Diverse beam search: a token an earlier sibling already
            # took is worth less, so the frontier cannot fill with
            # several spellings of the same word.
            if tok in seen_tokens:
                pen += config.SIBLING_PENALTY
            seen_tokens.add(tok)
            # Copying the context is free in logprob terms and was
            # winning every block. Charge for it.
            if config.COPY_PENALTY and len(clean) >= config.COPY_NGRAM:
                if clean[-config.COPY_NGRAM :] in ctx:
                    pen += config.COPY_PENALTY
            cands.append(
                _child(
                    b,
                    text=clean,
                    # chunk_lp already counts the last token, and this
                    # candidate REPLACES it - so take it back out
                    # before adding the alternative's.
                    score=b.score + chunk_lp - best_lp + lp,
                    tokens=b_tokens + 1,
                    penalty=b.penalty + pen,
                    done=marker is not None,
                    last=tok,
                )
            )
    return cands


def _apply_churn(beams, retired, entered, next_uid, closed):
    """Carry out what churn decided. Returns (beams, next_uid, closed).

    The retired conversations leave the frontier, the ones that entered join
    it at the end, and `next_uid` moves past the uids the newcomers were
    given. With nothing retired, all three come back unchanged.
    """
    if retired:
        next_uid += len(entered)
        # By IDENTITY. Beam is a dataclass, so `b not in retired`
        # compares every field and would drop any beam that merely
        # looks like the retired one - and near-identical beams are
        # exactly what this tree produces.
        out = {id(b) for b in retired}
        beams = [b for b in beams if id(b) not in out] + entered
        # A retirement with no replacement CLOSES the slot: the odds
        # said another 500 tokens was not worth spending. Counted, or
        # the next checkpoint's backfill would quietly reopen it.
        closed += len(retired) - len(entered)
    return beams, next_uid, closed


def _assign_split_uids(beams, next_uid):
    """Number the slots and give every split its own uid. Returns next_uid.

    A lineage that contributes two survivors has SPLIT: the second one is a
    new conversation and gets its own identity, so the UI can show it being
    born rather than just renumbering a slot. Writes `index` and `uid` onto
    the beams in place.
    """
    taken = set()
    for i, b in enumerate(beams):
        b.index = i
        if b.uid in taken:
            b.uid = next_uid
            next_uid += 1
        taken.add(b.uid)
    return next_uid


def _update_leader(contenders, leader, dry):
    """The block's best raw score after a judging round, and its dry streak.

    Returns (leader, dry). Did this checkpoint beat anything? A new best
    resets the streak; otherwise the block has gone another round without
    improving, and the churn starts closing slots once that has held for
    CHURN_PATIENCE checkpoints.

    It has to be a REAL improvement. `raw` drifts upward simply because a
    longer thought scores better on "progress", so "did anything beat the
    leader at all" would be reset by mere length and the block would never
    wind down. Requiring a margin also bounds it: raw is 0..1, so there can
    only ever be a hundred resets, and in practice it plateaus in a few.

    Without a new best the leader still becomes the larger of the two, with
    None read as 0.0 on either side.
    """
    top = max((c.raw for c in contenders if c.raw is not None), default=None)
    if top is not None and (leader is None or top > leader + 0.01):
        return top, 0
    return max(leader or 0.0, top or 0.0), dry + 1


def _award_medals(contenders):
    """Sort `contenders` best `overall` first, in place; medal the top three.

    Gold, silver and bronze are 1, 2 and 3; everyone else gets 0. The sort
    is stable, so a tie keeps the order the list came in.
    """
    contenders.sort(key=_by_overall)
    for i, c in enumerate(contenders):
        c.medal = i + 1 if i < 3 else 0


def _should_stop(finished, beams, leader):
    """Have enough thoughts finished that another one probably will not win?

    `finished` must already be in standings order, best first.

    For interchangeable draws the chance the next finisher is a new best is
    1/(n+1) - 1/2, 1/4, 1/6 - so records get rare fast. At three finished
    that is 25%, and if the leader is also clear of the runner-up there is
    little left to find. If it is close, let the others land.

    ...but NEVER while a conversation that is still scoring well is still
    writing. This used to break outright, killing every live beam
    mid-thought - including one that might have turned out to be the best
    in the block. Three others finishing is not a reason to stop the one
    that is still going strong.
    """
    if len(finished) >= config.STOP_AFTER_FINISHED and len(finished) > 1:
        lead = finished[0].overall - finished[1].overall
        contending = [b for b in beams if _in_contention(b, leader)]
        if lead >= config.STOP_MARGIN and not contending:
            return True
    return False


def _drop_ramblers(beams, finished, leader):
    """The live beams, less any that are both far too long and out of contention.

    A live beam grown far past the best finished thought MIGHT be rambling -
    or it might be the one actually doing the work on a hard question, where
    the short thought that finished first was the shallow one. Length is not
    badness. So a long beam is only dropped if it is ALSO clearly behind on
    score; if it is still scoring well it keeps going however long it has
    run.

    With nothing finished there is no length to compare against, and `beams`
    comes back as it is.
    """
    if not finished:
        return beams
    longest_good = max(f.tokens for f in finished)
    return [b for b in beams
            if b.tokens <= longest_good * config.RAMBLE_RATIO
            or _in_contention(b, leader)]


def _pick_winner(finished, beams):
    """The thought that becomes the block's monologue.

    THE WINNER. Exactly what the lines below do:

      - `longest` is the length, in characters of stripped text, of the
        longest thought here, finished or live.
      - A FINISHED thought qualifies if its text is at least a tenth of
        that. If any qualify, the winner is the qualifying thought with
        the best `raw`, and no live beam can win.
      - Only if none qualify (including when nothing finished) is the
        winner the best `raw` among ALL contenders, finished and live.
      - An unjudged thought (raw None) counts as -1. Ties go to list
        order: finished thoughts first, in their last standings order,
        then live beams. With no contenders the monologue is empty.

    Why a complete thought beats a half-written one: when `done` was only a
    tie-break here, a live beam scoring 0.79 beat a finished one at 0.78 and
    the block's monologue ended mid-sentence: "...not both circles. So".

    Why the length floor: the opposite failure. A beam that opened on a
    stray "!" closed itself immediately, was the only finished thought, and
    won its block at raw 0.536 against three live thoughts at 0.739-0.781 -
    the monologue was the single character "!". The floor is relative, like
    RAMBLE_RATIO, so "!" against 9450 characters fails and four comparable
    thoughts all pass.

    The order is `finished + beams`, where the medals use `beams +
    finished`. They are separate rules and stay separate: swapping either
    one would change who wins a tie.
    """
    contenders = finished + beams
    longest = max((len(b.text.strip()) for b in contenders), default=0)
    whole = [b for b in finished if len(b.text.strip()) >= longest / 10]
    return max(
        whole or contenders,
        key=lambda b: b.raw if b.raw is not None else -1.0,
        default=Beam(text="", score=0.0, parent=-1),
    )


async def run_block(llm, block, ctx, seed=None):
    """Run the thinking tree: fan out, then advance and judge per chunk.

    `ctx` is the WHOLE prompt up to the point the thought starts - system
    turn, history, question, assistant header and whatever of the answer
    exists. Everything downstream appends to it: the beams continue it, and
    the axis judge closes it with <|im_end|> and asks a question after it.

    The block first fans out on its opening token into up to BEAM_WIDTH
    conversations (fan_out). After that, each checkpoint is two requests
    regardless of how wide the tree is: one streamed request advances
    every live beam by up to THINK_CHUNK tokens and reports the top
    THINK_BRANCH_K alternatives for its last token, and one judges every
    live and finished thought on the axes. The width never grows: at most
    BEAM_WIDTH x THINK_BRANCH_K candidates are produced and at most
    BEAM_WIDTH survive. Slots are given up only by thoughts finishing or
    by the churn closing them.

    `seed` is what opens the monologue, and it is the ONLY difference
    between the opening thought and a triggered one. Mid-answer it is
    INLINE_SEED, which carries two words of "hold on" because without them
    the model restarts the sentence. At the native position it is the bare
    `<think>` of the template, because there is nothing to restart.

    A checkpoint is a fixed sequence of steps: advance (the request),
    _expand (every candidate), prune and the churn (churn,
    _apply_churn), _assign_split_uids, the judge (score_axes,
    _update_leader, _award_medals), then the two ways a block can end
    early (_should_stop, _drop_ramblers). finalise_block picks the
    winner (_pick_winner) once the loop is over.
    """
    seed = config.INLINE_SEED if seed is None else seed
    meter = UsageMeter(llm)
    # A beam's `text` is the GENERATED body only. The seed is prepended
    # when a prompt is built and never stored, because the seed itself
    # contains "<think>" - keeping it in the beam made cut_at_marker fire on
    # the first step, retire every beam at once and blank the monologue.
    finished = []      # thoughts that closed themselves
    closed = 0         # slots retired without a replacement
    # Is the block still finding better thoughts? `leader` is the best raw
    # score seen anywhere in it, `dry` the number of checkpoints since
    # that last moved. Together they are the answer to "is there a good
    # opportunity for a challenger", and they decide whether a retired
    # slot is refilled or shut.
    leader, dry = None, 0

    yield _block_open_event(block, seed)

    # --- fan out BEFORE advancing -------------------------------------
    # The loop below advances a beam by a whole chunk and branches only at
    # the end of it. Starting from one beam, that means the first 500
    # tokens are written by a single conversation - and since most thoughts
    # close inside their first chunk, the tree never branched at all.
    # Measured just now: the opening thought on "bat and ball" finished in
    # chunk 0, one card appeared, and three lineages never existed.
    #
    # So the block starts the way the design always described it: take the
    # BEAM_WIDTH most likely ways the thought could open and keep that many
    # conversations going. One request, one token, and from then on every
    # beam is writing its own chunk.
    beams = await fan_out(llm, ctx, seed)
    next_uid = len(beams)
    # The conversations exist now, before a single chunk has been written.
    # Without this the cards were only created at a CHECKPOINT, so a block
    # whose thoughts all finish inside their first chunk - the normal case
    # at 500 tokens - streamed its tokens into cards that did not exist yet
    # and showed an empty panel.
    yield {"type": "frontier", "block": block.id,
           # How many conversations the fan-out ACTUALLY started, against
           # how many were asked for. Below the width and the frontier is
           # narrow from the very first token; the backfill then fills
           # the missing slots with near-identical twins, so the block
           # looks wide but explores a single stream. Reported so it
           # cannot hide.
           "started": len(beams), "width": config.BEAM_WIDTH,
           "beams": [_beam_json(b) for b in beams]}

    # No limit. The block ends when its beams close it.
    step = -1
    while True:
        step += 1
        live = [b for b in beams if not b.done]
        if not live:
            break

        adv = _Advanced()
        async for ev in advance(llm, block, live, ctx, seed, adv):
            yield ev

        cands = _expand(live, adv.chunks, ctx)
        # A thought that closed itself LEAVES the frontier for the pool.
        # It is not standardised against half-written beams - those are
        # different kinds of thing - and its slot is not refilled. The
        # frontier only ever shrinks, which is why the block is guaranteed
        # to end, and it matches the measured diversity decay (0.76 -> 0.16
        # across a block): beams late on are near paraphrases, so keeping
        # the full width there buys almost nothing and loads a busy server.
        newly_done = [c for c in cands if c.done]
        cands = [c for c in cands if not c.done]
        if newly_done:
            finished.extend(newly_done)
        # NOTE: the break on "no live candidates" happens AFTER the events
        # below. With 500-token chunks most thoughts finish in the first
        # checkpoint, so leaving here would skip think_step (which creates
        # the cards) and standings (which draws the badges) - which is why
        # no medals ever appeared.

        # Candidates get in on the FREE signal. All k from one beam are the
        # same 500-token text with a different final token, so a model
        # judgement here has nothing to separate - which is why calling
        # rank at this point meant judging eight near-copies every
        # checkpoint. The model's judgement runs later instead: the axis
        # judge scores the survivors once each has its own 500 tokens.
        #
        # The labelled Choice (rank) is not called here, so the engine
        # does not read config.CHOICE_EVERY at present and the ranking is
        # always logprob (see docs/CONFIG.md).
        cands.sort(key=lambda b: -b.rank_score)
        ranked, used_choice = cands, False

        before = {b.uid for b in beams}
        # The frontier shrinks only by FINISHING. A thought that closed
        # itself keeps its slot: it is not an empty slot, it is one held by
        # a completed conversation that is still scored and can still win,
        # and slots getting fewer that way is how a block ends.
        width = max(1, config.BEAM_WIDTH - len(finished) - closed)
        beams = prune(ranked, width)
        retired, entered = churn(beams, ranked, next_uid, leader, dry)
        beams, next_uid, closed = _apply_churn(beams, retired, entered, next_uid, closed)
        # How many slots hold a chunk no other slot holds. Below the width
        # means lineages died and the backfill took siblings again - twins
        # that cannot diverge, since generation at temperature 0 is
        # effectively deterministic. Reported rather than silent.
        distinct_slots = len({b.parent for b in beams})
        next_uid = _assign_split_uids(beams, next_uid)
        block.steps = step + 1

        yield _think_step_event(block, step, cands, used_choice, beams, finished,
                                before, retired, entered, leader, dry, closed,
                                distinct_slots)

        # Standings, every checkpoint. Everything still IN the running is
        # scored - the ones still writing as well as the ones that have
        # finished - so the ranking is visible throughout rather than only
        # at the end. Beams that pruning discarded are out of the running
        # and are not scored.
        contenders = beams + finished
        if contenders:
            # Two groups, ONE request. Standardised within each: a
            # half-written thought scores badly on "progress" purely for
            # being half-written, so ranking it against finished ones
            # would retire live beams for being alive.
            await score_axes(llm, ctx, beams, finished)
            leader, dry = _update_leader(contenders, leader, dry)
            _award_medals(contenders)
            finished.sort(key=_by_overall)
            yield _standings_event(block, contenders, beams, finished)
        # Every thought closed itself. The events above have already gone
        # out, so the cards and the medals are drawn before we leave.
        if not cands:
            break

        # NOTHING STOPS A THOUGHT PART-WAY THROUGH. There was a rule here
        # that ended the block after four checkpoints without a better
        # score, and because it fired while beams were still writing, the
        # half-written text became the block's monologue: "...the
        # constraint: not both circles. So", "...but I acknowledge". A
        # thought ends when the model ends it.
        #
        # `dry` still decides whether a RETIRED slot is refilled or shut,
        # which is about a conversation already judged not to be a
        # challenger. That is a different thing from cutting a good one
        # off mid-sentence.

        # Stop when another thought probably will not beat the ones we
        # have - but never while a live one is still in contention.
        if _should_stop(finished, beams, leader):
            break

        # A long live beam is dropped only if it is ALSO clearly behind.
        if finished:
            beams = _drop_ramblers(beams, finished, leader)
            if not beams:
                break

    yield finalise_block(block, finished, beams, meter)


async def advance(llm, block, live, ctx, seed, out):
    """Advance every live beam by one chunk, in ONE streamed request.

    Streamed, so the conversations can be watched growing token by token
    even though the tree only checkpoints every chunk: each piece is
    yielded as a think_token event. The prompts follow `live` order, and
    so do the chunks, which go into `out` (an _Advanced) because an async
    generator cannot return a value.
    """
    async for kind, *rest in llm.top_next(
        [ctx + seed + b.text for b in live]
    ):
        if kind == "tok":
            i, piece = rest
            yield {"type": "think_token", "block": block.id,
                   "uid": live[i].uid, "text": piece}
        else:
            out.chunks = rest[0]


def finalise_block(block, finished, beams, meter):
    """Record the winner and what the block cost. Returns the monologue event.

    `finished` and `beams` are the pool and the frontier as the loop left
    them; `meter` has been running since the block opened.
    """
    block.beams = finished + beams
    winner = _pick_winner(finished, beams)
    block.monologue = winner.text
    block.winner_score = round(winner.raw, 3) if winner.raw is not None else 0.0
    block.finished_count = len(finished)
    used = meter.used()
    block.explored_tokens = used["completion_tokens"]
    block.requests = used["requests"]
    block.secs = meter.secs()
    return {"type": "monologue", "block": block.id, "text": block.monologue.strip()}


# vLLM's default --max-logprobs. Asking for more is a 400, so this is the
# ceiling on how many distinct openings a fan-out can ever see.
MAX_LOGPROBS = 20


async def fan_out(llm, ctx, seed):
    """The frontier a block starts from: one opening token each.

    The distribution over the FIRST thinking token is read in a one-token
    request and its best entries become the beams. That is the only moment
    the conversations can be made to differ by construction - at
    temperature 0 two beams sharing a prefix generate the same text
    forever, so beams not separated here can never separate later.

    It reads more alternatives than it needs and filters: a top-k full of
    whitespace and markers would otherwise start the block two
    conversations wide while claiming four.
    """
    want = min(MAX_LOGPROBS, config.BEAM_WIDTH * config.THINK_BRANCH_K)
    best = []
    # Read it twice if the first read comes back degenerate. Measured on
    # a live endpoint: two byte-identical one-token requests, seconds
    # apart, returned 20 usable openings ('Let', 'The', 'We', 'This', ...)
    # and then a list yielding only ('Let', '!'). Whatever causes that - on
    # a busy server batching is concurrent - a block that starts
    # one conversation wide has already lost, and the retry costs one
    # request only in the case that would otherwise be ruined.
    for _ in range(2):
        first = None
        async for kind, *rest in llm.top_next(
            [ctx + seed], k=want, max_tokens=1
        ):
            if kind == "end":
                first = rest[0][0]
        openings = []
        if first:
            for tok, lp in first[2] or []:
                clean, marker = cut_at_marker(tok)
                # A first token that closes the block, is punctuation, or
                # is nothing but whitespace, is not a conversation.
                if marker is not None or not clean.strip():
                    continue
                if not any(ch.isalnum() for ch in clean):
                    continue
                openings.append((clean, lp))
                if len(openings) >= config.BEAM_WIDTH:
                    break
        if len(openings) > len(best):
            best = openings
        if len(best) >= config.BEAM_WIDTH:
            break
    # The read failed or every alternative was a marker. The block still
    # runs, just without the fan-out.
    return [
        Beam(text=tok, score=lp, parent=-1, from_uid=-1,
             uid=i, index=i, tokens=1, last=tok)
        for i, (tok, lp) in enumerate(best)
    ] or [Beam(text="", score=0.0, parent=-1, uid=0)]


def churn(beams, ranked, next_uid, leader, dry):
    """Retire a conversation that will not be a challenger; refill or close.

    Returns (retired, entered). Two questions, in this order.

    IS IT A CHALLENGER? Measured against `leader`, the best thought
    anywhere in the block, because that is the thing it has to beat - not
    against the other live beams, which may all be poor. Absolute raw
    scores, so the comparison across live and finished means something. A
    conversation judged on its own last 500 tokens and still CHURN_MARGIN
    behind the leader is not going to catch it.

    IS THE SLOT WORTH ANOTHER 500 TOKENS? That depends on whether the
    block is still finding better thoughts. `dry` counts checkpoints since
    the last new best:

      - still improving -> REPLACE, with a variation of a good branch:
        same good prefix, different next word, already computed each
        checkpoint and otherwise thrown away
      - dry for CHURN_PATIENCE checkpoints -> new streams are not bringing
        a challenge, so CLOSE the slot instead. The frontier winds down,
        one slot per checkpoint, and the block converges

    The margin of safety is CHURN_PATIENCE: nothing closes on one quiet
    checkpoint. And throughout: at most one leaves, never the last one
    standing, never one that has not been judged, and it stays on screen
    with the score that retired it.
    """
    judged = [b for b in beams if b.raw is not None]
    if len(beams) < 2 or not judged or leader is None:
        return [], []
    worst = min(judged, key=lambda b: b.raw)
    if _in_contention(worst, leader):
        return [], []

    if dry >= config.CHURN_PATIENCE:
        return [worst], []

    # A variation of a GOOD stream: the best discarded sibling of the
    # highest-scoring lineage that is staying. Not of the one leaving -
    # that is the branch just judged not worth pursuing.
    best = max(judged, key=lambda b: b.raw)
    kept = {id(b) for b in beams}
    spare = next(
        (c for c in ranked
         if id(c) not in kept and not c.done and c.parent == best.parent),
        None,
    )
    if spare is None:
        # Nothing to put in its place, so the slot closes rather than be
        # held open by a conversation already judged out of the running.
        return [worst], []
    # A new conversation, so the UI shows a birth rather than a renumbered
    # slot, and `from_uid` says which branch it came off.
    spare.uid, spare.from_uid = next_uid, best.uid
    return [worst], [spare]


def prune(ranked, width=None):
    """Take the best BEAM_WIDTH, but not all from one parent.

    Without the cap the highest-scoring lineage's children sweep the whole
    frontier - its score dominates, so every survivor is a sibling and the
    conversations become spellings of one. Measured: six steps in a
    124-step sample had collapsed to a single parent.
    """
    width = config.BEAM_WIDTH if width is None else width
    cap = max(1, config.MAX_CHILDREN_PER_PARENT)
    kept, used = [], {}
    for b in ranked:
        if used.get(b.parent, 0) >= cap:
            continue
        kept.append(b)
        used[b.parent] = used.get(b.parent, 0) + 1
        if len(kept) >= width:
            return kept
    # Backfill if the cap was too strict to fill the frontier. By
    # IDENTITY: Beam is a dataclass, so `b not in kept` compares every
    # field, and candidates here differ by a single token.
    have = {id(b) for b in kept}
    for b in ranked:
        if len(kept) >= width:
            break
        if id(b) not in have:
            kept.append(b)
            have.add(id(b))
    return kept


async def rank(llm, ctx, cands, reverse=False, debias=None, seed=None):
    """Rank with a labelled Choice (inspired by jevfire), debiased.

    Every option goes in one prompt against verified single-token labels and
    the answer is read at a single position, so the model compares the
    candidates instead of judging each alone.

    Not called by the engine at present - run_block ranks by logprob and
    judges on the axes, and config.CHOICE_EVERY is not read.
    experiments/tree_probe.py calls it directly for its label-position check.

    The plain version does not work here. Measured on eight probes, the
    winner changed on FOUR when the options were listed in the opposite
    order - the model was reading the letter as much as the text. That is
    a known label-position bias: an earlier measurement found a field
    scoring 3/100 forward and 98/100 reversed.

    The fix is to ask both orderings and average each candidate's
    probability, which cancels a positional preference instead of hoping it
    is absent. It is free: both prompts go in the SAME batched request, so
    the step still costs one round trip.

    `reverse` forces a single reversed ordering, and exists only so the bias
    check can measure what happens without the averaging.
    """
    debias = config.CHOICE_DEBIAS if debias is None else debias
    order = list(reversed(cands)) if reverse else list(cands)
    n = len(order)
    labels = await llm.labels(n)
    names, ids = [t for t, _ in labels], [i for _, i in labels]
    texts = [b.text.strip() or "(nothing yet)" for b in order]

    seed = config.INLINE_SEED if seed is None else seed
    batch = [prompts.choice_prompt(ctx + seed, texts, names)]
    if debias and not reverse:
        batch.append(
            prompts.choice_prompt(ctx + seed, list(reversed(texts)), names)
        )
    rows = await llm.answer_probs(batch, ids)

    for i, b in enumerate(order):
        p = rows[0][i]
        if len(rows) > 1:
            p = (p + rows[1][n - 1 - i]) / 2
        b.p = p
    return sorted(order, key=lambda b: -(b.p or 0.0))
