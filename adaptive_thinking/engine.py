"""The engine: opening thought, scored gate, thinking TREE, age-based eviction.

None of this is the shape it started with, and each part changed because a
measurement said so.

THE OPENING THOUGHT. With OPEN_THINK on (the default), every turn starts with
the tree run at Qwen's own <think> position: the assistant header stops before
the thinking slot, the tree fills it, and the winning thought is rendered there
by chatml.think_head. At that position `</think>` means exactly what the
template says - the answer begins here - so there is no seam to repair.

THE GATE. Mid-answer, the model is asked where a block is worth opening. The
original design used entropy as a veto: probe only where the model looks
unsure. Measured, entropy and the probe are independent - see
config.ENTROPY_FLOOR. So entropy is now only a floor, and the gate is:

    stream a WINDOW of tokens, keeping each position's top-K distribution
      -> candidates = positions where branching could change anything at all
         (entropy above ENTROPY_FLOOR, at least two distinct alternatives),
         past WARMUP_TOKENS, MIN_GAP_TOKENS after the last block, at a
         sentence or clause boundary, and not inside text already written
      -> ONE batched request scores the PROBE_CANDIDATES highest-entropy
         candidates on the GATE_AXES (stakes / difficulty / commitment):
         levelled closed questions, each read at a single answer position,
         no generation -> crit, an absolute 0..1
      -> the best candidate at or above GATE_THRESHOLD opens a block

With GATE_LOG_YESNO on, the older Yes/No probe is asked as well, in one extra
request, and only recorded; PROBE_THRESHOLD is the number
experiments/gate_audit.py compares it at. Neither decides anything.

THE BLOCK. The original design branched ONCE on the top-K tokens and ran K
independent completions to the end. That is K one-shot guesses, not a search:
the entire shape of the exploration was decided by a single token, nothing
advanced together, and nothing was ever pruned. It is now a tree:

    rewind to the chosen position (the opening thought has nothing to rewind)
    fan out: the best BEAM_WIDTH first tokens of the thought, read in one
      one-token request, become BEAM_WIDTH conversations - at temperature 0,
      beams that share a prefix never separate, so they are separated here
    every checkpoint, two requests whatever the width:
      ONE streamed request advances every live beam by up to THINK_CHUNK
        tokens; each proposes THINK_BRANCH_K alternatives for its last token,
        and the candidates are ranked on penalised, length-normalised logprob
        and pruned straight back to the width. The width never grows: it
        stays BEAM_WIDTH.
      ONE request judges every live and finished thought on the AXES
    a thought that closes itself leaves the frontier for the finished pool,
      keeps its slot and its score, and can still win; slots are given up
      only that way or when the churn closes one
    churn: a live conversation clearly behind the best score (CHURN_MARGIN)
      is retired, and a variation of a good branch takes its slot while the
      block is still improving; after CHURN_PATIENCE checkpoints without a new
      best, the slot closes instead. Nothing but a clearly bad score
      (_in_contention) removes a live conversation
    the block ends when no live beam is left, or when STOP_AFTER_FINISHED
      thoughts have finished with a clear leader and no live beam is still in
      contention; the winning thought becomes its monologue (see tree.run_block)

The beams carry THINKING tokens. When the block closes, the winning
deliberation is what stays in the context; no token is forced into the answer.
The model then resumes and picks its own next token, having read its own
reasoning - and whether that token differs from the one it was about to write
is the only honest measure of whether any of this did anything.

Mid-answer, the note the model reads is its own monologue behind INLINE_SEED,
a fixed closing line, `</think>`, and then a replay of the answer's last words,
so the answer is already in flight at the point `</think>` says it begins (see
text._inline_note). If the model starts the answer over anyway, the note is
dropped and the answer resumes without it (see text._restarted).

Resuming after a rewind re-sends a prefix the server has already seen, so it is
a prefix-cache hit rather than a re-prefill.

A thinking block then STAYS in the context, at the offset where it was written,
for the rest of the conversation - the opening thought in the assistant header,
a mid-answer block inside the answer. It is not moved and it is not rewritten,
so a turn reads the same after it finishes as it did while running.
Blocks are reclaimed later by age: eviction triggers once one is
EVICT_AGE_TOKENS behind and then deletes the whole band older than
EVICT_KEEP_TOKENS at once. Nothing is summarised - the notes are simply gone,
the prefix changes, and the next request re-prefills. That is the cost of
keeping the cache stable the rest of the time.

WHERE THE PIECES LIVE. This module holds the Engine: one turn (respond and
its phases) and eviction. The gate is gate.py, the tree tree.py, the levelled
reads and the judge scoring.py; the state they work on is state.py, the
string helpers text.py, and the event builders events.py. The Engine keeps a
method for each step it delegates (_choose, _run_block, _fan_out, _prune,
_churn, _score_axes, _rank, ...), so every step can still be driven through
an Engine.
"""

import math
from dataclasses import dataclass, field

from . import chatml, config, gate, scoring, tree
from .events import _block_close_event, _done_stats, _window_event
from .gate import _Cand
from .llm import VLLM, UsageMeter, transient
from .state import HEAD_OFF, Block, Conversation, Note, Turn
from .text import _inline_note, _restarted, _whole_sentences, cut_at_marker, splice


# -- one turn -----------------------------------------------------------------


@dataclass
class _TurnState:
    """Everything one turn of Engine.respond carries from phase to phase.

    `vis` is the answer the reader sees. `notes` are thinking records spliced
    into what the MODEL sees, at a visible-text offset. With
    KEEP_BLOCKS_INLINE off the two are identical and the block leaves no
    trace, so continuing after it is seamless by construction - which is
    precisely why that setting cannot answer whether the model copes with an
    interrupted answer. Turn it on and the seam becomes measurable.

    The phases rebind `vis` and `notes` rather than editing them in place,
    exactly as the single method they came from did, so whatever reads them
    reads the current value from here.
    """

    prefix: str  # the prompt up to the answer, the opening thought included
    vis: str = ""
    notes: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    idx: int = 0  # tokens currently standing in `vis`
    wasted: int = 0  # generated, then rewound away
    # Starts at 0, not at -MIN_GAP: without this the first block can open on
    # token 1, where the only context the monologue has to reflect on is the
    # word "To".
    last_block_at: int = 0
    next_id: int = 0
    resume_left: int = 0  # tokens still being watched after a block
    spills: list = field(default_factory=list)  # transcript markers cut out of the answer, with position
    restarts: list = field(default_factory=list)  # blocks whose note made the model rewrite the answer
    resume_from: tuple = None  # (offset, idx) to roll back to if that happens
    pending_greedy: str = None  # token the gate step was about to emit
    pending_block: Block = None
    logprobs_all: list = field(default_factory=list)
    logprobs_resume: list = field(default_factory=list)
    degraded: str = None  # the endpoint failure that cut the answer short, if one did


@dataclass
class _Window:
    """What one streamed window leaves for the rest of the loop.

    Filled in by Engine._stream_window, which is an async generator and so
    cannot return a value.
    """

    cands: list = field(default_factory=list)  # a _Cand per token, for the gate
    hit_stop: bool = False  # a transcript marker ended the stream
    restarted: bool = False  # the answer began again after a block

class Engine:
    def __init__(self, llm: VLLM):
        self.llm = llm

    async def respond(self, conv: Conversation, user_text: str):
        """Async generator of events, each a dict with a "type".

        start, then for the opening thought (and each triggered block):
        block_open, frontier, think_token, think_step, standings, monologue,
        block_close. While the answer streams: token, window (every position
        the gate scored), rewind, resumed (the first token after a block),
        restart (a note was dropped), degraded (the endpoint gave out), evict,
        and finally done, whose `stats` summarise the turn.

        The turn runs in phases that share one _TurnState: the opening
        thought (_opening_thought), then windows of answer (_stream_window),
        each checked by the gate, with a block wherever it fires (_think_at)
        and a rollback wherever a note made the answer start over
        (_undo_restart); last, the turn is written into the conversation
        (_commit_turn) and summarised (_done_stats).
        """
        meter = UsageMeter(self.llm)
        # With the opening thought on, the header stops before the thinking
        # slot so the tree can fill it. With it off this is the old prompt
        # exactly: the empty NO_THINK block and a cold start.
        st = _TurnState(prefix=conv.render(pending_user=user_text, think=config.OPEN_THINK))

        yield {"type": "start", "user": user_text, "prefix_chars": len(st.prefix)}

        # Outside the `try` below, on purpose: an endpoint failure during the
        # opening thought still ends the turn with an error.
        if config.OPEN_THINK:
            async for ev in self._opening_thought(conv, st):
                yield ev

        # No cap on the answer. It ends when the model ends its turn - or, if
        # the endpoint gives out, when it gives out. A 524 from the gateway
        # used to take the whole turn down AFTER the opening thought had
        # closed, throwing away 374 seconds of thinking to show an error. The
        # thinking is already in `notes` by then and the answer so far is in
        # `vis`, so the turn finishes with what it has and says it is partial.
        try:
            while True:
                win = _Window()
                async for ev in self._stream_window(st, win):
                    yield ev

                # A restart means the note did more harm than the thinking did
                # good. Roll back to the block, DROP the note, and carry on from
                # there without it - losing the deliberation, which is the cheaper
                # of the two losses. The MIN_GAP_TOKENS guard stops the gate
                # re-firing on the same position, so this cannot loop.
                if win.restarted:
                    yield self._undo_restart(st)
                    continue

                # Read only now: the stream has been closed, so its finish
                # reason is final.
                finished = win.hit_stop or self.llm.last_finish == "stop"

                if not win.cands:
                    break

                # No cap on how many times it may stop to think - that was the
                # opposite of the idea. The gate's own guards decide:
                # GATE_THRESHOLD on the axis score, plus ENTROPY_FLOOR,
                # MIN_GAP_TOKENS, WARMUP_TOKENS and the boundary and
                # repetition checks in _choose.
                opened = await self._choose(win.cands, st.last_block_at, st.prefix, st.vis, st.notes)
                if opened is not None:
                    yield _window_event(opened)

                if opened is None or opened["pick"] is None:
                    if finished:
                        break
                    continue

                cand, gate = opened["pick"]
                async for ev in self._think_at(conv, st, cand, gate):
                    yield ev
        except Exception as exc:
            # Only an endpoint failure is swallowed. A bug in here must
            # still surface, loudly, rather than be reported as a partial
            # answer.
            if not transient(exc):
                raise
            st.degraded = f"{type(exc).__name__}: {exc}"
            yield {"type": "degraded", "error": st.degraded, "kept": st.idx}

        text = self._commit_turn(conv, st, user_text)

        async for ev in self.maybe_evict(conv):
            yield ev

        used = meter.used()
        yield {
            "type": "done",
            "text": text,
            "stats": _done_stats(st, meter.secs(), used, len(conv.live_notes())),
        }

    async def _opening_thought(self, conv, st):
        """The tree run at Qwen's own <think> position, before the answer.

        The same tree, run at the position Qwen's template reserves for it,
        before a single answer token exists. Nothing gates it: thinking
        before answering is the default, not something to be triggered.

        It needs none of the machinery the mid-answer blocks need. There is
        nothing to rewind, nothing to resume, no sentence in flight to
        continue, and `</think>` means exactly what the template says it
        means - the answer begins here - so there is no seam to measure.

        Yields the block's events, then its block_close. Leaves the thought
        in `st`: as a note, in the prefix, and among the turn's blocks.
        """
        head_block = Block(id=st.next_id, tail="", crit=None, kind="open")
        st.next_id += 1
        async for ev in self._run_block(head_block, st.prefix,
                                        seed=config.OPEN_SEED):
            yield ev
        # It goes into the context as a note like any other, so age-based
        # eviction reclaims it with the rest and nothing here is special-
        # cased. HEAD_OFF puts it in the header rather than in the answer.
        head_text = _whole_sentences(head_block.monologue)
        st.notes.append(Note(off=HEAD_OFF, text=head_text, born=conv.clock))
        st.prefix += chatml.think_head(head_text)
        st.blocks.append(head_block)
        # No token was pending: the answer had not started.
        yield _block_close_event(head_block, was=None)

    async def _stream_window(self, st, out):
        """Stream one WINDOW of the answer into `st`, a token at a time.

        Yields `resumed` (the first token after a block) and a `token` per
        token. What the rest of the loop needs afterwards goes into `out`, a
        _Window: every token as a gate candidate, whether a transcript marker
        stopped the stream, and whether the answer started over after a
        block.
        """
        budget = config.WINDOW_TOKENS
        # Held explicitly so it can be closed on an early break. Abandoning
        # a suspended httpx stream to the garbage collector raises
        # "aclose(): asynchronous generator is already running" later, in an
        # unrelated part of the program.
        gen = self.llm.stream(st.prefix + splice(st.vis, st.notes), max_tokens=budget)
        async for step in gen:
            # vLLM streams the triggering token before applying `stop`, so
            # without this the marker lands in the visible answer.
            clean, marker = cut_at_marker(step.text)
            if marker is not None:
                step.text = clean
                out.hit_stop = True
                # Which marker, and where. A bare count says a run was
                # dirty without saying whether the model ended its turn in
                # the same chunk as its last word - harmless - or started
                # writing the transcript, which is not.
                st.spills.append({"marker": marker, "at": st.idx,
                                  "kept": clean, "empty": not clean})
                if not step.text:
                    break
            out.cands.append(_Cand(st.idx, len(st.vis), step))
            st.vis += step.text
            st.idx += 1
            # How surprised is the model by its own continuation? Free, and
            # it is the whole answer to "can it pick up where it left off":
            # a confused resume shows up as a logprob drop right after the
            # block and nowhere else.
            st.logprobs_all.append(step.logprob)
            # The first token after a block answers "did thinking change
            # anything" directly, which the old winner-index test never did.
            if st.pending_greedy is not None:
                st.pending_block.changed_next_token = step.text != st.pending_greedy
                yield {
                    "type": "resumed",
                    "block": st.pending_block.id,
                    "was": st.pending_greedy,
                    "now": step.text,
                    "changed": st.pending_block.changed_next_token,
                }
                st.pending_greedy = st.pending_block = None
            if st.resume_left > 0:
                st.logprobs_resume.append(step.logprob)
                st.resume_left -= 1
            yield {
                "type": "token",
                "text": step.text,
                "h": round(step.entropy, 3),
                "lp": round(step.logprob, 3),
                "resume": st.resume_left > 0,
                "top": [{"t": t, "p": round(math.exp(v), 4)} for t, v in step.top[: config.BRANCH_K]],
            }
            if out.hit_stop:
                break
            # Did the block make it start the answer over? Watched only
            # for a short while after a block; past that, a repeat is the
            # answer's own business.
            if st.resume_from is not None:
                if _restarted(st.vis, st.resume_from[0]):
                    out.restarted = True
                    break
                if len(st.vis) - st.resume_from[0] > 400:
                    st.resume_from = None
        await gen.aclose()

    @staticmethod
    def _undo_restart(st):
        """Roll the answer back to the last block and drop that block's note.

        Called when the note made the model start the answer over. Returns
        the `restart` event.
        """
        off0, idx0 = st.resume_from
        st.restarts.append({"at": idx0, "rewrote": st.idx - idx0})
        st.vis, st.idx = st.vis[:off0], idx0
        st.notes = [n for n in st.notes if n.off != off0]
        st.resume_from = None
        st.pending_greedy = st.pending_block = None
        st.resume_left = 0
        return {"type": "restart", "at": idx0,
                "rewrote": st.restarts[-1]["rewrote"]}

    async def _think_at(self, conv, st, cand, gate):
        """Open a block where the gate fired: rewind, think, leave the note.

        Yields `rewind` (when tokens were dropped), the block's own events,
        and its block_close.
        """
        # --- rewind ---------------------------------------------------
        dropped = st.idx - cand.idx
        st.wasted += dropped
        st.vis = st.vis[: cand.off]
        st.notes = [n for n in st.notes if n.off <= cand.off]
        st.idx = cand.idx
        if dropped:
            yield {"type": "rewind", "tokens": dropped}

        block = Block(id=st.next_id, tail=st.vis[-70:], crit=gate["crit"], gate=gate)
        st.next_id += 1
        # The FULL prompt, not just the answer so far. This used to pass
        # splice(vis, notes) alone, so the beams deliberated over a bare
        # fragment with no system turn, no question and no assistant
        # header - and _score_axes, which appends <|im_end|> to this same
        # string, was building malformed ChatML on top of it. The gate
        # probe always used the full prefix; the block that the gate opens
        # now does too.
        async for ev in self._run_block(block, st.prefix + splice(st.vis, st.notes)):
            yield ev

        # Nothing is committed into the answer any more. The block produced
        # thinking, not a token; generation resumes at the same position and
        # the model picks its own next token, having now read its own
        # deliberation. What it picks is the measurement - see below.
        if config.KEEP_BLOCKS_INLINE:
            # The note quotes the answer's last words back, so the model
            # leaves the block with the sentence in flight rather than
            # facing what reads as a blank page - see _inline_note.
            st.notes.append(Note(off=len(st.vis),
                                 text=_inline_note(block, st.vis[-240:]),
                                 born=conv.clock + st.idx))
            # Where to roll back to if it begins the answer again anyway.
            st.resume_from = (len(st.vis), st.idx)
        st.last_block_at = st.idx
        st.resume_left = config.RESUME_WATCH_TOKENS
        # The token the gate step was about to emit. If the next token after
        # the block differs from this, thinking changed the answer.
        st.pending_greedy = cand.step.text
        st.pending_block = block
        st.blocks.append(block)

        yield _block_close_event(block, was=st.pending_greedy)

    @staticmethod
    def _commit_turn(conv, st, user_text):
        """Write the finished turn into the conversation. Returns its text."""
        # Last line of defence. If a marker survived every earlier cut, it is a
        # bug worth seeing in the stats, not something to hide from the reader.
        clean, marker = cut_at_marker(st.vis)
        if marker is not None:
            st.spills.append({"marker": marker, "at": st.idx, "kept": "", "empty": False})
            st.vis = clean
        text = st.vis.rstrip()

        # The notes go on the turn, at the offsets they were written at, and the
        # clock advances so their age can be measured from here on.
        for n in st.notes:
            n.off = min(n.off, len(text))
        conv.turns.append(Turn("user", user_text))
        conv.turns.append(Turn("assistant", text, blocks=st.blocks, notes=st.notes))
        conv.clock += st.idx
        return text

    # -- the gate and the tree -------------------------------------------------
    #
    # Thin delegates to gate.py, tree.py and scoring.py, which hold the code.
    # Each passes this engine's client on and returns exactly what the module
    # function returns - the same coroutine or async generator, not a wrapper
    # around it - so every step can still be driven through an Engine, as the
    # tests and the measurement scripts do.

    def _choose(self, window, last_block_at, prefix, vis, notes):
        """Pick where in this window to open a block, if anywhere (gate.choose)."""
        return gate.choose(self.llm, window, last_block_at, prefix, vis, notes)

    def _gate_scores(self, ctxs):
        """How critical the next step is at each position (gate.gate_scores)."""
        return gate.gate_scores(self.llm, ctxs)

    def _run_block(self, block, ctx, seed=None):
        """Run one block's thinking tree, yielding its events (tree.run_block)."""
        return tree.run_block(self.llm, block, ctx, seed)

    def _fan_out(self, ctx, seed):
        """The frontier a block starts from (tree.fan_out)."""
        return tree.fan_out(self.llm, ctx, seed)

    def _score_axes(self, ctx, *groups):
        """Judge every candidate on the axes, in one request (scoring.score_axes)."""
        return scoring.score_axes(self.llm, ctx, *groups)

    def _rank(self, ctx, cands, reverse=False, debias=None, seed=None):
        """Rank candidates with the labelled Choice (tree.rank)."""
        return tree.rank(self.llm, ctx, cands, reverse, debias, seed)

    _prune = staticmethod(tree.prune)
    _churn = staticmethod(tree.churn)
    _finalise_block = staticmethod(tree.finalise_block)
    # vLLM's default --max-logprobs: the most openings a fan-out can see.
    MAX_LOGPROBS = tree.MAX_LOGPROBS

    # -- context management -------------------------------------------------

    async def maybe_evict(self, conv):
        """Drop thinking blocks once they are far enough behind to be dead weight.

        Age, not size. A note's age is how many tokens have been generated since
        it was written, so a block stays exactly as long as it is still near the
        live end of the conversation.

        Two watermarks, and the gap between them is the whole point. Eviction
        only TRIGGERS once something is EVICT_AGE_TOKENS old, but when it fires
        it clears everything older than EVICT_KEEP_TOKENS - the entire band in
        one go. Evicting at a single threshold would fire continuously, and
        every firing rewrites the prefix and throws away the KV cache from that
        point on. One big rare rewrite costs far less than a steady drip.

        Nothing is summarised. The notes are deleted, the prefix changes, and
        the next request re-prefills from the first evicted position. That cost
        is real and is reported rather than glossed over.
        """
        live = conv.live_notes()
        if not live:
            return
        oldest = max(conv.clock - n.born for _, n in live)
        if oldest < config.EVICT_AGE_TOKENS:
            return

        before = await self._size(conv)
        cache_before = await self.llm.prefix_cache_stats()

        dropped = 0
        for t in conv.turns:
            keep = [n for n in t.notes if conv.clock - n.born < config.EVICT_KEEP_TOKENS]
            dropped += len(t.notes) - len(keep)
            t.notes = keep
        if not dropped:
            return

        yield {
            "type": "evict",
            "notes": dropped,
            "remaining": len(conv.live_notes()),
            "oldest_age": oldest,
            "keep_newer_than": config.EVICT_KEEP_TOKENS,
            "before_tokens": before,
            "after_tokens": await self._size(conv),
            "cache_hit_before": cache_before,
        }

    async def _size(self, conv):
        try:
            return len(await self.llm.tokenize(conv.render()))
        except Exception:
            return len(conv.render()) // 4
