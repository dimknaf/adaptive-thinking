"""The thinking block: fan-out, the checkpointed tree, judging, and the winner.

Engine._run_block is driven directly with a scripted endpoint. Each chunk
request is answered per beam, keyed on the beam's text so far; each judge
prompt is answered per thought, keyed on the thought it quotes. Every
expected number below is derived from the code, and the arithmetic is written
out next to it.
"""

import asyncio

import pytest

from adaptive_thinking import chatml, config, prompts
from adaptive_thinking.engine import Engine
from adaptive_thinking.state import Beam, Block
from fake_endpoint import (
    LABEL_ID,
    FakeVLLM,
    asked_axis,
    chunk,
    closed,
    of_type,
    openings,
    lvl,
    thought_of,
    types,
)

CTX = (chatml.turn("system", "S") + chatml.turn("user", "Q")
       + chatml.assistant_header(think=True))
SEED = config.OPEN_SEED
BLOCK_ID = 7


def run_block(fake, ctx=CTX, seed=SEED, block=None):
    """One _run_block to completion. Returns (events, block)."""
    block = block or Block(id=BLOCK_ID, tail="", crit=None, kind="open")
    engine = Engine(fake)

    async def go():
        return [ev async for ev in engine._run_block(block, ctx, seed=seed)]

    return asyncio.run(go()), block


def fan_out(fake, ctx=CTX, seed=SEED):
    return asyncio.run(Engine(fake)._fan_out(ctx, seed))


def body(prompt, ctx=CTX, seed=SEED):
    """The beam's text so far, from a chunk prompt (ctx + seed + text)."""
    assert prompt.startswith(ctx + seed), prompt[-80:]
    return prompt[len(ctx + seed):]


def scripted(per_beam, limit=6):
    """A chunk script: per_beam(n, text_so_far) answers each live beam at
    checkpoint n. After `limit` checkpoints it raises, so a block that fails
    to end fails its test instead of running forever. `.calls` records how
    many beams each request carried."""
    calls = []

    def chunks(prompts):
        n = len(calls)
        if n >= limit:
            raise AssertionError(f"the block was still running after {limit} checkpoints")
        calls.append(len(prompts))
        return [per_beam(n, body(p)) for p in prompts]

    chunks.calls = calls
    return chunks


def judged_by(levels):
    """score(): judge prompts answered by the first matching prefix of the
    quoted thought, {prefix: level}. Anything else is a test error."""

    def score(prompt):
        family, _ = asked_axis(prompt)
        assert family == "judge", "only thoughts are judged inside a block"
        t = thought_of(prompt)
        for prefix, level in levels.items():
            if t.startswith(prefix):
                return lvl(level(t) if callable(level) else level)
        raise AssertionError(f"no score scripted for thought {t!r}")

    return score


def steps_of(events):
    return of_type(events, "think_step")


# ---------------------------------------------------------------------------
# fan-out
# ---------------------------------------------------------------------------


def test_fan_out_reads_once_when_the_first_read_has_enough_openings(cfg):
    cfg(BEAM_WIDTH=4, THINK_BRANCH_K=5)
    fake = FakeVLLM(fanouts=[openings("Let", "The", "We", "This", "So", "Now")])

    beams = fan_out(fake)

    assert fake.names() == ["fanout"]
    (call,) = fake.of("fanout")
    # min(MAX_LOGPROBS=20, BEAM_WIDTH * THINK_BRANCH_K = 20), one prompt, one token
    assert call == {"prompts": [CTX + SEED], "k": 20, "max_tokens": 1}
    assert [(b.text, b.score) for b in beams] == [
        ("Let", -0.5), ("The", -1.0), ("We", -1.5), ("This", -2.0)]
    assert [(b.uid, b.index, b.parent, b.from_uid, b.tokens, b.last, b.done)
            for b in beams] == [
        (i, i, -1, -1, 1, t, False) for i, t in enumerate(["Let", "The", "We", "This"])]


def test_fan_out_asks_for_width_times_k_capped_at_the_logprobs_limit(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=3)
    fake = FakeVLLM(fanouts=[openings("Let", "We")])
    fan_out(fake)
    assert fake.of("fanout")[0]["k"] == 6

    cfg(BEAM_WIDTH=8, THINK_BRANCH_K=5)
    fake = FakeVLLM(fanouts=[openings(*[f"w{i}" for i in range(8)])])
    fan_out(fake)
    assert fake.of("fanout")[0]["k"] == Engine.MAX_LOGPROBS == 20


def test_fan_out_skips_markers_whitespace_and_punctuation(cfg):
    cfg(BEAM_WIDTH=3, THINK_BRANCH_K=5)
    top = [("!", -0.1), (" ", -0.2), ("</think>", -0.3), (" Let", -0.4),
           ("\n\n", -0.5), ("...", -0.6), ("x</think>", -0.7), ("2", -0.8),
           ("The", -0.9), ("We", -1.0)]
    fake = FakeVLLM(fanouts=[top])

    beams = fan_out(fake)

    # " Let" keeps its leading space; "x</think>" carries a marker, so it is
    # skipped even though "x" alone would qualify; a digit is alphanumeric.
    assert [(b.text, b.score) for b in beams] == [(" Let", -0.4), ("2", -0.8), ("The", -0.9)]
    assert fake.names() == ["fanout"]


def test_a_degenerate_first_read_is_read_again(cfg):
    cfg(BEAM_WIDTH=4, THINK_BRANCH_K=5)
    fake = FakeVLLM(fanouts=[
        [("Let", -0.1), ("!", -2.0)],            # the stub read: one usable opening
        openings("Let", "The", "We", "This"),
    ])

    beams = fan_out(fake)

    assert fake.names() == ["fanout", "fanout"]
    first, second = fake.of("fanout")
    assert first == second  # the same request, sent again
    assert [b.text for b in beams] == ["Let", "The", "We", "This"]
    assert [b.uid for b in beams] == [0, 1, 2, 3]


@pytest.mark.parametrize("reads, kept", [
    ([openings("A1", "A2"), openings("B1")], ["A1", "A2"]),
    ([openings("A1"), openings("B1", "B2")], ["B1", "B2"]),
    # Equally poor: the first read is kept (the comparison is strict).
    ([openings("A1", "A2"), openings("B1", "B2")], ["A1", "A2"]),
], ids=["first-better", "second-better", "tie"])
def test_of_two_poor_reads_the_better_is_kept(cfg, reads, kept):
    cfg(BEAM_WIDTH=4, THINK_BRANCH_K=5)
    fake = FakeVLLM(fanouts=reads)

    beams = fan_out(fake)

    assert fake.names() == ["fanout", "fanout"]
    assert [b.text for b in beams] == kept
    assert [b.uid for b in beams] == list(range(len(kept)))


@pytest.mark.parametrize("read", [
    [("!", -0.1), ("\n", -0.2), ("</think>", -0.3), ("<|im_end|>", -0.4), ("  ", -0.5)],
    [],  # nothing usable came back at all
], ids=["all-junk", "empty"])
def test_no_usable_opening_gives_one_empty_beam(cfg, read):
    cfg(BEAM_WIDTH=4, THINK_BRANCH_K=5)
    fake = FakeVLLM(fanouts=[read, read])

    beams = fan_out(fake)

    assert fake.names() == ["fanout", "fanout"]
    (b,) = beams
    assert (b.text, b.score, b.parent, b.uid, b.from_uid, b.index, b.tokens) == (
        "", 0.0, -1, 0, -1, 0, 0)


def test_a_block_with_no_usable_opening_still_runs_on_one_beam(cfg):
    cfg(BEAM_WIDTH=4, THINK_BRANCH_K=5)
    fake = FakeVLLM(fanouts=[[("!", -0.1)], [("!", -0.1)]],
                    chunks=[[closed("I think it is 4.")]],
                    score=judged_by({"I think": 3}))

    events, block = run_block(fake)

    (frontier,) = of_type(events, "frontier")
    assert (frontier["started"], frontier["width"]) == (1, 4)
    # The beam's prompt is the seed alone: no opening token was chosen.
    assert fake.of("chunk")[0]["prompts"] == [CTX + SEED]
    assert block.monologue == "I think it is 4."
    assert block.winner_score == 0.75


# ---------------------------------------------------------------------------
# the block's events
# ---------------------------------------------------------------------------


def two_checkpoints(fake_llm=FakeVLLM):
    """Let continues then finishes; We finishes at once. Let is judged best."""
    return fake_llm(
        fanouts=[openings("Let", "We")],
        chunks=[
            [chunk(" add 3 and", lp=-2.0, top=[(" and", -0.1), (" or", -1.0)]),
             closed(" need both cases.", lp=-3.0)],
            [closed(" 1 to get 4.", lp=-1.0)],
        ],
        score=judged_by({"Let": 4, "We": 2}),
    )


def test_a_block_emits_its_events_in_order(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=2)
    fake = two_checkpoints()

    events, block = run_block(fake)

    assert types(events) == [
        "block_open", "frontier",
        "think_token", "think_token", "think_step", "standings",
        "think_token", "think_step", "standings",
        "monologue",
    ]
    assert all(e.get("block", e.get("id")) == BLOCK_ID for e in events)
    assert events[0] == {"type": "block_open", "id": BLOCK_ID, "kind": "open",
                         "width": 2, "branch_k": 2, "tail": "", "crit": None,
                         "gate": None, "seed": SEED}
    assert events[1] == {
        "type": "frontier", "block": BLOCK_ID, "started": 2, "width": 2,
        "beams": [
            {"i": 0, "uid": 0, "from_uid": -1, "parent": -1, "tok": "Let",
             "text": "Let", "score": -0.5, "total": -0.5, "p": None, "done": False},
            {"i": 1, "uid": 1, "from_uid": -1, "parent": -1, "tok": "We",
             "text": "We", "score": -1.0, "total": -1.0, "p": None, "done": False},
        ],
    }
    # Streamed chunk text is attributed to the lineage writing it.
    assert [(e["uid"], e["text"]) for e in of_type(events, "think_token")] == [
        (0, " add 3 and"), (1, " need both cases."), (0, " 1 to get 4.")]
    assert of_type(events, "monologue") == [
        {"type": "monologue", "block": BLOCK_ID, "text": "Let add 3 and 1 to get 4."}]

    assert block.monologue == "Let add 3 and 1 to get 4."
    assert block.winner_score == 1.0
    assert block.steps == 2
    assert block.finished_count == 2
    # fan-out + 2 chunk requests + 2 judge requests (labels is not a request)
    assert block.requests == 5
    assert fake.names() == ["fanout", "chunk", "labels", "answer_probs",
                            "chunk", "labels", "answer_probs"]
    fake.assert_consumed()


def test_think_step_and_standings_report_the_tree(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=2)
    fake = two_checkpoints()

    events, _ = run_block(fake)

    s0, s1 = steps_of(events)
    # Checkpoint 0: Let branches into " and" / " or"; We closes itself.
    #   Let + " and": score -0.5 - 2.0 - (-0.1) + (-0.1) = -2.5 over 4 tokens
    #   We, finished: score -1.0 - 3.0 = -4.0 over 1 + 17 // 4 = 5 tokens
    assert s0["step"] == 0 and s0["considered"] == 2 and s0["ranked_by"] == "logprob"
    assert [(b["uid"], b["text"], b["tok"], b["score"], b["total"]) for b in s0["beams"]] == [
        (0, "Let add 3 and", " and", -0.625, -2.5)]
    assert [(b["uid"], b["text"], b["done"], b["score"]) for b in s0["pool"]] == [
        (1, "We need both cases.", True, -0.8)]
    # Finishing is not being dropped: We is in the pool, not in `ended`.
    assert s0["ended"] == [] and s0["retired"] == [] and s0["entered"] == []
    assert (s0["leader"], s0["dry"], s0["closed"]) == (None, 0, 0)
    assert (s0["dropped"], s0["distinct"], s0["backfilled"]) == (1, 1, 0)

    assert s1["considered"] == 0 and s1["beams"] == []
    assert [b["text"] for b in s1["pool"]] == ["We need both cases.", "Let add 3 and 1 to get 4."]
    assert s1["leader"] == 1.0  # set by checkpoint 0's standings

    st0, st1 = of_type(events, "standings")
    # Live and finished are standardised separately; each group has one
    # thought here, so both `overall` scores are 0 and the list order stands.
    assert [(r["uid"], r["rank"], r["medal"], r["done"], r["overall"], r["raw"])
            for r in st0["ranked"]] == [(0, 1, 1, False, 0.0, 1.0), (1, 2, 2, True, 0.0, 0.5)]
    assert (st0["live"], st0["finished"]) == (1, 1)
    assert st0["ranked"][0]["axes"] == {a: 1.0 for a in prompts.AXES}
    # Checkpoint 1: two finished thoughts, standardised against each other.
    assert [(r["uid"], r["overall"], r["raw"]) for r in st1["ranked"]] == [
        (0, 2.3, 1.0), (1, -2.3, 0.5)]
    assert (st1["live"], st1["finished"]) == (0, 2)


def test_a_block_without_a_seed_opens_with_the_inline_seed(cfg):
    cfg(BEAM_WIDTH=1, THINK_BRANCH_K=2)
    fake = FakeVLLM(fanouts=[openings("Let")], chunks=[[closed(" me check.")]],
                    score=judged_by({"Let": 2}))
    block = Block(id=3, tail="so far", crit=0.9, gate={"crit": 0.9})

    events, _ = run_block(fake, seed=None, block=block)

    assert events[0]["seed"] == config.INLINE_SEED
    assert (events[0]["kind"], events[0]["tail"], events[0]["crit"], events[0]["gate"]) == (
        "gate", "so far", 0.9, {"crit": 0.9})
    assert fake.of("fanout")[0]["prompts"] == [CTX + config.INLINE_SEED]
    assert fake.of("chunk")[0]["prompts"] == [CTX + config.INLINE_SEED + "Let"]


# ---------------------------------------------------------------------------
# the width never grows
# ---------------------------------------------------------------------------


def test_the_frontier_never_exceeds_beam_width_and_shrinks_only_by_finishing(cfg):
    cfg(BEAM_WIDTH=3, THINK_BRANCH_K=3)

    def per_beam(n, text):
        if n == 3 or (n == 1 and text.startswith("We")):
            return closed(" done.")
        return chunk(" step.", lp=-1.0,
                     top=[(" step.", -0.1), (" pause.", -0.5), (" halt.", -1.0)])

    chunks = scripted(per_beam)
    sizes = chunks.calls
    fake = FakeVLLM(fanouts=[openings("Let", "The", "We")], chunks=chunks)

    events, block = run_block(fake)

    steps = steps_of(events)
    width = config.BEAM_WIDTH
    assert [len(s["beams"]) for s in steps] == [3, 2, 2, 0]
    assert [len(s["pool"]) for s in steps] == [0, 1, 1, 3]
    assert [s["considered"] for s in steps] == [9, 6, 6, 0]
    assert sizes == [3, 3, 2, 2]  # live beams sent per chunk request
    for s, live in zip(steps, sizes):
        assert len(s["beams"]) <= width
        assert s["considered"] <= live * config.THINK_BRANCH_K
        # The code's own width: BEAM_WIDTH less finished thoughts and closed slots.
        assert len(s["beams"]) <= max(1, width - len(s["pool"]) - s["closed"])
        # Nothing was churned here, so every slot is live or finished.
        assert len(s["beams"]) + len(s["pool"]) == width
        # One survivor per lineage: no sibling backfill was needed.
        assert s["distinct"] == len(s["beams"]) and s["backfilled"] == 0
        assert s["ended"] == []
    # Lineages keep their identity across checkpoints.
    assert [[b["uid"] for b in s["beams"]] for s in steps] == [[0, 1, 2], [0, 1], [0, 1], []]
    assert block.finished_count == 3
    fake.assert_consumed()


# ---------------------------------------------------------------------------
# candidate scores
# ---------------------------------------------------------------------------

CHUNK40 = " carry the one and add the tens, then: x"


def one_opening_two_candidates(ctx=CTX):
    """One opening (Let, lp -1.0, 1 token) that writes CHUNK40 (lp -4.0) and
    proposes " x" (-0.2, the token it actually wrote) or " y" (-1.2)."""
    return FakeVLLM(
        fanout=lambda prompts, k: [("Let", -1.0)],
        chunks=[
            [chunk(CHUNK40, lp=-4.0, top=[(" x", -0.2), (" y", -1.2)])],
            lambda prompts: [closed(" = 12.") for _ in prompts],
        ],
        score=judged_by({"Let": 2}),
    )


def test_candidates_replace_the_last_token_and_are_length_normalised(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=2)
    assert len(CHUNK40) == 40 and CHUNK40.endswith(" x")
    fake = one_opening_two_candidates()

    events, _ = run_block(fake)

    # Only one usable opening, so the fan-out read twice and the block runs
    # one lineage; its two candidates fill the width as siblings.
    assert fake.names()[:2] == ["fanout", "fanout"]
    s0 = steps_of(events)[0]
    # tokens: 1 (opening) + 40 // 4 (chunk) + 1 (the replaced last token) = 12
    #   " x": -1.0 + -4.0 - (-0.2) + (-0.2) = -5.0 -> -5.0 / 12 = -0.417
    #   " y": -1.0 + -4.0 - (-0.2) + (-1.2) = -6.0 -> -6.0 / 12 = -0.5
    # The chunk ends with the token it proposes, so " x" is the chunk itself,
    # not the chunk with " x" written twice.
    stem = "Let" + CHUNK40[:-2]
    assert [(b["uid"], b["from_uid"], b["parent"], b["tok"], b["text"], b["score"], b["total"])
            for b in s0["beams"]] == [
        (0, 0, 0, " x", stem + " x", -0.417, -5.0),
        (1, 0, 0, " y", stem + " y", -0.5, -6.0),   # a split: a new identity
    ]
    assert (s0["distinct"], s0["backfilled"]) == (1, 1)
    assert fake.of("chunk")[1]["prompts"] == [CTX + SEED + stem + " x",
                                              CTX + SEED + stem + " y"]


def test_copying_the_context_is_penalised(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=2, COPY_PENALTY=0.6, COPY_NGRAM=24)
    # The last 24 characters of the " x" candidate already sit in the context.
    ctx = CTX + "Earlier:" + CHUNK40
    fake = one_opening_two_candidates(ctx)

    events, _ = run_block(fake, ctx=ctx)

    s0 = steps_of(events)[0]
    # " x": -5.0 / 12 - 0.6 = -1.017, now ranked below " y".
    assert [(b["tok"], b["score"], b["total"]) for b in s0["beams"]] == [
        (" y", -0.5, -6.0), (" x", -1.017, -5.0)]


def test_the_copy_penalty_can_be_switched_off(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=2, COPY_PENALTY=0.0)
    ctx = CTX + "Earlier:" + CHUNK40
    fake = one_opening_two_candidates(ctx)

    events, _ = run_block(fake, ctx=ctx)

    s0 = steps_of(events)[0]
    assert [(b["tok"], b["score"]) for b in s0["beams"]] == [(" x", -0.417), (" y", -0.5)]


def test_a_thought_that_starts_copying_the_nudge_is_cut_and_finished(cfg):
    cfg(BEAM_WIDTH=1, THINK_BRANCH_K=2)
    text = " see: 4. " + prompts.NUDGE_PREFIX + " So x"
    fake = FakeVLLM(
        fanouts=[openings("Let")],
        # Not stopped and no marker: only the nudge ends it.
        chunks=[[chunk(text, lp=-3.0, top=[(" x", -0.1), (" z", -2.0)])]],
        score=judged_by({"Let": 3}),
    )

    events, block = run_block(fake)

    (s0,) = steps_of(events)
    assert s0["beams"] == []
    assert [(b["text"], b["done"], b["tok"]) for b in s0["pool"]] == [
        ("Let see: 4. ", True, " x"), ("Let see: 4. ", True, " z")]
    assert block.monologue == "Let see: 4. "
    assert of_type(events, "monologue")[0]["text"] == "Let see: 4."
    assert fake.names().count("chunk") == 1
    fake.assert_consumed()


# ---------------------------------------------------------------------------
# when a block stops
# ---------------------------------------------------------------------------

FOUR = ("Ann", "Bob", "Cy", "Dee")


def three_finish_one_writes(levels, dee_steps=1):
    """Ann, Bob and Cy close in the first chunk; Dee keeps writing for
    `dee_steps` chunks, then closes. Judged by `levels` {name: level}."""

    def per_beam(n, text):
        if not text.startswith("Dee"):
            return closed(" is done here.")
        if n < dee_steps:
            return chunk(" keeps going", lp=-1.0, top=[(" going", -0.1), (" on", -1.0)])
        return closed(" and is done.")

    return FakeVLLM(fanouts=[openings(*FOUR)], chunks=scripted(per_beam),
                    score=judged_by(levels))


def test_the_block_stops_once_a_clear_leader_has_finished_and_no_live_beam_contends(cfg):
    cfg(BEAM_WIDTH=4, THINK_BRANCH_K=2, STOP_AFTER_FINISHED=3, STOP_MARGIN=0.35,
        CHURN_MARGIN=0.12)
    # Finished: Ann raw 1.0, Bob and Cy 0.0. Standardised within the finished
    # group, Ann leads Bob by 4.6 >= STOP_MARGIN. Dee (live) scores 0.25, which
    # is 0.75 behind the leader: not in contention.
    fake = three_finish_one_writes({"Ann": 4, "Bob": 0, "Cy": 0, "Dee": 1})

    events, block = run_block(fake)

    assert fake.names().count("chunk") == 1  # Dee is never advanced again
    st = of_type(events, "standings")[-1]
    assert (st["live"], st["finished"]) == (1, 3)
    assert block.monologue == "Ann is done here."
    assert block.finished_count == 3
    # Dee is still in the block's record, unfinished.
    assert [(b.text, b.done) for b in block.beams][-1] == ("Dee keeps going", False)


@pytest.mark.parametrize("levels", [
    # Dee scores as well as the leader: a live beam still in contention is
    # never cut off, however many others have finished.
    {"Ann": 4, "Bob": 0, "Cy": 0, "Dee": 4},
    # Ann and Bob tie among the finished: no clear leader, so the rest may land.
    {"Ann": 4, "Bob": 4, "Cy": 0, "Dee": 1},
], ids=["live-beam-in-contention", "no-clear-leader"])
def test_the_block_does_not_stop_early_otherwise(cfg, levels):
    cfg(BEAM_WIDTH=4, THINK_BRANCH_K=2, STOP_AFTER_FINISHED=3, STOP_MARGIN=0.35,
        CHURN_MARGIN=0.12)
    fake = three_finish_one_writes(levels)

    events, block = run_block(fake)

    assert fake.names().count("chunk") == 2  # Dee writes on and closes itself
    assert block.finished_count == 4
    assert of_type(events, "standings")[-1]["live"] == 0


LONG = " works through every case one at a time and writes each one down"


def one_finishes_one_rambles(dee_level):
    def per_beam(n, text):
        if text.startswith("Ann"):
            return closed(" ok.")                     # 1 + 4 // 4 = 2 tokens
        if n == 0:
            return chunk(LONG, lp=-1.0, top=[(" down", -0.1), (" up", -1.0)])
        return closed(" and stops.")

    return FakeVLLM(fanouts=[openings("Ann", "Dee")], chunks=scripted(per_beam),
                    score=judged_by({"Ann": 4, "Dee": dee_level}))


def test_a_long_live_beam_out_of_contention_is_dropped(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=2, RAMBLE_RATIO=2.5, CHURN_MARGIN=0.12)
    # Dee: 1 + len(LONG) // 4 + 1 = 18 tokens > 2.5 x Ann's 2, and it scores
    # 0.0 against a leader of 1.0.
    fake = one_finishes_one_rambles(dee_level=0)

    events, block = run_block(fake)

    assert fake.names().count("chunk") == 1
    assert [b.text for b in block.beams] == ["Ann ok."]
    assert block.monologue == "Ann ok."


def test_a_long_live_beam_that_still_contends_keeps_going(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=2, RAMBLE_RATIO=2.5, CHURN_MARGIN=0.12)
    fake = one_finishes_one_rambles(dee_level=4)

    events, block = run_block(fake)

    assert fake.names().count("chunk") == 2
    assert block.finished_count == 2


# ---------------------------------------------------------------------------
# the churn, inside a block
# ---------------------------------------------------------------------------


def churn_block():
    """Ann and Bob write two chunks, then close. Ann is judged 1.0, Bob 0.25,
    so at the second checkpoint Bob is 0.75 behind the leader."""

    def per_beam(n, text):
        if n == 2:
            return closed(" end.")
        piece = " one x" if n == 0 else " two x"
        return chunk(piece, lp=-1.0, top=[(" x", -0.1), (" y", -0.9)])

    return FakeVLLM(fanouts=[openings("Ann", "Bob")], chunks=scripted(per_beam),
                    score=judged_by({"Ann": 4, "Bob": 1}))


def test_a_clearly_behind_conversation_is_replaced_by_a_variation_of_the_best(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=2, CHURN_MARGIN=0.12, CHURN_PATIENCE=2)
    fake = churn_block()

    events, block = run_block(fake)

    s0, s1, s2 = steps_of(events)
    assert s0["retired"] == [] and s0["leader"] is None
    # Checkpoint 1: leader 1.0, dry 0 < CHURN_PATIENCE, so Bob's slot is
    # refilled with Ann's discarded sibling, as a new lineage off Ann's.
    # Bob's `overall` (-2.3) is the one checkpoint 0 gave him against Ann.
    assert s1["retired"] == [{"uid": 1, "overall": -2.3, "replaced": True}]
    assert s1["entered"] == [{"uid": 2, "from_uid": 0, "tok": " y"}]
    assert s1["ended"] == [1] and s1["closed"] == 0
    assert [(b["uid"], b["text"]) for b in s1["beams"]] == [
        (0, "Ann one x two x"), (2, "Ann one x two y")]
    assert (s1["leader"], s1["dry"]) == (1.0, 0)
    assert len(s2["pool"]) == 2
    assert block.finished_count == 2


def test_a_clearly_behind_conversation_closes_its_slot_once_the_block_is_dry(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=2, CHURN_MARGIN=0.12, CHURN_PATIENCE=0)
    fake = churn_block()

    events, block = run_block(fake)

    s0, s1, s2 = steps_of(events)
    assert s1["retired"] == [{"uid": 1, "overall": -2.3, "replaced": False}]
    assert s1["entered"] == []
    assert s1["closed"] == 1
    assert [b["uid"] for b in s1["beams"]] == [0]
    # The closed slot stays closed.
    assert s2["closed"] == 1 and len(s2["pool"]) == 1
    assert block.finished_count == 1


# ---------------------------------------------------------------------------
# the winner
# ---------------------------------------------------------------------------


def test_a_finished_thought_too_short_to_count_loses_to_a_longer_one(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=2)
    long_end = " is correct, because both cases lead to the same answer in the end."

    def per_beam(n, text):
        if text.startswith("Ann"):
            return closed(" ok.")
        if n == 0:
            return chunk(" says", lp=-1.0, top=[(" says", -0.1), (" said", -1.0)])
        return closed(long_end)

    fake = FakeVLLM(fanouts=[openings("Ann", "Bob")], chunks=scripted(per_beam),
                    score=judged_by({"Ann": 4, "Bob": 2}))

    events, block = run_block(fake)

    # "Ann ok." (7 characters) is under a tenth of the longest thought, so it
    # does not qualify, whatever its score: Bob wins at raw 0.5 against 1.0.
    bob = "Bob says" + long_end
    assert len("Ann ok.") < len(bob) / 10
    assert block.monologue == bob
    assert block.winner_score == 0.5


def test_among_qualifying_finished_thoughts_the_best_raw_wins(cfg):
    cfg(BEAM_WIDTH=3, THINK_BRANCH_K=2)
    fake = FakeVLLM(
        fanouts=[openings("Ann", "Bob", "Cy")],
        chunks=[[closed(" thinks it is 3."), closed(" thinks it is 4."),
                 closed(" thinks it is 5.")]],
        score=judged_by({"Ann": 2, "Bob": 4, "Cy": 3}),
    )

    _, block = run_block(fake)

    assert block.monologue == "Bob thinks it is 4."
    assert block.winner_score == 1.0


def test_equal_scores_go_to_the_first_finished_thought(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=2)
    fake = FakeVLLM(
        fanouts=[openings("Ann", "Bob")],
        chunks=[[closed(" thinks it is 3."), closed(" thinks it is 4.")]],
        score=judged_by({"Ann": 3, "Bob": 3}),
    )

    _, block = run_block(fake)

    assert block.monologue == "Ann thinks it is 3."


def test_when_no_finished_thought_qualifies_the_best_raw_of_all_wins_even_a_live_one(cfg):
    cfg(BEAM_WIDTH=4, THINK_BRANCH_K=2, STOP_AFTER_FINISHED=3, STOP_MARGIN=0.35,
        CHURN_MARGIN=0.12)
    long_piece = (" goes through the cases one by one: if Ann is married she looks"
                  " at George, and if she is not then Jack looks at her")

    def per_beam(n, text):
        if n == 0:
            return chunk(" is", lp=-1.0, top=[(" is", -0.1), (" was", -1.0)])
        if not text.startswith("Dee"):
            return closed(" ok.")
        return chunk(long_piece, lp=-5.0, top=[(" her", -0.1), (" him", -2.0)])

    def dee(t):
        # 1.0 at the first checkpoint, which makes it the block's leader; 0.5
        # once it has written the long piece, 0.5 behind that leader.
        return 4 if len(t) < 20 else 2

    fake = FakeVLLM(fanouts=[openings(*FOUR)], chunks=scripted(per_beam),
                    score=judged_by({"Ann": 1, "Bob": 0, "Cy": 0, "Dee": dee}))

    events, block = run_block(fake)

    # Checkpoint 1: Ann, Bob and Cy finish with a clear leader among them, and
    # Dee is out of contention, so the block stops with Dee still writing.
    assert fake.names().count("chunk") == 2
    assert of_type(events, "standings")[-1]["live"] == 1
    # Every finished thought is under a tenth of Dee's length, so none
    # qualifies and the best raw of ALL contenders wins: Dee, live, at 0.5
    # against Ann's 0.25.
    dee_text = "Dee is" + long_piece
    assert all(len(t) < len(dee_text) / 10 for t in ("Ann is ok.", "Bob is ok.", "Cy is ok."))
    assert block.monologue == dee_text
    assert block.winner_score == 0.5
    assert block.finished_count == 3


# ---------------------------------------------------------------------------
# judging on the axes
# ---------------------------------------------------------------------------


def test_score_axes_standardises_each_group_separately_in_one_request():
    def score(prompt):
        return {"F1": lvl(4), "F2": lvl(2), "L": lvl(0)}[thought_of(prompt)]

    fake = FakeVLLM(score=score)
    f1, f2, live = (Beam(text=t, score=0.0, parent=0) for t in ("F1", "F2", "L"))

    asyncio.run(Engine(fake)._score_axes(CTX, [live], [f1, f2], []))

    assert fake.names() == ["labels", "answer_probs"]
    (ap,) = fake.of("answer_probs")
    assert ap["ids"] == [LABEL_ID + i for i in range(5)]
    assert len(ap["prompts"]) == 3 * len(prompts.AXES)
    assert ap["prompts"][0] == prompts.axis_prompt(CTX, "L", next(iter(prompts.AXES)))
    # raw is absolute and comparable across groups.
    assert (f1.raw, f2.raw, live.raw) == (pytest.approx(1.0), pytest.approx(0.5), 0.0)
    # overall is standardised WITHIN a group: z = +-0.5 on every axis, weights
    # summing to 4.6. A group of one has nothing to be compared with.
    assert f1.overall == pytest.approx(2.3) and f2.overall == pytest.approx(-2.3)
    assert live.overall == 0.0
    for a in prompts.AXES:
        assert (f1.axes[a], f1.axes[a + "_z"]) == (1.0, 0.5)
        assert (f2.axes[a], f2.axes[a + "_z"]) == (0.5, -0.5)


def test_rank_averages_both_option_orders():
    # Engine._rank is not called by the engine at present; this covers its
    # arithmetic only. Forward order reads [0.8, 0.2]; the reversed prompt
    # reads [0.7, 0.3] over (c1, c0). Averaged: c0 0.55, c1 0.45.
    first, second = "first idea", "second idea"

    def score(prompt):
        return [0.8, 0.2] if prompt.index(first) < prompt.index(second) else [0.7, 0.3]

    fake = FakeVLLM(score=score)
    c0, c1 = (Beam(text=t, score=0.0, parent=0) for t in (first, second))

    ranked = asyncio.run(Engine(fake)._rank(CTX, [c0, c1], debias=True))

    assert ranked[0] is c0 and ranked[1] is c1
    assert c0.p == pytest.approx(0.55) and c1.p == pytest.approx(0.45)
    (ap,) = fake.of("answer_probs")
    assert len(ap["prompts"]) == 2  # both orders in one request
