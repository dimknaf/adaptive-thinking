"""The gate: which streamed positions may open a block, and which one does.

Engine._choose drops every position that fails one of six rules (entropy floor,
fewer than two distinct alternatives, MIN_GAP_TOKENS since the last block,
WARMUP_TOKENS, text already written, not at a sentence or clause boundary),
keeps the PROBE_CANDIDATES highest-entropy survivors, scores them on the gate
axes in one batched request, and picks the best one whose rounded crit is at or
above GATE_THRESHOLD.

Each rule is tested on its own, with a position that passes every other rule,
so a failure names the rule that changed.
"""

import asyncio
import json
import math

import pytest

from adaptive_thinking import chatml, config, prompts
from adaptive_thinking.engine import Engine
from adaptive_thinking.events import _window_event
from adaptive_thinking.gate import _Cand
from adaptive_thinking.state import Note
from adaptive_thinking.text import _at_boundary
from fake_endpoint import (
    LABEL_ID,
    FakeVLLM,
    Stream,
    asked_axis,
    context_of,
    ln,
    lvl,
    of_type,
    quiet,
    run_turn,
    step,
    types,
)

PREFIX = (chatml.turn("system", "S") + chatml.turn("user", "Q")
          + chatml.assistant_header())


def hot(text, p=0.5):
    """A token with two alternatives. Entropy: ln 2 = 0.693 at p=0.5, 0.673 at
    0.6, 0.611 at 0.7, 0.500 at 0.8 - all above the default floor of 0.4."""
    return step(text, top=[(text, ln(p)), (text + "~", ln(1 - p))])


def entropy2(p):
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def cand(idx, off, st):
    return _Cand(idx, off, st)


def choose(fake, window, vis, last_block_at=0, notes=()):
    engine = Engine(fake)
    return asyncio.run(
        engine._choose(list(window), last_block_at, PREFIX, vis, list(notes)))


def high(prompt):
    """Every gate axis at its top level: crit 1.0."""
    assert asked_axis(prompt)[0] == "gate", prompt
    return lvl(4)


@pytest.fixture
def gate(cfg):
    """The two position rules switched off, so a test can switch one back on,
    and every other gate knob pinned at its code default."""
    return cfg(WARMUP_TOKENS=0, MIN_GAP_TOKENS=0, ENTROPY_FLOOR=0.4,
               PROBE_CANDIDATES=6, GATE_THRESHOLD=0.55, PROBE_ENABLED=True,
               GATE_LOG_YESNO=False)


# A position that passes every rule: " They" starts a new sentence.
VIS = "Trains leave at 09:35. They arrive at 13:12."
OFF = VIS.index(" They")
IDX = 5


def base(st=None, idx=IDX, off=OFF):
    return cand(idx, off, st or hot(" They"))


def assert_rejected(fake, got):
    assert got is None
    # Nothing survived, so nothing was asked.
    assert fake.calls == []


def assert_accepted(fake, got, c):
    assert got is not None
    assert [x for x, _ in got["probed"]] == [c]
    assert got["pick"][0] is c


# ---------------------------------------------------------------------------
# a position that passes every rule
# ---------------------------------------------------------------------------


def test_a_position_that_passes_every_rule_is_scored_and_opens(gate):
    fake = FakeVLLM(score=high)
    c = base()

    got = choose(fake, [c], VIS)

    assert_accepted(fake, got, c)
    assert got["pick"][1] == {"crit": 1.0, "stakes": 1.0, "difficulty": 1.0,
                              "commitment": 1.0}
    # One labels lookup and ONE batched request: one prompt per gate axis, all
    # asked at the position where the candidate token would have been written.
    assert fake.names() == ["labels", "answer_probs"]
    assert fake.of("labels") == [{"n": 5}]
    (ap,) = fake.of("answer_probs")
    assert ap["ids"] == [LABEL_ID + i for i in range(5)]
    assert [asked_axis(p) for p in ap["prompts"]] == [
        ("gate", a) for a in prompts.GATE_AXES]
    assert {context_of(p) for p in ap["prompts"]} == {PREFIX + VIS[:OFF]}


# ---------------------------------------------------------------------------
# the rejection rules, one at a time
# ---------------------------------------------------------------------------


def test_entropy_below_the_floor_is_rejected(gate):
    low = hot(" They", p=0.95)  # two alternatives, entropy 0.199
    assert len(low.alternatives) == 2
    assert low.entropy < config.ENTROPY_FLOOR
    fake = FakeVLLM(score=high)
    assert_rejected(fake, choose(fake, [base(low)], VIS))


def test_entropy_exactly_at_the_floor_is_accepted(cfg, gate):
    st = hot(" They", p=0.8)
    cfg(ENTROPY_FLOOR=st.entropy)
    fake = FakeVLLM(score=high)
    c = base(st)
    assert_accepted(fake, choose(fake, [c], VIS), c)

    cfg(ENTROPY_FLOOR=st.entropy + 1e-9)
    fake = FakeVLLM(score=high)
    assert_rejected(fake, choose(fake, [base(st)], VIS))


@pytest.mark.parametrize("top", [
    # One token and residual mass: entropy ln 2, but a single alternative.
    [(" They", ln(0.5))],
    # An empty-string alternative does not count as one.
    [(" They", ln(0.5)), ("", ln(0.5))],
], ids=["single-token", "empty-alternative"])
def test_fewer_than_two_distinct_alternatives_is_rejected(gate, top):
    st = step(" They", top=top)
    assert st.entropy == pytest.approx(math.log(2))
    assert st.alternatives == [" They"]
    fake = FakeVLLM(score=high)
    assert_rejected(fake, choose(fake, [base(st)], VIS))


def test_min_gap_after_the_last_block(cfg, gate):
    cfg(MIN_GAP_TOKENS=12)
    fake = FakeVLLM(score=high)
    assert_rejected(fake, choose(fake, [base(idx=21)], VIS, last_block_at=10))

    fake = FakeVLLM(score=high)
    c = base(idx=22)  # exactly MIN_GAP_TOKENS after the block: allowed
    assert_accepted(fake, choose(fake, [c], VIS, last_block_at=10), c)


def test_warmup_tokens(cfg, gate):
    cfg(WARMUP_TOKENS=28)
    fake = FakeVLLM(score=high)
    assert_rejected(fake, choose(fake, [base(idx=27)], VIS))

    fake = FakeVLLM(score=high)
    c = base(idx=28)
    assert_accepted(fake, choose(fake, [c], VIS), c)


SENTENCE = "The first train leaves at 09:35 and takes two hours. "
OTHER = "The second one leaves at 10:05 and takes ninety minutes. "


def test_text_already_written_is_rejected(gate):
    # The 48 characters before the position already appear earlier: the model
    # is restating itself, and a block there would deepen the loop.
    vis = SENTENCE + SENTENCE + "So"
    off = 2 * len(SENTENCE)
    assert len(SENTENCE) >= 48 and _at_boundary(vis, off)
    fake = FakeVLLM(score=high)
    assert_rejected(fake, choose(fake, [base(hot(" So"), off=off)], vis))


def test_new_text_of_the_same_length_is_accepted(gate):
    vis = SENTENCE + OTHER + "So"
    off = len(SENTENCE + OTHER)
    fake = FakeVLLM(score=high)
    c = base(hot("So"), off=off)
    assert_accepted(fake, choose(fake, [c], vis), c)


@pytest.mark.parametrize("before, ok", [
    ("", True),                        # the very start of the answer
    ("Trains leave at 09:35.", True),
    ("Is it late?", True),
    ("Wait!", True),
    ("Two cases:", True),
    ("the first; ", True),
    ("Step one\n", True),              # a line break counts
    ("The value is 3.", True),         # any full stop counts, even in a number
    ("The value is 3.14", False),
    ("Trains leave at 09:35,", False),
    ("Trains leave at", False),
])
def test_only_a_sentence_or_clause_boundary_can_open_a_block(gate, before, ok):
    vis = before + " next"
    fake = FakeVLLM(score=high)
    c = base(hot(" next"), off=len(before))
    got = choose(fake, [c], vis)
    if ok:
        assert_accepted(fake, got, c)
    else:
        assert_rejected(fake, got)


# ---------------------------------------------------------------------------
# which survivors are probed, and which one opens
# ---------------------------------------------------------------------------

VIS4 = "One. Two. Three. Four. Five."


def at(word, p, idx):
    return cand(idx, VIS4.index(" " + word), hot(" " + word, p))


def window4():
    # In text order; entropies 0.500, 0.693, 0.611, 0.673.
    return [at("Two", 0.8, 1), at("Three", 0.5, 2), at("Four", 0.7, 3),
            at("Five", 0.6, 4)]


def test_survivors_are_probed_highest_entropy_first(gate):
    fake = FakeVLLM()  # uniform: crit 0.5 everywhere, nothing opens
    w = window4()

    got = choose(fake, w, VIS4)

    by_entropy = [w[1], w[3], w[2], w[0]]  # Three, Five, Four, Two
    assert [c for c, _ in got["probed"]] == by_entropy
    assert [s["crit"] for _, s in got["probed"]] == [0.5] * 4
    assert got["pick"] is None
    (ap,) = fake.of("answer_probs")
    n = len(prompts.GATE_AXES)
    assert [context_of(p) for p in ap["prompts"][::n]] == [
        PREFIX + VIS4[: c.off] for c in by_entropy]


def test_only_probe_candidates_positions_are_probed(cfg, gate):
    cfg(PROBE_CANDIDATES=2)
    fake = FakeVLLM()
    w = window4()

    got = choose(fake, w, VIS4)

    assert [c for c, _ in got["probed"]] == [w[1], w[3]]  # Three, Five
    (ap,) = fake.of("answer_probs")
    assert len(ap["prompts"]) == 2 * len(prompts.GATE_AXES)


def test_the_most_critical_position_opens_not_the_most_uncertain(gate):
    w = window4()[:3]  # Two (0.500), Three (0.693), Four (0.611)
    level = {w[0].off: 4, w[1].off: 1, w[2].off: 3}

    def score(prompt):
        return lvl(level[len(context_of(prompt)) - len(PREFIX)])

    fake = FakeVLLM(score=score)
    got = choose(fake, w, VIS4)

    # Probed in crit order, best first; the pick is the best.
    assert [(c.step.text, s["crit"]) for c, s in got["probed"]] == [
        (" Two", 1.0), (" Four", 0.75), (" Three", 0.25)]
    assert got["pick"][0] is w[0]


SPLIT = {"stakes": 4, "difficulty": 2, "commitment": 0}


def split_score(prompt):
    """stakes E, difficulty C, commitment A: crit (1.4 + 0.6 + 0) / 3.6."""
    family, axis = asked_axis(prompt)
    assert family == "gate"
    return lvl(SPLIT[axis])


@pytest.mark.parametrize("threshold, opens", [
    (0.55, True),
    # 0.5556 is ABOVE the unrounded crit 0.55555..., so opening here shows the
    # comparison is made on the crit rounded to four places.
    (0.5556, True),
    (0.5557, False),
])
def test_the_threshold_is_compared_with_the_rounded_crit(cfg, gate, threshold, opens):
    cfg(GATE_THRESHOLD=threshold)
    fake = FakeVLLM(score=split_score)
    c = base()

    got = choose(fake, [c], VIS)

    expected = {"crit": 0.5556, "stakes": 1.0, "difficulty": 0.5, "commitment": 0.0}
    assert 2.0 / 3.6 < expected["crit"]
    assert got["probed"] == [(c, expected)]
    if opens:
        assert got["pick"] == (c, expected)
    else:
        assert got["pick"] is None


def test_the_gate_reads_only_notes_at_or_before_the_position(gate):
    notes = [
        Note(off=-1, text="HEAD", born=0),     # the header's, not the body's
        Note(off=3, text="<A>", born=0),
        Note(off=OFF, text="<B>", born=0),     # exactly at the position
        Note(off=OFF + 1, text="<C>", born=0),  # after it
    ]
    fake = FakeVLLM(score=high)

    choose(fake, [base()], VIS, notes=notes)

    (ap,) = fake.of("answer_probs")
    want = PREFIX + VIS[:3] + "<A>" + VIS[3:OFF] + "<B>"
    assert {context_of(p) for p in ap["prompts"]} == {want}


def test_the_yes_no_probe_is_recorded_and_decides_nothing(cfg, gate):
    cfg(GATE_LOG_YESNO=True)
    c = base()

    fake = FakeVLLM(score=high, yes=lambda prompt: 0.0)
    got = choose(fake, [c], VIS)
    assert got["pick"] == (c, {"crit": 1.0, "stakes": 1.0, "difficulty": 1.0,
                               "commitment": 1.0, "p_yes": 0.0})
    assert fake.names() == ["labels", "answer_probs", "p_yes"]
    assert fake.of("p_yes") == [{"prompts": [
        prompts.probe_prompt(PREFIX + VIS[:OFF], prompts.GATE_PROBE)]}]

    fake = FakeVLLM(yes=lambda prompt: 1.0)  # uniform axes: crit 0.5
    got = choose(fake, [c], VIS)
    assert got["probed"][0][1]["p_yes"] == 1.0
    assert got["pick"] is None


def test_the_window_event_carries_every_probed_position(gate):
    w = window4()[:2]
    fake = FakeVLLM(score=high)
    got = choose(fake, w, VIS4)

    ev = _window_event(got)

    assert ev == {
        "type": "window",
        "probed": [
            {"pos": 2, "tok": " Three", "h": 0.693, "crit": 1.0, "stakes": 1.0,
             "difficulty": 1.0, "commitment": 1.0},
            {"pos": 1, "tok": " Two", "h": 0.5, "crit": 1.0, "stakes": 1.0,
             "difficulty": 1.0, "commitment": 1.0},
        ],
        "opened": True,
    }
    json.dumps(ev, allow_nan=False)  # strict JSON: the browser parses it


def test_known_issue_probe_disabled_picks_by_entropy_with_a_nan_crit(cfg, gate):
    # Documents CURRENT behaviour, listed in the README under "Limitations and
    # known issues"; it is not a specification. With PROBE_ENABLED off the
    # highest-entropy survivor always opens, nothing is asked, and its crit is
    # NaN, which strict JSON (and the browser's JSON.parse) rejects.
    cfg(PROBE_ENABLED=False, PROBE_CANDIDATES=2)
    fake = FakeVLLM()
    w = window4()

    got = choose(fake, w, VIS4)

    assert fake.calls == []
    assert got["probed"] == [(w[1], {"crit": None}), (w[3], {"crit": None})]
    best, s = got["pick"]
    assert best is w[1] and math.isnan(s["crit"])
    assert _window_event(got)["opened"] is True
    with pytest.raises(ValueError):
        json.dumps(s, allow_nan=False)


# ---------------------------------------------------------------------------
# the gate inside respond()
# ---------------------------------------------------------------------------


def test_a_window_below_the_threshold_is_reported_and_the_answer_carries_on(cfg):
    cfg(OPEN_THINK=False, WINDOW_TOKENS=8, WARMUP_TOKENS=2, MIN_GAP_TOKENS=1,
        ENTROPY_FLOOR=0.4, GATE_THRESHOLD=0.55, GATE_LOG_YESNO=False)
    steps = quiet("First", " step", ".") + [hot(" Add")] + quiet(" 3", " and", " 1", ".")
    fake = FakeVLLM(streams=[Stream(steps, "stop")])  # uniform: crit 0.5

    events, conv = run_turn(fake, "What is 3 + 1?")

    assert types(events) == ["start"] + ["token"] * 8 + ["window", "done"]
    (window,) = of_type(events, "window")
    assert window == {
        "type": "window",
        "probed": [{"pos": 3, "tok": " Add", "h": round(entropy2(0.5), 3),
                    "crit": 0.5, "stakes": 0.5, "difficulty": 0.5,
                    "commitment": 0.5}],
        "opened": False,
    }
    assert events[-1]["text"] == "First step. Add 3 and 1."
    assert events[-1]["stats"]["blocks"] == 0
    assert events[-1]["stats"]["rewound_tokens"] == 0
    assert fake.names() == ["stream", "labels", "answer_probs"]
    assert conv.turns[1].notes == []
    fake.assert_consumed()


def test_no_window_event_when_no_position_survives(cfg):
    # " Add" sits inside the warm-up, so nothing is even asked.
    cfg(OPEN_THINK=False, WINDOW_TOKENS=8, WARMUP_TOKENS=4, MIN_GAP_TOKENS=1)
    steps = quiet("First", " step", ".") + [hot(" Add")] + quiet(" 3", ".")
    fake = FakeVLLM(streams=[Stream(steps, "stop")], score=high)

    events, _ = run_turn(fake, "What is 3 + 1?")

    assert "window" not in types(events)
    assert fake.names() == ["stream"]
