"""The levelled reads: how the gate scores a position and how a thought is judged.

Both ask closed questions with lettered levels and read the answer at a single
position. _expected turns one read into a number (worst level 0, best 1).
_gate_scores combines the GATE_AXES into `crit`, which is ABSOLUTE and never
standardised across positions. _score_axes combines the AXES into two numbers
per thought: `raw`, the plain weighted mean (0..1, comparable across any two
thoughts), and `overall`, standardised within a group (live beams against live
beams, finished thoughts against finished thoughts).

The scripted endpoint answers each prompt with a one-hot level chosen by which
axis it asks about and, for the judge, which thought it quotes.
"""

import asyncio
import random

import pytest

from adaptive_thinking import prompts
from adaptive_thinking.engine import Engine
from adaptive_thinking.events import _mean
from adaptive_thinking.scoring import _expected
from adaptive_thinking.state import Beam
from fake_endpoint import LABEL_ID, asked_axis, context_of, lvl, thought_of

CTX = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nWhen does it arrive?"
       "<|im_end|>\n<|im_start|>assistant\n")
A, B, C, D, E = range(5)  # level indices, worst to best


def _weights(axes):
    return {name: a["weight"] for name, a in axes.items()}


# ---------------------------------------------------------------------------
# _mean and _expected
# ---------------------------------------------------------------------------


def test_mean():
    assert _mean([]) is None
    assert _mean([1, 2, 2]) == 1.6667
    assert _mean([-0.1, -0.2]) == pytest.approx(-0.15)
    assert _mean([3]) == 3


@pytest.mark.parametrize("level", range(5))
def test_expected_of_a_certain_level(level):
    assert _expected(lvl(level), 5) == level / 4


@pytest.mark.parametrize("probs, n, expected", [
    ([0.2] * 5, 5, 0.5),
    ([0.5, 0, 0, 0, 0.5], 5, 0.5),
    ([0, 0.5, 0.5, 0, 0], 5, 0.375),
    # Renormalised over the levels, so unnormalised input works too.
    ([0, 2, 0, 0, 0], 5, 0.25),
    ([0, 0, 0, 0, 0], 5, 0.0),
    # Only the first n entries are levels.
    ([0, 0, 1, 5, 5], 3, 1.0),
    ([1, 1, 0], 2, 0.5),
])
def test_expected(probs, n, expected):
    assert _expected(probs, n) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _gate_scores
# ---------------------------------------------------------------------------


def gate_levels(per_context):
    """score(): the gate level for each (context, axis). `per_context` maps
    a context to {axis: level}; a single dict applies to every context."""

    def score(prompt):
        family, axis = asked_axis(prompt)
        assert family == "gate", f"unexpected question: {prompt[-120:]!r}"
        levels = per_context.get(context_of(prompt), per_context)
        return lvl(levels[axis])

    return score


def gate(fake, ctxs):
    return asyncio.run(Engine(fake)._gate_scores(ctxs))


def test_gate_crit_is_the_weighted_mean_of_the_axes(fake_llm):
    fake = fake_llm(score=gate_levels({"stakes": E, "difficulty": C, "commitment": A}))
    (got,) = gate(fake, [CTX])
    w = _weights(prompts.GATE_AXES)
    assert w == {"stakes": 1.4, "difficulty": 1.2, "commitment": 1.0}
    # (1.4 * 1.0 + 1.2 * 0.5 + 1.0 * 0.0) / 3.6
    assert got == {"crit": 0.5556, "stakes": 1.0, "difficulty": 0.5, "commitment": 0.0}


def test_gate_crit_from_spread_answers(fake_llm):
    def score(prompt):
        _, axis = asked_axis(prompt)
        return [0, 0.5, 0.5, 0, 0] if axis == "difficulty" else lvl(E)

    (got,) = gate(fake_llm(score=score), [CTX])
    # (1.4 + 1.2 * 0.375 + 1.0) / 3.6 = 0.79166...
    assert got == {"crit": 0.7917, "stakes": 1.0, "difficulty": 0.375,
                   "commitment": 1.0}


def test_gate_scores_every_position_on_every_axis_in_one_request(fake_llm):
    ctxs = [CTX + "It leaves at 09:35.", CTX + "It leaves at 09:35. The trip"]
    fake = fake_llm(score=gate_levels({"stakes": D, "difficulty": D, "commitment": D}))
    gate(fake, ctxs)

    assert fake.names() == ["labels", "answer_probs"]
    assert fake.of("labels") == [{"n": 5}]
    (call,) = fake.of("answer_probs")
    assert call["ids"] == [LABEL_ID + i for i in range(5)]
    # Position-major, axis-minor.
    assert call["prompts"] == [prompts.gate_prompt(x, a)
                               for x in ctxs for a in prompts.GATE_AXES]
    assert len(call["prompts"]) == 6


def test_gate_crit_is_absolute_not_standardised_across_positions(fake_llm):
    low, mid, high, same = (CTX + s for s in ("a", "b", "c", "d"))
    fake = fake_llm(score=gate_levels({
        low: {"stakes": A, "difficulty": A, "commitment": A},
        mid: {"stakes": D, "difficulty": D, "commitment": D},
        high: {"stakes": E, "difficulty": E, "commitment": E},
        same: {"stakes": D, "difficulty": D, "commitment": D},
    }))
    got = gate(fake, [low, mid, high, same])
    # Results come back in position order, each on the absolute 0..1 scale:
    # identical answers give identical scores wherever they sit in the window.
    assert [g["crit"] for g in got] == [0.0, 0.75, 1.0, 0.75]


def test_gate_weights_and_axes_come_from_config(cfg, fake_llm):
    levels = ["no", "little", "some", "much", "all"]
    cfg(GATE_AXES={
        "x": dict(weight=3.0, question="Question X?", levels=levels),
        "y": dict(weight=1.0, question="Question Y?", levels=levels),
    })
    fake = fake_llm(score=gate_levels({"x": E, "y": A}))
    (got,) = gate(fake, [CTX])
    assert got == {"crit": 0.75, "x": 1.0, "y": 0.0}
    assert len(fake.of("answer_probs")[0]["prompts"]) == 2


# ---------------------------------------------------------------------------
# _score_axes
# ---------------------------------------------------------------------------

AXIS_NAMES = list(prompts.AXES)
WSUM = sum(a["weight"] for a in prompts.AXES.values())


def judge(table):
    """score(): the level for each (thought, axis). A table value is one
    level for every axis, or {axis: level}."""

    def score(prompt):
        family, axis = asked_axis(prompt)
        assert family == "judge", f"unexpected question: {prompt[-120:]!r}"
        v = table[thought_of(prompt)]
        return lvl(v[axis] if isinstance(v, dict) else v)

    return score


def thought(text, done=False):
    return Beam(text=text, score=-1.0, parent=0, done=done)


def score_axes(fake, *groups):
    asyncio.run(Engine(fake)._score_axes(CTX, *groups))


def test_axis_weights_sum_to_4_6():
    assert _weights(prompts.AXES) == {"looping": 1.0, "progress": 1.3,
                                     "confusion": 1.6, "concreteness": 0.7}
    assert WSUM == pytest.approx(4.6)


def test_raw_and_overall_for_a_finished_group(fake_llm):
    f1, f2 = thought("Carry the hour. It is 16:43.", True), thought("Add them.", True)
    score_axes(fake_llm(score=judge({f1.text: E, f2.text: C})), [], [f1, f2])

    assert f1.raw == pytest.approx(1.0) and f2.raw == pytest.approx(0.5)
    for a in AXIS_NAMES:
        assert (f1.axes[a], f2.axes[a]) == (1.0, 0.5)
        # Centred on the group mean (0.75), divided by the range (0.5).
        assert f1.axes[a + "_z"] == pytest.approx(0.5)
        assert f2.axes[a + "_z"] == pytest.approx(-0.5)
    assert f1.overall == pytest.approx(0.5 * WSUM) == pytest.approx(2.3)
    assert f2.overall == pytest.approx(-2.3)


def test_raw_is_the_weighted_mean_of_the_raw_axes(fake_llm):
    t = thought("Sound and not repetitive, but gets nowhere.")
    score_axes(fake_llm(score=judge({t.text: {
        "looping": E, "progress": A, "confusion": E, "concreteness": A}})), [t])
    # (1.0 * 1 + 1.3 * 0 + 1.6 * 1 + 0.7 * 0) / 4.6
    assert t.raw == pytest.approx(2.6 / 4.6)
    assert t.axes == {"looping": 1.0, "progress": 0.0, "confusion": 1.0,
                      "concreteness": 0.0, "looping_z": 0.0, "progress_z": 0.0,
                      "confusion_z": 0.0, "concreteness_z": 0.0}


def test_a_single_candidate_group_has_overall_zero(fake_llm):
    t = thought("The only finished thought.", True)
    score_axes(fake_llm(score=judge({t.text: D})), [], [t])
    assert t.raw == pytest.approx(0.75)
    assert t.overall == 0.0
    assert all(t.axes[a + "_z"] == 0.0 for a in AXIS_NAMES)


def test_live_and_finished_groups_are_standardised_separately(fake_llm):
    l1, l2 = thought("Half way: 13:12 minus"), thought("The question asks")
    f1, f2 = thought("It arrives at 16:43.", True), thought("About 16:40.", True)
    score_axes(fake_llm(score=judge({l1.text: B, l2.text: A, f1.text: E, f2.text: D})),
               [l1, l2], [f1, f2])

    # raw is absolute and comparable across the groups...
    assert [t.raw for t in (l1, l2, f1, f2)] == pytest.approx([0.25, 0.0, 1.0, 0.75])
    # ...overall is relative to the group. The better live beam ranks as high
    # among the live beams as the best finished thought does among those.
    assert [t.overall for t in (l1, l2, f1, f2)] == pytest.approx([2.3, -2.3, 2.3, -2.3])


def test_one_group_instead_of_two_would_rank_the_live_beam_down(fake_llm):
    # The contrast to the test above: standardised together, the same live
    # beam sits below the mean of the field.
    l1, l2 = thought("Half way: 13:12 minus"), thought("The question asks")
    f1, f2 = thought("It arrives at 16:43.", True), thought("About 16:40.", True)
    score_axes(fake_llm(score=judge({l1.text: B, l2.text: A, f1.text: E, f2.text: D})),
               [l1, l2, f1, f2])
    assert l1.overall < 0 < f1.overall


def test_all_candidates_ride_in_one_request(fake_llm):
    live = [thought("First live."), thought("Second live.")]
    done = [thought("Finished.", True)]
    fake = fake_llm(score=judge({"First live.": C, "Second live.": B, "Finished.": E}))
    score_axes(fake, live, done)

    assert fake.names() == ["labels", "answer_probs"]
    assert fake.of("labels") == [{"n": 5}]
    (call,) = fake.of("answer_probs")
    assert call["ids"] == [LABEL_ID + i for i in range(5)]
    # Candidate-major, axis-minor, groups in the order passed.
    assert call["prompts"] == [prompts.axis_prompt(CTX, t.text, a)
                               for t in live + done for a in AXIS_NAMES]


def test_nothing_to_score_makes_no_request(fake_llm):
    fake = fake_llm(score=judge({}))
    score_axes(fake)
    score_axes(fake, [], [])
    assert fake.calls == []


def test_an_empty_group_is_skipped(fake_llm):
    t = thought("Alone.", True)
    fake = fake_llm(score=judge({"Alone.": E}))
    score_axes(fake, [], [t])
    assert t.overall == 0.0 and t.raw == pytest.approx(1.0)
    assert len(fake.of("answer_probs")[0]["prompts"]) == len(AXIS_NAMES)


def test_standardisation_uses_the_group_mean_and_range(fake_llm):
    ts = [thought("p0"), thought("p1"), thought("p4")]
    table = {t.text: {"looping": C, "confusion": C, "concreteness": C,
                      "progress": p} for t, p in zip(ts, (A, B, E))}
    score_axes(fake_llm(score=judge(table)), ts)

    # progress raw 0, 0.25, 1: mean 5/12, range 1.
    zs = [t.axes["progress_z"] for t in ts]
    assert zs == pytest.approx([-5 / 12, -2 / 12, 7 / 12])
    # Axes with no spread contribute nothing.
    assert all(t.axes["looping_z"] == 0.0 for t in ts)
    assert [t.overall for t in ts] == pytest.approx([1.3 * z for z in zs])


def test_the_axis_clip_is_applied(cfg, fake_llm):
    cfg(AXIS_CLIP=0.25)
    f1, f2 = thought("Good.", True), thought("Fair.", True)
    score_axes(fake_llm(score=judge({"Good.": E, "Fair.": C})), [f1, f2])
    assert all(f1.axes[a + "_z"] == 0.25 for a in AXIS_NAMES)
    assert f1.overall == pytest.approx(0.25 * WSUM)
    assert f2.overall == pytest.approx(-0.25 * WSUM)


def test_range_standardisation_keeps_every_term_within_one(fake_llm):
    # |x - mean| never exceeds the range, so each standardised term lies in
    # [-1, 1] and a clip wider than 1 (the default is 2.0) does not change it.
    rng = random.Random(5)
    for _ in range(40):
        ts = [thought(f"t{i}") for i in range(rng.randint(1, 6))]
        table = {t.text: {a: rng.randrange(5) for a in AXIS_NAMES} for t in ts}
        score_axes(fake_llm(score=judge(table)), ts)
        for t in ts:
            assert all(-1.0 <= t.axes[a + "_z"] <= 1.0 for a in AXIS_NAMES)
            assert 0.0 <= t.raw <= 1.0


def test_scores_replace_whatever_the_beam_carried(fake_llm):
    # A candidate inherits its parent's judgement until it is judged itself.
    t = thought("Re-judged.")
    t.axes, t.raw, t.overall = {"stale": 1.0}, 0.1, 9.0
    score_axes(fake_llm(score=judge({"Re-judged.": E})), [t])
    assert "stale" not in t.axes
    assert t.raw == pytest.approx(1.0) and t.overall == 0.0
