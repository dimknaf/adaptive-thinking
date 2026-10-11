"""The pure operations on the thinking tree's frontier.

Beam.rank_score is what candidates are sorted on; Engine._prune cuts them back
to the width without letting one lineage take every slot; Engine._churn
retires a live conversation that is clearly behind and either refills or
closes its slot; _in_contention is the single score test that decides whether
a live conversation may be removed at all.

These are static and pure, so they are driven directly with hand-built beams.
Knobs are set explicitly through the cfg fixture, never left to defaults.
"""

import random

import pytest

from adaptive_thinking.engine import Engine
from adaptive_thinking.events import _beam_json
from adaptive_thinking.scoring import _in_contention
from adaptive_thinking.state import Beam


def beam(name, parent=0, raw=None, done=False, uid=0, score=-1.0, tokens=1):
    """A beam whose text names it, so failures read clearly."""
    return Beam(text=name, score=score, parent=parent, uid=uid, raw=raw,
                done=done, tokens=tokens)


def names(beams):
    return [b.text for b in beams]


# ---------------------------------------------------------------------------
# Beam.rank_score
# ---------------------------------------------------------------------------


def test_rank_score_is_length_normalised_and_penalised():
    assert Beam("", -2.0, 0, tokens=4, penalty=0.35).rank_score == pytest.approx(-0.85)
    assert Beam("", -2.0, 0, tokens=4).rank_score == pytest.approx(-0.5)


def test_rank_score_with_no_tokens_divides_by_one():
    assert Beam("", -2.0, 0, tokens=0, penalty=0.35).rank_score == pytest.approx(-2.35)
    assert Beam("", -2.0, 0, tokens=1).rank_score == pytest.approx(-2.0)


def test_rank_score_prefers_the_longer_beam_at_equal_mean_logprob():
    # Cumulative logprob would rank the short beam first; per token the two
    # are the same, which is what length normalisation is for.
    short = Beam("a", -1.0, 0, tokens=2)
    long = Beam("a b c d", -4.0, 0, tokens=8)
    assert short.score > long.score
    assert short.rank_score == long.rank_score


# ---------------------------------------------------------------------------
# _beam_json
# ---------------------------------------------------------------------------


def test_beam_json_exact_shape():
    b = Beam(text="t", score=-1.234, parent=2, uid=7, from_uid=3, index=1,
             tokens=2, penalty=0.1, last="x", p=0.123456, done=True)
    assert _beam_json(b) == {
        "i": 1, "uid": 7, "from_uid": 3, "parent": 2, "tok": "x", "text": "t",
        "score": -0.717, "total": -1.23, "p": 0.1235, "done": True,
    }


def test_beam_json_without_a_choice_probability():
    got = _beam_json(Beam(text="", score=0.0, parent=-1))
    assert got == {"i": 0, "uid": 0, "from_uid": -1, "parent": -1, "tok": "",
                   "text": "", "score": 0.0, "total": 0.0, "p": None,
                   "done": False}


def test_beam_json_ignores_its_step_argument():
    b = Beam(text="t", score=-1.0, parent=0, tokens=2)
    assert _beam_json(b, step=5) == _beam_json(b)


# ---------------------------------------------------------------------------
# _in_contention: the one test that may remove a live conversation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, leader, expected", [
    # 0.125 and 0.375 are exact in binary, so the boundary is exact too:
    # being exactly CHURN_MARGIN behind is OUT of contention (strict <).
    (0.375, 0.5, False),
    (0.4, 0.5, True),
    (0.2, 0.5, False),
    (0.5, 0.5, True),
    (0.9, 0.5, True),     # ahead of the leader
    (None, 0.5, True),    # not judged yet
    (0.1, None, True),    # nothing judged in the block yet
    (None, None, True),
])
def test_in_contention(cfg, raw, leader, expected):
    cfg(CHURN_MARGIN=0.125)
    assert _in_contention(beam("b", raw=raw), leader) is expected


def test_in_contention_on_a_measured_field(cfg):
    # Four contenders from one recorded checkpoint: a normal field within a
    # few hundredths of each other and one thought 0.18 behind. The margin
    # retires only the clearly bad one, not the runner-up.
    cfg(CHURN_MARGIN=0.12)
    field = [0.735, 0.713, 0.743, 0.563]
    leader = max(field)
    assert [_in_contention(beam("b", raw=r), leader) for r in field] == [
        True, True, True, False]


def test_in_contention_reads_the_margin_at_call_time(cfg):
    b = beam("b", raw=0.4)
    cfg(CHURN_MARGIN=0.2)
    assert _in_contention(b, 0.5) is True
    cfg(CHURN_MARGIN=0.05)
    assert _in_contention(b, 0.5) is False


# ---------------------------------------------------------------------------
# Engine._prune
# ---------------------------------------------------------------------------


def _ranked():
    """[a(p0), b(p0), c(p1), d(p2)], already in rank order."""
    return [beam("a", 0), beam("b", 0), beam("c", 1), beam("d", 2)]


@pytest.mark.parametrize("width, expected", [
    (1, ["a"]),
    (2, ["a", "c"]),
    (3, ["a", "c", "d"]),
    # The cap cannot fill four slots from three parents, so the best
    # remaining candidate is backfilled after the capped pass.
    (4, ["a", "c", "d", "b"]),
    (9, ["a", "c", "d", "b"]),
])
def test_prune_caps_children_per_parent_then_backfills(cfg, width, expected):
    cfg(MAX_CHILDREN_PER_PARENT=1)
    assert names(Engine._prune(_ranked(), width)) == expected


def test_prune_with_a_cap_of_two(cfg):
    cfg(MAX_CHILDREN_PER_PARENT=2)
    assert names(Engine._prune(_ranked(), 3)) == ["a", "b", "c"]


def test_prune_treats_a_cap_below_one_as_one(cfg):
    cfg(MAX_CHILDREN_PER_PARENT=0)
    assert names(Engine._prune(_ranked(), 2)) == ["a", "c"]


def test_prune_defaults_to_the_beam_width(cfg):
    cfg(MAX_CHILDREN_PER_PARENT=1, BEAM_WIDTH=2)
    assert names(Engine._prune(_ranked())) == ["a", "c"]


def test_prune_of_nothing_is_nothing(cfg):
    cfg(MAX_CHILDREN_PER_PARENT=1)
    assert Engine._prune([], 4) == []


def test_prune_returns_the_candidates_themselves(cfg):
    cfg(MAX_CHILDREN_PER_PARENT=1)
    ranked = _ranked()
    out = Engine._prune(ranked, 4)
    assert [id(b) for b in out] == [id(ranked[i]) for i in (0, 2, 3, 1)]


def test_prune_keeps_order_given_not_rank_score(cfg):
    # _prune trusts the order it is handed; sorting is the caller's job.
    cfg(MAX_CHILDREN_PER_PARENT=1)
    worse = beam("worse", 0, score=-9.0)
    better = beam("better", 1, score=-0.1)
    assert names(Engine._prune([worse, better], 1)) == ["worse"]


def test_prune_dedupes_by_identity_not_by_value(cfg):
    # Beam is a dataclass, so two candidates that differ in nothing compare
    # equal. Both are separate conversations and both must be kept.
    cfg(MAX_CHILDREN_PER_PARENT=1)
    x, y = beam("same", 0), beam("same", 0)
    assert x == y and x is not y
    out = Engine._prune([x, y], 2)
    assert len(out) == 2
    assert out[0] is x and out[1] is y


def test_prune_property(cfg):
    rng = random.Random(31)
    for _ in range(400):
        cap = rng.randint(1, 3)
        cfg(MAX_CHILDREN_PER_PARENT=cap)
        ranked = [beam(f"c{i}", rng.randint(0, 3)) for i in range(rng.randint(0, 12))]
        width = rng.randint(1, 8)
        out = Engine._prune(ranked, width)

        # Exactly as many as fit, each a distinct candidate from `ranked`.
        assert len(out) == min(width, len(ranked))
        assert len({id(b) for b in out}) == len(out)
        assert {id(b) for b in out} <= {id(b) for b in ranked}

        per_parent = {}
        for b in ranked:
            per_parent.setdefault(b.parent, []).append(b)
        # Within a lineage the better candidates are never skipped for worse.
        for p, kids in per_parent.items():
            kept = [b for b in kids if any(b is o for o in out)]
            assert all(a is b for a, b in zip(kept, kids))
        # When the cap alone can fill the frontier, the cap holds.
        if sum(min(len(k), cap) for k in per_parent.values()) >= width:
            for p in per_parent:
                assert sum(1 for b in out if b.parent == p) <= cap


# ---------------------------------------------------------------------------
# Engine._churn
# ---------------------------------------------------------------------------


@pytest.fixture
def churn_knobs(cfg):
    return cfg(CHURN_MARGIN=0.125, CHURN_PATIENCE=2)


def _field(b_raw=0.5):
    """Live beams A (good, lineage 0) and B (lineage 1), with a discarded
    sibling of A still in the ranked list."""
    a = beam("A", parent=0, raw=0.9, uid=0)
    b = beam("B", parent=1, raw=b_raw, uid=1)
    a2 = beam("A2", parent=0, raw=0.9, uid=0)
    return a, b, a2, [a, a2, b]


def test_churn_replaces_a_laggard_while_the_block_is_improving(churn_knobs):
    a, b, a2, ranked = _field()
    retired, entered = Engine._churn([a, b], ranked, next_uid=10, leader=0.9, dry=0)
    assert retired == [b] and retired[0] is b
    assert entered == [a2] and entered[0] is a2
    # The replacement is a new conversation, split off the best lineage.
    assert (a2.uid, a2.from_uid) == (10, a.uid)
    # The retired beam keeps its identity, so it stays on screen as itself.
    assert b.uid == 1


def test_churn_still_replaces_one_checkpoint_short_of_patience(churn_knobs):
    a, b, a2, ranked = _field()
    retired, entered = Engine._churn([a, b], ranked, next_uid=10, leader=0.9, dry=1)
    assert retired == [b] and entered == [a2]


def test_churn_closes_the_slot_once_the_block_has_gone_dry(churn_knobs):
    a, b, a2, ranked = _field()
    retired, entered = Engine._churn([a, b], ranked, next_uid=10, leader=0.9, dry=2)
    assert retired == [b] and entered == []
    assert a2.uid == 0  # nothing was taken from the ranked list


def test_churn_closes_the_slot_when_there_is_no_spare(churn_knobs):
    a, b, _, _ = _field()
    assert Engine._churn([a, b], [a, b], next_uid=10, leader=0.9, dry=0) == ([b], [])


def test_churn_never_refills_with_a_finished_thought(churn_knobs):
    a, b, _, _ = _field()
    done = beam("A-done", parent=0, raw=0.9, done=True)
    assert Engine._churn([a, b], [a, done, b], 10, 0.9, 0) == ([b], [])


def test_churn_refills_from_the_best_lineage_not_the_retired_one(churn_knobs):
    a, b, a2, _ = _field()
    b2 = beam("B2", parent=1, raw=0.5)
    # B's sibling ranks first, but it is a variation of the branch just
    # judged not worth pursuing.
    retired, entered = Engine._churn([a, b], [b2, a, b, a2], 10, 0.9, 0)
    assert retired == [b] and entered == [a2]


def test_churn_takes_the_first_spare_in_rank_order(churn_knobs):
    a, b, a2, _ = _field()
    a3 = beam("A3", parent=0, raw=0.9)
    _, entered = Engine._churn([a, b], [a, a3, a2, b], 10, 0.9, 0)
    assert entered == [a3]


def test_churn_leaves_a_contender_alone(churn_knobs):
    a, b, _, ranked = _field(b_raw=0.85)
    assert Engine._churn([a, b], ranked, 10, 0.9, 0) == ([], [])


def test_churn_margin_boundary_is_strict(churn_knobs):
    a = beam("A", parent=0, raw=0.5)
    b = beam("B", parent=1, raw=0.375)   # exactly CHURN_MARGIN behind
    retired, _ = Engine._churn([a, b], [a, b], 10, 0.5, 0)
    assert retired == [b]
    b.raw = 0.4
    assert Engine._churn([a, b], [a, b], 10, 0.5, 0) == ([], [])


def test_churn_measures_against_the_block_leader_not_the_live_field(churn_knobs):
    # Both live beams are close to each other, but a finished thought set the
    # leader well above them: the worse live one is not a challenger.
    a = beam("A", parent=0, raw=0.82)
    b = beam("B", parent=1, raw=0.80)
    retired, _ = Engine._churn([a, b], [a, b], 10, 0.95, 2)
    assert retired == [b]


def test_churn_retires_at_most_one_per_checkpoint(churn_knobs):
    a = beam("A", parent=0, raw=0.9)
    b = beam("B", parent=1, raw=0.5)
    c = beam("C", parent=2, raw=0.4)
    retired, _ = Engine._churn([a, b, c], [a, b, c], 10, 0.9, 2)
    assert retired == [c]


def test_churn_never_retires_an_unjudged_beam(churn_knobs):
    a = beam("A", parent=0, raw=0.9)
    u = beam("U", parent=1, raw=None)
    b = beam("B", parent=2, raw=0.5)
    retired, _ = Engine._churn([a, u, b], [a, u, b], 10, 0.9, 2)
    assert retired == [b]
    # With the only judged beam leading, nothing leaves.
    assert Engine._churn([a, u], [a, u], 10, 0.9, 2) == ([], [])


@pytest.mark.parametrize("beams, leader", [
    ([beam("only", raw=0.1)], 0.9),                        # the last one standing
    ([], 0.9),
    ([beam("x", raw=None), beam("y", 1, raw=None)], 0.9),  # nothing judged
    ([beam("x", raw=0.1), beam("y", 1, raw=0.9)], None),   # no leader yet
])
def test_churn_does_nothing_without_a_field_to_judge(churn_knobs, beams, leader):
    assert Engine._churn(beams, list(beams), 10, leader, 5) == ([], [])


def test_churn_patience_is_read_from_config(cfg):
    cfg(CHURN_MARGIN=0.125, CHURN_PATIENCE=1)
    a, b, _, ranked = _field()
    assert Engine._churn([a, b], ranked, 10, 0.9, 1) == ([b], [])


def test_prune_then_churn_refills_with_the_pruned_sibling(cfg):
    # The order _run_block uses them in: prune the ranked candidates to the
    # width, then let the churn look for a spare among what was cut.
    cfg(MAX_CHILDREN_PER_PARENT=1, CHURN_MARGIN=0.125, CHURN_PATIENCE=2)
    a0 = beam("a0", parent=0, raw=0.9, uid=0)
    b0 = beam("b0", parent=1, raw=0.5, uid=1)
    a1 = beam("a1", parent=0, raw=0.9, uid=0)
    b1 = beam("b1", parent=1, raw=0.5, uid=1)
    ranked = [a0, b0, a1, b1]

    beams = Engine._prune(ranked, 2)
    assert beams == [a0, b0]
    retired, entered = Engine._churn(beams, ranked, next_uid=4, leader=0.9, dry=0)
    assert retired == [b0] and entered == [a1]
    assert (a1.uid, a1.from_uid) == (4, 0)
