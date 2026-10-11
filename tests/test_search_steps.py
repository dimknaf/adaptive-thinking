"""The steps a thinking block is built from, each on its own.

Engine._run_block runs every checkpoint as the same sequence of steps, and
most of them are pure functions of the frontier: _expand turns the chunks
into candidates, _apply_churn carries out the churn's decision,
_assign_split_uids numbers the slots, _update_leader and _award_medals read
the judge's scores, and _should_stop and _drop_ramblers decide whether the
block goes on. _pick_winner chooses the monologue and _finalise_block records
it. _standardise_group is the second half of Engine._score_axes, and _eligible
is the filter at the top of Engine._choose. Three event builders fix the key
order of block_open, think_step and standings.

test_block.py and test_gate.py cover the same code end to end. Here each step
is driven directly with hand-built beams, so a failure names the step that
changed. Every knob a step reads is set explicitly through the cfg fixture.
"""

import copy

import pytest

from adaptive_thinking import gate
from adaptive_thinking.engine import Engine
from adaptive_thinking.events import (
    _beam_json,
    _block_open_event,
    _standings_event,
    _think_step_event,
)
from adaptive_thinking.gate import _Cand, _eligible
from adaptive_thinking.scoring import _standardise_group
from adaptive_thinking.state import Beam, Block
from adaptive_thinking.text import _already_written, _at_boundary
from adaptive_thinking.tree import (
    _apply_churn,
    _assign_split_uids,
    _award_medals,
    _drop_ramblers,
    _expand,
    _pick_winner,
    _should_stop,
    _update_leader,
)
from fake_endpoint import ln, step


def beam(name, score=-1.0, parent=0, **fields):
    """A beam whose text names it, so failures read clearly."""
    return Beam(text=name, score=score, parent=parent, **fields)


def names(beams):
    return [b.text for b in beams]


# ---------------------------------------------------------------------------
# _eligible: the gate's six rules
# ---------------------------------------------------------------------------

# A position that passes every rule: " They" starts a new sentence.
VIS = "Trains leave at 09:35. They arrive at 13:12."
OFF = VIS.index(" They")
IDX = 5


def hot(text, p=0.5):
    """A token with two distinct alternatives; entropy ln 2 = 0.693 at p=0.5."""
    return step(text, top=[(text, ln(p)), (text + "~", ln(1 - p))])


@pytest.fixture
def rules(cfg):
    """Every rule's knob at a value the reference position passes."""
    return cfg(ENTROPY_FLOOR=0.4, MIN_GAP_TOKENS=0, WARMUP_TOKENS=0)


def test_eligible_passes_a_position_that_meets_every_rule(rules):
    assert _eligible(_Cand(IDX, OFF, hot(" They")), 0, VIS) is True


def test_eligible_needs_entropy_at_or_above_the_floor(rules, cfg):
    c = _Cand(IDX, OFF, hot(" They"))  # ln 2 = 0.6931
    cfg(ENTROPY_FLOOR=0.7)
    assert _eligible(c, 0, VIS) is False
    cfg(ENTROPY_FLOOR=0.69)
    assert _eligible(c, 0, VIS) is True


def test_eligible_needs_two_distinct_alternatives(rules):
    # The same token twice: plenty of entropy, one alternative.
    twice = step(" They", top=[(" They", ln(0.5)), (" They", ln(0.5))])
    assert twice.entropy > 0.4 and twice.alternatives == [" They"]
    assert _eligible(_Cand(IDX, OFF, twice), 0, VIS) is False


def test_eligible_needs_min_gap_tokens_since_the_last_block(rules, cfg):
    cfg(MIN_GAP_TOKENS=3)
    c = _Cand(IDX, OFF, hot(" They"))
    assert _eligible(c, IDX - 2, VIS) is False
    assert _eligible(c, IDX - 3, VIS) is True  # exactly the gap is enough


def test_eligible_needs_the_warmup_to_have_passed(rules, cfg):
    c = _Cand(IDX, OFF, hot(" They"))
    cfg(WARMUP_TOKENS=IDX + 1)
    assert _eligible(c, 0, VIS) is False
    cfg(WARMUP_TOKENS=IDX)
    assert _eligible(c, 0, VIS) is True


def test_eligible_refuses_a_position_inside_text_already_written(rules):
    sentence = "Trains leave at 09:35 and arrive at 13:12 on time. "  # 51 chars
    repeated = sentence + sentence + "Then they wait."
    off = 2 * len(sentence)
    assert _at_boundary(repeated, off)
    assert _already_written(repeated, off)
    assert _eligible(_Cand(IDX, off, hot(" Then")), 0, repeated) is False
    # The same sentence once, after a different one: nothing repeats.
    first = "Buses leave at 10:05 and arrive at 14:40 on time. "
    fresh = first + sentence + "Then."
    assert _eligible(_Cand(IDX, len(first + sentence), hot(" Then")), 0, fresh) is True


def test_eligible_refuses_a_position_mid_sentence(rules):
    off = VIS.index(" arrive")
    assert not _already_written(VIS, off)
    assert _eligible(_Cand(IDX, off, hot(" arrive")), 0, VIS) is False


class _EntropyOnly:
    """A step that is below any floor, and whose alternatives must not be read."""

    entropy = 0.0

    @property
    def alternatives(self):
        raise AssertionError("alternatives read after the entropy floor failed")


def test_eligible_stops_at_the_first_rule_that_fails(rules, cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(gate, "_already_written",
                        lambda vis, off: calls.append("written") or False)
    monkeypatch.setattr(gate, "_at_boundary",
                        lambda vis, off: calls.append("boundary") or True)
    c = _Cand(IDX, OFF, hot(" They"))

    assert _eligible(c, 0, VIS) is True
    assert calls == ["written", "boundary"]

    # The entropy floor is the first rule: nothing after it is looked at.
    calls.clear()
    assert _eligible(_Cand(IDX, OFF, _EntropyOnly()), 0, VIS) is False
    assert calls == []

    # The position rules come before either text rule.
    cfg(WARMUP_TOKENS=IDX + 1)
    assert _eligible(c, 0, VIS) is False
    assert calls == []

    # Already written: the boundary is never asked.
    cfg(WARMUP_TOKENS=0)
    monkeypatch.setattr(gate, "_already_written",
                        lambda vis, off: calls.append("written") or True)
    assert _eligible(c, 0, VIS) is False
    assert calls == ["written"]


# ---------------------------------------------------------------------------
# _expand: chunks -> candidates
# ---------------------------------------------------------------------------

CTX = "Q: two plus two?"


@pytest.fixture
def knobs(cfg):
    return cfg(SIBLING_PENALTY=0.25, COPY_PENALTY=0.5, COPY_NGRAM=8)


def test_expand_a_chunk_that_closed_its_thought_is_one_finished_candidate(knobs):
    b = beam("So", score=-0.5, parent=3, uid=2, from_uid=1, index=1, tokens=4,
             penalty=0.125, axes={"progress": 0.5}, overall=0.25, raw=0.75)
    chunk = " it is four.</think>tail"  # 24 characters
    [c] = _expand([b], [(chunk, -2.0, [(" x", -0.1)], False)], CTX)
    assert c.text == "So it is four."  # cut at the marker
    assert c.score == -0.5 + -2.0
    assert c.tokens == 4 + 24 // 4
    assert c.penalty == 0.125  # carried, no new penalty
    assert c.done is True
    assert c.last == "</think>tail"  # the chunk's last 12 characters
    # It stays in b's lineage and carries b's judgement down.
    assert (c.parent, c.uid, c.from_uid) == (1, 2, 2)
    assert (c.overall, c.raw) == (0.25, 0.75)
    assert c.axes == {"progress": 0.5} and c.axes is not b.axes


@pytest.mark.parametrize("top, stopped", [
    ([(" four", -0.1)], True),  # the server said the stream stopped
    ([], False),               # no alternatives to branch on
])
def test_expand_a_stopped_or_unbranchable_chunk_is_finished(knobs, top, stopped):
    [c] = _expand([beam("So")], [(" it is four", -1.0, top, stopped)], CTX)
    assert c.done is True and c.text == "So it is four"


def test_expand_branches_at_the_last_token_of_the_chunk(knobs):
    b = beam("So", score=-0.1, tokens=4)
    chunk = " it is four"  # 11 characters
    cands = _expand([b], [(chunk, -0.2, [(" four", -0.05), (" 4", -0.6)], False)], CTX)
    # The chunk's last token is REPLACED by each alternative, not repeated.
    assert names(cands) == ["So it is four", "So it is 4"]
    assert [c.last for c in cands] == [" four", " 4"]
    assert [c.tokens for c in cands] == [4 + 11 // 4 + 1] * 2
    assert [c.done for c in cands] == [False, False]
    assert [c.penalty for c in cands] == [0.0, 0.0]
    assert cands[0].score == -0.1 + -0.2 - -0.05 + -0.05
    # Evaluated left to right, exactly as written. Regrouped as
    # b.score + (chunk_lp - best_lp + lp) it would be -0.85, a different
    # float, and a different float can flip a tie.
    assert cands[1].score == -0.8500000000000001
    assert -0.1 + (-0.2 - -0.05 + -0.6) == -0.85


def test_expand_keeps_the_whole_chunk_when_the_top_token_is_not_its_end(knobs):
    cands = _expand([beam("So")], [(" it is four", -1.0, [(" five", -0.1)], False)], CTX)
    assert names(cands) == ["So it is four five"]


def test_expand_retires_a_candidate_that_starts_copying_the_nudge(knobs, cfg):
    cfg(NUDGE_PREFIX="Right - back to the answer.")
    top = [(" to", -0.1), (" at", -0.5)]
    cands = _expand([beam("So")], [(" fine. Right - back to", -1.0, top, False)], CTX)
    # Cut back to before the nudge's first 12 characters, and finished.
    assert names(cands) == ["So fine. ", "So fine. "]
    assert [c.done for c in cands] == [True, True]


def test_expand_sibling_penalty_applies_only_to_a_repeated_token(knobs):
    b = beam("So", penalty=0.125)
    # The client reads the top-k from a dict, so its tokens are distinct and
    # this penalty never applies in a live run. It applies here only because
    # the list is hand-built with a repeat.
    distinct = _expand([b], [(" x", -1.0, [(" a", -0.1), (" b", -0.2)], False)], CTX)
    assert [c.penalty for c in distinct] == [0.125, 0.125]
    repeat = _expand([b], [(" x", -1.0, [(" a", -0.1), (" a", -0.2), (" b", -0.3)], False)], CTX)
    assert [c.penalty for c in repeat] == [0.125, 0.125 + 0.25, 0.125]


def test_expand_charges_for_copying_the_context(knobs, cfg):
    ctx = "The answer: So it is four."
    top = [(" four", -0.05), (" 4", -0.6)]
    cands = _expand([beam("So", penalty=0.125)], [(" it is four", -1.0, top, False)], ctx)
    # "So it is four"[-8:] = " is four" is in the context; " it is 4" is not.
    assert [c.penalty for c in cands] == [0.125 + 0.5, 0.125]

    cfg(COPY_PENALTY=0.0)
    cands = _expand([beam("So")], [(" it is four", -1.0, top, False)], ctx)
    assert [c.penalty for c in cands] == [0.0, 0.0]

    cfg(COPY_PENALTY=0.5, COPY_NGRAM=50)  # longer than the candidate
    cands = _expand([beam("So")], [(" it is four", -1.0, top, False)], ctx)
    assert [c.penalty for c in cands] == [0.0, 0.0]


def test_expand_lists_candidates_beam_by_beam_in_top_order(knobs):
    live = [beam("A", uid=0, index=0), beam("B", uid=1, index=1)]
    chunks = [(" one", -1.0, [(" one", -0.1), (" won", -0.9)], False),
              (" two", -1.0, [(" two", -0.1), (" too", -0.9)], False)]
    cands = _expand(live, chunks, CTX)
    assert names(cands) == ["A one", "A won", "B two", "B too"]
    assert [c.parent for c in cands] == [0, 0, 1, 1]


def test_expand_leaves_its_inputs_untouched(knobs):
    live = [beam("So", uid=3, axes={"progress": 0.5}, raw=0.5, overall=0.1)]
    chunks = [(" it is four", -1.0, [(" four", -0.05), (" 4", -0.6)], False)]
    before = (copy.deepcopy(live), copy.deepcopy(chunks))
    cands = _expand(live, chunks, CTX)
    assert (live, chunks) == before
    assert all(c is not live[0] and c.axes is not live[0].axes for c in cands)


def test_expand_without_chunks_raises_as_the_loop_did():
    # A stream that never delivered its chunks leaves None behind; zipping it
    # fails loudly rather than ending the block quietly.
    with pytest.raises(TypeError):
        _expand([beam("So")], None, CTX)


# ---------------------------------------------------------------------------
# _apply_churn and _assign_split_uids
# ---------------------------------------------------------------------------


def test_apply_churn_with_nothing_retired_changes_nothing():
    beams = [beam("a"), beam("b")]
    got, next_uid, closed = _apply_churn(beams, [], [], 7, 1)
    assert got is beams and (next_uid, closed) == (7, 1)


def test_apply_churn_replaces_the_retired_conversation_at_the_end():
    a, b, c, new = beam("a"), beam("b"), beam("c"), beam("new")
    got, next_uid, closed = _apply_churn([a, b, c], [b], [new], 7, 1)
    assert names(got) == ["a", "c", "new"]
    assert (next_uid, closed) == (8, 1)


def test_apply_churn_without_a_replacement_closes_the_slot():
    a, b, c = beam("a"), beam("b"), beam("c")
    got, next_uid, closed = _apply_churn([a, b, c], [b], [], 7, 1)
    assert names(got) == ["a", "c"]
    assert (next_uid, closed) == (7, 2)


def test_apply_churn_retires_by_identity_not_by_value():
    b = beam("same", raw=0.5)
    twin = beam("same", raw=0.5)
    assert b == twin
    got, _, _ = _apply_churn([b, twin], [b], [], 0, 0)
    assert len(got) == 1 and got[0] is twin


def test_assign_split_uids_renumbers_slots_and_gives_each_split_a_new_uid():
    beams = [beam(n, uid=u, index=9) for n, u in zip("abcd", [0, 0, 1, 0])]
    same = list(beams)
    assert _assign_split_uids(beams, 5) == 7
    assert [b.uid for b in beams] == [0, 5, 1, 6]
    assert [b.index for b in beams] == [0, 1, 2, 3]
    assert all(x is y for x, y in zip(beams, same))  # edited in place


def test_assign_split_uids_without_a_split_keeps_every_uid():
    beams = [beam(n, uid=u, index=5) for n, u in zip("abc", [4, 2, 7])]
    assert _assign_split_uids(beams, 10) == 10
    assert [b.uid for b in beams] == [4, 2, 7]
    assert [b.index for b in beams] == [0, 1, 2]


# ---------------------------------------------------------------------------
# _update_leader and _award_medals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raws, leader, dry, expected", [
    ([0.6], None, 3, (0.6, 0)),           # the first score is a new best
    ([0.5, 0.75], 0.5, 2, (0.75, 0)),     # clear of the margin
    ([0.25], 0.5, 2, (0.5, 3)),           # worse: the streak grows
    ([None, 0.7, None], 0.5, 0, (0.7, 0)),  # unjudged thoughts are skipped
    ([None], 0.5, 1, (0.5, 2)),           # nothing judged this round
    # Nothing judged at all: the leader becomes 0.0, not None.
    ([None, None], None, 0, (0.0, 1)),
    ([], None, 4, (0.0, 5)),
])
def test_update_leader(raws, leader, dry, expected):
    contenders = [beam(f"c{i}", raw=r) for i, r in enumerate(raws)]
    assert _update_leader(contenders, leader, dry) == expected


def test_update_leader_needs_more_than_the_margin_to_reset_the_streak():
    # Exactly 0.01 better is not a new best, but the leader still moves up.
    assert _update_leader([beam("c", raw=0.5 + 0.01)], 0.5, 2) == (0.5 + 0.01, 3)
    assert _update_leader([beam("c", raw=0.5 + 0.02)], 0.5, 2) == (0.5 + 0.02, 0)


def test_award_medals_sorts_in_place_and_medals_the_top_three():
    a, b, c, d, e = (beam(n, overall=o) for n, o in
                     zip("abcde", [0.1, None, 0.5, 0.5, -0.2]))
    e.medal = 1  # a medal from an earlier checkpoint is taken back
    contenders = [a, b, c, d, e]
    same = contenders
    _award_medals(contenders)
    assert contenders is same
    # Best overall first; a tie keeps list order; unjudged (None) last.
    assert names(contenders) == ["c", "d", "a", "e", "b"]
    assert [x.medal for x in contenders] == [1, 2, 3, 0, 0]


# ---------------------------------------------------------------------------
# _should_stop and _drop_ramblers
# ---------------------------------------------------------------------------


@pytest.fixture
def stop(cfg):
    return cfg(STOP_AFTER_FINISHED=3, STOP_MARGIN=0.25, CHURN_MARGIN=0.125)


def finished_with(*overalls):
    return [beam(f"f{i}", overall=o, done=True) for i, o in enumerate(overalls)]


@pytest.mark.parametrize("overalls, live_raws, expected", [
    ((0.75, 0.25, 0.0), [], True),        # enough finished, a clear leader
    ((0.5, 0.25, 0.0), [], True),         # a lead of exactly the margin
    ((0.5, 0.375, 0.0), [], False),       # the lead is too small
    ((0.75, 0.25), [], False),            # too few finished
    ((0.75, 0.25, 0.0), [0.5], True),     # the live beam is out of contention
    ((0.75, 0.25, 0.0), [0.85], False),   # ...but never while one is still in it
    ((0.75, 0.25, 0.0), [None], False),   # unjudged counts as in contention
    ((0.25, 0.75, 0.0), [], False),       # reads the list as ordered, best first
])
def test_should_stop(stop, overalls, live_raws, expected):
    live = [beam(f"l{i}", raw=r) for i, r in enumerate(live_raws)]
    assert _should_stop(finished_with(*overalls), live, leader=0.9) is expected


def test_should_stop_needs_two_finished_to_compare(stop, cfg):
    cfg(STOP_AFTER_FINISHED=1)
    assert _should_stop(finished_with(0.9), [], leader=0.9) is False
    assert _should_stop(finished_with(0.9, 0.1), [], leader=0.9) is True


@pytest.fixture
def ramble(cfg):
    return cfg(RAMBLE_RATIO=2.0, CHURN_MARGIN=0.125)


def test_drop_ramblers_drops_only_long_beams_that_are_also_behind(ramble):
    finished = [beam("f1", tokens=10), beam("f2", tokens=20)]  # limit 2.0 x 20 = 40
    live = [
        beam("short-behind", tokens=30, raw=0.1),
        beam("at-limit-behind", tokens=40, raw=0.1),
        beam("long-behind", tokens=41, raw=0.1),
        beam("long-strong", tokens=90, raw=0.85),
        beam("long-unjudged", tokens=100, raw=None),
    ]
    got = _drop_ramblers(live, finished, leader=0.9)
    assert names(got) == ["short-behind", "at-limit-behind", "long-strong", "long-unjudged"]
    assert got is not live


def test_drop_ramblers_with_nothing_finished_returns_the_beams_as_they_are(ramble):
    live = [beam("long", tokens=1000, raw=0.0)]
    assert _drop_ramblers(live, [], leader=0.9) is live


def test_drop_ramblers_can_drop_every_beam(ramble):
    live = [beam("long", tokens=100, raw=0.1)]
    assert _drop_ramblers(live, [beam("f", tokens=10)], leader=0.9) == []


# ---------------------------------------------------------------------------
# _pick_winner and _finalise_block
# ---------------------------------------------------------------------------


def test_pick_winner_a_whole_finished_thought_beats_a_better_live_one():
    done = beam("A finished thought.", raw=0.6, done=True)
    live = beam("A longer thought that is still being written", raw=0.9)
    assert _pick_winner([done], [live]) is done


def test_pick_winner_ignores_a_finished_thought_below_the_length_floor():
    stub = beam("!", raw=0.5, done=True)
    live = beam("x" * 100, raw=0.75)
    assert _pick_winner([stub], [live]) is live


def test_pick_winner_the_floor_is_a_tenth_of_the_longest_stripped_text():
    live = beam("x" * 100, raw=0.9)
    ten = beam("  " + "y" * 10 + "  ", raw=0.1, done=True)  # 10 once stripped
    nine = beam("y" * 9, raw=0.1, done=True)
    assert _pick_winner([ten], [live]) is ten
    assert _pick_winner([nine], [live]) is live


def test_pick_winner_breaks_ties_in_finished_then_live_order():
    # No finished thought qualifies, so every contender is in the running,
    # finished first. The medals order the other way round (beams +
    # finished); the two rules are deliberately separate.
    stub = beam("!", raw=0.75, done=True)
    live = beam("x" * 100, raw=0.75)
    assert _pick_winner([stub], [live]) is stub
    f1, f2 = beam("first", raw=0.5, done=True), beam("other", raw=0.5, done=True)
    assert _pick_winner([f1, f2], []) is f1


def test_pick_winner_counts_an_unjudged_thought_as_minus_one():
    unjudged, zero = beam("unjudged"), beam("judged", raw=0.0)
    assert _pick_winner([], [unjudged, zero]) is zero


def test_pick_winner_with_no_contenders_is_an_empty_thought():
    w = _pick_winner([], [])
    assert (w.text, w.score, w.parent, w.raw) == ("", 0.0, -1, None)


class _Meter:
    """Stands in for llm.UsageMeter: fixed usage and wall time."""

    def used(self):
        return {"requests": 7, "retries": 0, "prompt_tokens": 900, "completion_tokens": 120}

    def secs(self):
        return 2.5


def test_finalise_block_records_the_winner_and_the_cost():
    block = Block(id=4, tail="", crit=None, kind="open")
    done = beam("  The finished thought.  ", raw=0.87654, done=True)
    live = beam("still going", raw=0.2)
    ev = Engine._finalise_block(block, [done], [live], _Meter())
    assert ev == {"type": "monologue", "block": 4, "text": "The finished thought."}
    assert list(ev) == ["type", "block", "text"]
    assert block.beams == [done, live]
    assert block.monologue == "  The finished thought.  "  # stripped only in the event
    assert block.winner_score == 0.877
    assert (block.finished_count, block.explored_tokens, block.requests, block.secs) == (
        1, 120, 7, 2.5)


def test_finalise_block_scores_an_unjudged_winner_as_zero():
    block = Block(id=0, tail="", crit=None)
    Engine._finalise_block(block, [], [beam("only")], _Meter())
    assert (block.monologue, block.winner_score) == ("only", 0.0)


# ---------------------------------------------------------------------------
# _standardise_group
# ---------------------------------------------------------------------------

TWO_AXES = {
    "a": {"weight": 1.0, "levels": ["low", "high"]},
    "b": {"weight": 2.0, "levels": ["low", "high"]},
}


def group_of(*pairs):
    return [beam(f"g{i}", axes={"a": a, "b": b}) for i, (a, b) in enumerate(pairs)]


def test_standardise_group_by_range_clipped_then_weighted(cfg):
    cfg(AXES=TWO_AXES, AXIS_CLIP=2.0)
    group = group_of((0.0, 0.25), (0.5, 0.25), (1.0, 0.25))
    _standardise_group(group, ["a", "b"])
    # a: range 1.0, mean 0.5. b: no spread, so it divides by 1.0 and is 0.
    assert [c.axes["a_z"] for c in group] == [-0.5, 0.0, 0.5]
    assert [c.axes["b_z"] for c in group] == [0.0, 0.0, 0.0]
    assert [c.overall for c in group] == [-0.5, 0.0, 0.5]
    # The raw axes are left as they were.
    assert [(c.axes["a"], c.axes["b"]) for c in group] == [(0.0, 0.25), (0.5, 0.25), (1.0, 0.25)]


def test_standardise_group_clips_each_axis(cfg):
    cfg(AXES=TWO_AXES, AXIS_CLIP=0.25)
    group = group_of((0.0, 0.0), (0.5, 1.0), (1.0, 0.5))
    _standardise_group(group, ["a", "b"])
    assert [c.axes["a_z"] for c in group] == [-0.25, 0.0, 0.25]
    assert [c.axes["b_z"] for c in group] == [-0.25, 0.25, 0.0]
    assert [c.overall for c in group] == [-0.75, 0.5, 0.25]  # 1.0 * a_z + 2.0 * b_z


def test_standardise_group_reads_only_the_axes_it_is_given(cfg):
    cfg(AXES=TWO_AXES, AXIS_CLIP=2.0)
    group = group_of((0.0, 0.0), (1.0, 1.0))
    _standardise_group(group, ["a"])
    assert "b_z" not in group[0].axes
    assert [c.overall for c in group] == [-0.5, 0.5]


def test_standardise_group_of_one_scores_zero(cfg):
    cfg(AXES=TWO_AXES, AXIS_CLIP=2.0)
    [only] = group_of((0.9, 0.1))
    _standardise_group([only], ["a", "b"])
    assert (only.axes["a_z"], only.axes["b_z"], only.overall) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Event builders: exact keys, in order
# ---------------------------------------------------------------------------


def test_block_open_event(cfg):
    cfg(BEAM_WIDTH=4, THINK_BRANCH_K=5)
    block = Block(id=3, tail="so far", crit=0.9, gate={"crit": 0.9}, kind="gate")
    ev = _block_open_event(block, "<think>Hmm")
    assert list(ev) == ["type", "id", "kind", "width", "branch_k", "tail", "crit", "gate", "seed"]
    assert ev == {"type": "block_open", "id": 3, "kind": "gate", "width": 4, "branch_k": 5,
                  "tail": "so far", "crit": 0.9, "gate": {"crit": 0.9}, "seed": "<think>Hmm"}


def test_block_open_event_for_the_opening_thought_has_no_crit(cfg):
    cfg(BEAM_WIDTH=2, THINK_BRANCH_K=3)
    ev = _block_open_event(Block(id=0, tail="", crit=None, kind="open"), "<think>\n")
    assert (ev["kind"], ev["crit"], ev["gate"]) == ("open", None, None)


THINK_STEP_KEYS = ["type", "block", "step", "considered", "ranked_by", "beams", "pool",
                   "ended", "retired", "entered", "leader", "dry", "closed", "dropped",
                   "distinct", "backfilled"]


def test_think_step_event():
    block = Block(id=2, tail="", crit=None)
    a = beam("a", uid=0, parent=0, index=0)
    new = beam("new", uid=5, from_uid=0, parent=0, index=1, last=" x")
    c = beam("c", uid=2, parent=1, index=2)
    f = beam("f", uid=3, done=True)
    gone = beam("gone", uid=1, overall=0.123456)
    ev = _think_step_event(block, 4, cands=[object()] * 6, used_choice=False,
                           beams=[a, new, c], finished=[f], before={0, 1, 2, 3},
                           retired=[gone], entered=[new], leader=0.87654, dry=2,
                           closed=1, distinct=2)
    assert list(ev) == THINK_STEP_KEYS
    assert ev["beams"] == [_beam_json(b) for b in (a, new, c)]
    assert ev["pool"] == [_beam_json(f)]
    assert {k: ev[k] for k in THINK_STEP_KEYS if k not in ("beams", "pool")} == {
        "type": "think_step", "block": 2, "step": 4, "considered": 6, "ranked_by": "logprob",
        # Alive before, not carried forward, and not finished: uid 1 only.
        "ended": [1],
        "retired": [{"uid": 1, "overall": 0.123, "replaced": True}],
        "entered": [{"uid": 5, "from_uid": 0, "tok": " x"}],
        "leader": 0.877, "dry": 2, "closed": 1,
        "dropped": 6 - 3, "distinct": 2, "backfilled": 3 - 2,
    }


def test_think_step_event_before_anything_was_judged():
    ev = _think_step_event(Block(id=0, tail="", crit=None), 0, cands=[], used_choice=True,
                           beams=[], finished=[], before=set(), retired=[], entered=[],
                           leader=None, dry=0, closed=0, distinct=0)
    assert (ev["ranked_by"], ev["leader"], ev["retired"], ev["ended"]) == ("choice", None, [], [])


def test_standings_event(cfg):
    cfg(AXES=TWO_AXES)
    x = beam("x", uid=4, medal=1, done=True, overall=0.12345, raw=0.67891,
             axes={"a": 0.11111, "a_z": 9.0})
    y = beam("y", uid=6, medal=2)
    ev = _standings_event(Block(id=5, tail="", crit=None), [x, y], beams=[y], finished=[x])
    assert list(ev) == ["type", "block", "ranked", "live", "finished"]
    assert ev == {
        "type": "standings", "block": 5,
        "ranked": [
            {"uid": 4, "rank": 1, "medal": 1, "done": True, "overall": 0.123, "raw": 0.679,
             "axes": {"a": 0.111, "b": 0.0}},  # every configured axis, 0.0 if unread
            {"uid": 6, "rank": 2, "medal": 2, "done": False, "overall": None, "raw": None,
             "axes": {"a": 0.0, "b": 0.0}},
        ],
        "live": 1, "finished": 1,
    }
    assert list(ev["ranked"][0]) == ["uid", "rank", "medal", "done", "overall", "raw", "axes"]
