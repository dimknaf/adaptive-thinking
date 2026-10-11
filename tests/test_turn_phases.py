"""The phases of one turn, and the state they share.

Engine.respond keeps a turn's state in a _TurnState and runs the turn in
phases. Three of them touch nothing but that state and the conversation, so
they are driven directly here: _undo_restart rolls the answer back to a block
and drops its note, _commit_turn writes the finished turn into the
conversation, and _done_stats builds the done event's stats. The phases that
stream or think (_opening_thought, _stream_window, _think_at) are covered end
to end by test_respond.py and the goldens.
"""

import pytest

from adaptive_thinking.engine import Engine, _TurnState, _Window
from adaptive_thinking.events import _done_stats
from adaptive_thinking.state import HEAD_OFF, Block, Conversation, Note

# ---------------------------------------------------------------------------
# The shared state
# ---------------------------------------------------------------------------


def test_turn_state_starts_empty():
    st = _TurnState(prefix="P")
    assert (st.prefix, st.vis, st.notes, st.blocks) == ("P", "", [], [])
    assert (st.idx, st.wasted, st.last_block_at, st.next_id, st.resume_left) == (0, 0, 0, 0, 0)
    assert (st.spills, st.restarts, st.logprobs_all, st.logprobs_resume) == ([], [], [], [])
    assert st.resume_from is None and st.degraded is None
    assert st.pending_greedy is None and st.pending_block is None


def test_turn_states_never_share_a_list():
    one, two = _TurnState(prefix="P"), _TurnState(prefix="P")
    one.notes.append("n")
    one.spills.append("s")
    one.logprobs_all.append(-1.0)
    assert (two.notes, two.spills, two.logprobs_all) == ([], [], [])


def test_window_starts_empty():
    one, two = _Window(), _Window()
    one.cands.append("c")
    assert (two.cands, two.hit_stop, two.restarted) == ([], False, False)


# ---------------------------------------------------------------------------
# _undo_restart
# ---------------------------------------------------------------------------


def test_undo_restart_rolls_back_to_the_block_and_drops_its_note():
    head = Note(off=HEAD_OFF, text="HEAD", born=0)
    early = Note(off=3, text="<A>", born=1)
    mine = Note(off=10, text="<B>", born=4)
    notes = [head, early, mine]
    st = _TurnState(prefix="P", vis="0123456789 and it began again", notes=notes, idx=15,
                    resume_from=(10, 6), pending_greedy=" x",
                    pending_block=Block(id=1, tail="", crit=0.9), resume_left=3,
                    restarts=[{"at": 1, "rewrote": 2}])

    ev = Engine._undo_restart(st)

    assert ev == {"type": "restart", "at": 6, "rewrote": 15 - 6}
    assert list(ev) == ["type", "at", "rewrote"]
    assert (st.vis, st.idx) == ("0123456789", 6)
    assert st.notes == [head, early]
    # A new list, as the loop always built: the old one is left as it was.
    assert st.notes is not notes and notes == [head, early, mine]
    assert st.restarts == [{"at": 1, "rewrote": 2}, {"at": 6, "rewrote": 9}]
    assert (st.resume_from, st.pending_greedy, st.pending_block, st.resume_left) == (
        None, None, None, 0)


def test_undo_restart_drops_only_the_note_at_the_block():
    # The filter is `off != off0`: a note at any other offset stays.
    at, later = Note(off=10, text="<B>", born=4), Note(off=12, text="<C>", born=5)
    st = _TurnState(prefix="P", vis="x" * 20, notes=[at, later], idx=9, resume_from=(10, 6))
    Engine._undo_restart(st)
    assert st.notes == [later]


# ---------------------------------------------------------------------------
# _commit_turn
# ---------------------------------------------------------------------------


def test_commit_turn_writes_the_turn_and_advances_the_clock():
    conv = Conversation()
    conv.clock = 100
    head = Note(off=HEAD_OFF, text="H", born=100)
    note = Note(off=9, text="<N>", born=103)
    blocks = [Block(id=0, tail="", crit=None, kind="open")]
    st = _TurnState(prefix="P", vis="Four.   \n", notes=[head, note], blocks=blocks, idx=7)

    text = Engine._commit_turn(conv, st, "Q?")

    assert text == "Four."
    assert [(t.role, t.text) for t in conv.turns] == [("user", "Q?"), ("assistant", "Four.")]
    turn = conv.turns[1]
    assert turn.notes is st.notes and turn.blocks is blocks
    # Clamped to the stripped text; the header's note keeps its offset.
    assert [n.off for n in turn.notes] == [HEAD_OFF, 5]
    assert conv.clock == 107
    assert st.spills == []


def test_commit_turn_cuts_a_marker_that_survived_and_records_it():
    conv = Conversation()
    st = _TurnState(prefix="P", vis="Four.<|im_end|>junk", idx=4,
                    spills=[{"marker": "</think>", "at": 1, "kept": "", "empty": True}])
    assert Engine._commit_turn(conv, st, "Q?") == "Four."
    assert st.vis == "Four."
    assert st.spills[-1] == {"marker": "<|im_end|>", "at": 4, "kept": "", "empty": False}
    assert list(st.spills[-1]) == ["marker", "at", "kept", "empty"]
    assert len(st.spills) == 2


# ---------------------------------------------------------------------------
# _done_stats
# ---------------------------------------------------------------------------

STATS_KEYS = [
    "secs", "kept_tokens", "rewound_tokens", "blocks", "opening_thought",
    "blocks_that_changed_next_token", "think_steps", "explored_tokens", "spills",
    "spill_markers", "restarts", "restart_detail", "degraded", "retries",
    "notes_in_context", "requests", "prompt_tokens", "completion_tokens",
    "inline_blocks", "logprob", "resume_logprob", "resume_logprob_delta",
    "resume_samples",
]

USED = {"requests": 9, "retries": 1, "prompt_tokens": 800, "completion_tokens": 50}


def block(kind, changed, steps, explored):
    b = Block(id=0, tail="", crit=None, kind=kind)
    b.changed_next_token, b.steps, b.explored_tokens = changed, steps, explored
    return b


def test_done_stats_keys_and_values(cfg):
    cfg(OPEN_THINK=True, KEEP_BLOCKS_INLINE=False)
    spills = [{"marker": "<|im_end|>", "at": 7, "kept": ".", "empty": False}]
    restarts = [{"at": 3, "rewrote": 4}]
    st = _TurnState(
        prefix="P", idx=12, wasted=5, spills=spills, restarts=restarts,
        degraded="ReadTimeout: slow",
        blocks=[
            # The opening thought never counts as changing a token, even if
            # the flag were set: only gate blocks replaced one.
            block("open", True, 1, 10),
            block("gate", True, 2, 30),
            block("gate", False, 3, 40),
        ],
        logprobs_all=[-0.5, -0.25], logprobs_resume=[-0.75],
    )
    stats = _done_stats(st, 1.23456, USED, 2)
    assert list(stats) == STATS_KEYS
    assert stats == {
        "secs": 1.23, "kept_tokens": 12, "rewound_tokens": 5, "blocks": 3,
        "opening_thought": True, "blocks_that_changed_next_token": 1, "think_steps": 6,
        "explored_tokens": 80, "spills": 1, "spill_markers": spills, "restarts": 1,
        "restart_detail": restarts, "degraded": "ReadTimeout: slow", "retries": 1,
        "notes_in_context": 2, "requests": 9, "prompt_tokens": 800, "completion_tokens": 50,
        "inline_blocks": False, "logprob": -0.375, "resume_logprob": -0.75,
        "resume_logprob_delta": -0.375, "resume_samples": 1,
    }
    # The lists themselves, not copies.
    assert stats["spill_markers"] is spills and stats["restart_detail"] is restarts


def test_done_stats_rounds_each_mean_before_taking_the_delta(cfg):
    cfg(OPEN_THINK=False, KEEP_BLOCKS_INLINE=True)
    st = _TurnState(prefix="P", logprobs_all=[-0.00004], logprobs_resume=[-0.00006])
    stats = _done_stats(st, 0.0, USED, 0)
    assert (stats["logprob"], stats["resume_logprob"]) == (-0.0, -0.0001)
    # -0.0001 - -0.0, where the unrounded delta would round to -0.0.
    assert stats["resume_logprob_delta"] == -0.0001
    assert round(-0.00006 - -0.00004, 4) == 0.0


@pytest.mark.parametrize("every, resume", [([], []), ([], [-1.0]), ([-1.0], [])])
def test_done_stats_without_samples_has_no_delta(cfg, every, resume):
    cfg(OPEN_THINK=False, KEEP_BLOCKS_INLINE=True)
    st = _TurnState(prefix="P", logprobs_all=every, logprobs_resume=resume)
    stats = _done_stats(st, 0.0, USED, 0)
    assert stats["resume_logprob_delta"] is None
    assert stats["resume_samples"] == len(resume)


def test_done_stats_reads_the_knobs_when_called(cfg):
    st = _TurnState(prefix="P")
    cfg(OPEN_THINK=True, KEEP_BLOCKS_INLINE=False)
    first = _done_stats(st, 0.0, USED, 0)
    cfg(OPEN_THINK=False, KEEP_BLOCKS_INLINE=True)
    second = _done_stats(st, 0.0, USED, 0)
    assert (first["opening_thought"], first["inline_blocks"]) == (True, False)
    assert (second["opening_thought"], second["inline_blocks"]) == (False, True)
    assert (second["logprob"], second["resume_logprob"], second["degraded"]) == (None, None, None)
