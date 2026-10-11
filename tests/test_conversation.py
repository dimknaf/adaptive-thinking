"""Where thinking sits in the context, and how a conversation renders.

A block stays at the character offset where it was written (splice), the
opening thought sits in the assistant header (head_note, HEAD_OFF), and
Conversation.render() turns all of it back into raw ChatML. The render has to
be byte-stable: a turn must read the same after it finishes as it did while it
was being generated, or every later request sits on a shifted prefix and the
server's prefix cache is thrown away.
"""

import random

import pytest

from adaptive_thinking import chatml, prompts
from adaptive_thinking.state import HEAD_OFF, Block, Conversation, Note, Turn
from adaptive_thinking.text import _inline_note, head_note, splice

S = "<|im_start|>system\n" + prompts.DEFAULT_SYSTEM + "<|im_end|>\n"


def _note(off, text, born=0):
    return Note(off=off, text=text, born=born)


# ---------------------------------------------------------------------------
# splice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vis, notes, expected", [
    ("abcdef", [_note(3, "X")], "abcXdef"),
    # Sorted by offset, whatever order they are stored in.
    ("abcdef", [_note(5, "B"), _note(1, "A")], "aAbcdeBf"),
    # The same offset keeps insertion order (a stable sort).
    ("abcdef", [_note(3, "X"), _note(3, "Y")], "abcXYdef"),
    ("abc", [_note(0, "X")], "Xabc"),
    ("abc", [_note(3, "X")], "abcX"),
    ("abc", [_note(10, "X")], "abcX"),
    # The opening thought belongs to the header, not the body.
    ("abc", [_note(HEAD_OFF, "H")], "abc"),
    ("abc", [_note(HEAD_OFF, "H"), _note(1, "N")], "aNbc"),
    ("abc", [], "abc"),
    ("abc", None, "abc"),
    ("", [_note(0, "X")], "X"),
])
def test_splice(vis, notes, expected):
    assert splice(vis, notes) == expected


def test_splice_does_not_reorder_or_modify_the_notes_list():
    notes = [_note(5, "B"), _note(HEAD_OFF, "H"), _note(1, "A")]
    before = list(notes)
    splice("abcdef", notes)
    assert notes == before


def _remove_in_offset_order(body, notes):
    """Delete each body note from `body`, in offset order, checking that each
    one sits exactly where its offset says it does."""
    shift = 0
    out = body
    for n in sorted((n for n in notes if n.off >= 0), key=lambda n: n.off):
        at = n.off + shift
        assert out[at:at + len(n.text)] == n.text, (n, out)
        out = out[:at] + out[at + len(n.text):]
        # Earlier notes are gone from `out`, so later offsets need no shift.
    return out


def test_splice_property_removing_notes_in_offset_order_gives_back_vis():
    rng = random.Random(2024)
    pieces = ["The ", "answer ", "is ", "4. ", "So ", "13:12", "\n", ", ", "x"]
    for _ in range(400):
        vis = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 15)))
        notes = []
        for k in range(rng.randint(0, 5)):
            off = HEAD_OFF if rng.random() < 0.15 else rng.randint(0, len(vis))
            notes.append(_note(off, f"<think>n{k} {rng.random():.3f}</think>", k))
        body = splice(vis, notes)

        assert len(body) == len(vis) + sum(len(n.text) for n in notes if n.off >= 0)
        assert _remove_in_offset_order(body, notes) == vis


def test_splice_notes_appear_in_offset_order():
    notes = [_note(9, "<C>"), _note(0, "<A>"), _note(4, "<B>")]
    body = splice("The answer.", notes)
    assert body == "<A>The <B>answe<C>r."
    assert body.index("<A>") < body.index("<B>") < body.index("<C>")


# ---------------------------------------------------------------------------
# head_note
# ---------------------------------------------------------------------------


def test_head_note_finds_the_opening_thought():
    head = _note(HEAD_OFF, "H")
    assert HEAD_OFF < 0
    assert head_note([_note(3, "x"), head]) is head
    assert head_note([head, _note(HEAD_OFF, "second")]) is head


def test_head_note_is_none_without_one():
    assert head_note([]) is None
    assert head_note([_note(0, "at offset zero is in the body")]) is None


# ---------------------------------------------------------------------------
# Conversation.render: exact byte shapes
# ---------------------------------------------------------------------------


def test_the_system_turn_falls_back_to_the_default():
    assert Conversation().system == prompts.DEFAULT_SYSTEM
    assert Conversation(system=None).system == prompts.DEFAULT_SYSTEM
    assert Conversation(system="").system == prompts.DEFAULT_SYSTEM
    assert Conversation(system="Be brief.").system == "Be brief."


def test_the_default_system_is_read_at_construction(cfg):
    cfg(DEFAULT_SYSTEM="Custom default.")
    assert Conversation().system == "Custom default."


def test_a_new_conversation_is_empty():
    conv = Conversation()
    assert conv.turns == [] and conv.clock == 0
    assert conv.live_notes() == []


def test_render_with_no_turns_is_the_system_turn():
    assert Conversation().render() == S
    assert Conversation(system="S").render() == "<|im_start|>system\nS<|im_end|>\n"


def test_render_pending_user_without_thinking_closes_the_thinking_slot():
    got = Conversation().render(pending_user="Q")
    assert got == (S + "<|im_start|>user\nQ<|im_end|>\n"
                   + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert got.endswith(chatml.NO_THINK)
    assert Conversation().render(pending_user="Q", think=False) == got


def test_render_pending_user_with_thinking_leaves_the_slot_open():
    got = Conversation().render(pending_user="Q", think=True)
    assert got == S + "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n"


def test_render_an_assistant_turn_with_a_header_and_an_inline_note():
    conv = Conversation(system="S")
    conv.turns += [Turn("user", "Q"),
                   Turn("assistant", "A.", notes=[_note(HEAD_OFF, "H"), _note(1, "<n>")])]
    assert conv.render() == (
        "<|im_start|>system\nS<|im_end|>\n"
        "<|im_start|>user\nQ<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nH\n</think>\n\nA<n>.<|im_end|>\n"
    )


def test_render_strips_the_opening_thought_in_the_header():
    conv = Conversation(system="S")
    conv.turns += [Turn("user", "Q"),
                   Turn("assistant", "A.", notes=[_note(HEAD_OFF, "  H \n")])]
    assert conv.render().endswith(
        "<|im_start|>assistant\n<think>\nH\n</think>\n\nA.<|im_end|>\n")


def test_a_turn_without_an_opening_thought_renders_no_think():
    conv = Conversation(system="S")
    conv.turns += [Turn("user", "Q"), Turn("assistant", "A.")]
    assert conv.render() == (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n"
        "<|im_start|>assistant\n" + chatml.NO_THINK + "A.<|im_end|>\n"
    )


def test_an_empty_opening_thought_renders_exactly_like_none():
    with_empty = Conversation(system="S")
    with_empty.turns += [Turn("user", "Q"),
                         Turn("assistant", "A.", notes=[_note(HEAD_OFF, "")])]
    without = Conversation(system="S")
    without.turns += [Turn("user", "Q"), Turn("assistant", "A.")]
    assert chatml.think_head("") == chatml.NO_THINK
    assert with_empty.render() == without.render()


def test_eviction_of_the_opening_thought_falls_back_to_no_think():
    conv = Conversation(system="S")
    notes = [_note(HEAD_OFF, "H"), _note(1, "<n>")]
    conv.turns += [Turn("user", "Q1"), Turn("assistant", "A.", notes=notes),
                   Turn("user", "Q2"), Turn("assistant", "B.")]
    before = conv.render()

    # What eviction does to a turn: its notes list loses the evicted ones.
    conv.turns[1].notes = [n for n in notes if n.off >= 0]
    after = conv.render()

    assert "<think>\nH\n</think>" in before
    assert after == before.replace("<think>\nH\n</think>\n\n", chatml.NO_THINK)
    # Only the evicted position onwards changes; the prefix before it is the
    # same bytes, so the cache holds up to there.
    k = before.index("<think>\nH")
    assert after[:k] == before[:k]
    # The mid-answer note is untouched.
    assert "A<n>.<|im_end|>" in after


def test_render_of_several_turns_in_order():
    conv = Conversation(system="S")
    conv.turns += [Turn("user", "Q1"), Turn("assistant", "A1."),
                   Turn("user", "Q2"),
                   Turn("assistant", "A2.", notes=[_note(HEAD_OFF, "T2")])]
    assert conv.render(pending_user="Q3", think=True) == (
        "<|im_start|>system\nS<|im_end|>\n"
        "<|im_start|>user\nQ1<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nA1.<|im_end|>\n"
        "<|im_start|>user\nQ2<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nT2\n</think>\n\nA2.<|im_end|>\n"
        "<|im_start|>user\nQ3<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_render_with_no_pending_user_ends_after_the_last_turn():
    conv = Conversation(system="S")
    conv.turns += [Turn("user", "Q"), Turn("assistant", "A.")]
    assert conv.render().endswith("A.<|im_end|>\n")
    assert conv.render(think=True) == conv.render()


# ---------------------------------------------------------------------------
# The KV-cache invariant: a turn renders the same after it finishes as the
# prompt that generated it
# ---------------------------------------------------------------------------


def _history():
    conv = Conversation()
    conv.turns += [Turn("user", "What is 2 + 2?"),
                   Turn("assistant", "4.", notes=[_note(HEAD_OFF, "Two and two.")])]
    return conv


def test_a_finished_turn_renders_as_it_was_generated_with_an_opening_thought():
    conv = _history()
    vis = "It is 4:43 pm. That is because 13:12 plus 3:31 carries an hour."
    head = "Add the minutes first, then carry."
    inline = _inline_note(Block(id=1, tail="", crit=0.9, monologue="12 + 31 is 43."),
                          vis[:15])
    notes = [_note(HEAD_OFF, head), _note(15, inline, 7)]

    # The prompt the engine sends while the answer is being written: the
    # header thought is appended to the prefix, the body goes through splice.
    during = (conv.render(pending_user="When?", think=True)
              + chatml.think_head(head) + splice(vis, notes))

    conv.turns += [Turn("user", "When?"), Turn("assistant", vis, notes=notes)]
    assert conv.render() == during + chatml.IM_END + "\n"


def test_a_finished_turn_renders_as_it_was_generated_without_one():
    conv = _history()
    vis = "Four. The sum of two and two is four."
    notes = [_note(6, "<think>Hmm, hold on. Sure.</think>The")]
    during = conv.render(pending_user="Again?", think=False) + splice(vis, notes)

    conv.turns += [Turn("user", "Again?"), Turn("assistant", vis, notes=notes)]
    assert conv.render() == during + chatml.IM_END + "\n"


# ---------------------------------------------------------------------------
# live_notes
# ---------------------------------------------------------------------------


def test_live_notes_returns_every_note_with_its_turn_in_order():
    h1, n1 = _note(HEAD_OFF, "H1"), _note(2, "N1")
    h2 = _note(HEAD_OFF, "H2")
    conv = Conversation()
    t1 = Turn("assistant", "A1.", notes=[h1, n1])
    t2 = Turn("assistant", "A2.", notes=[h2])
    conv.turns += [Turn("user", "Q1"), t1, Turn("user", "Q2"), t2]

    live = conv.live_notes()
    assert len(live) == 3
    # The pairs hold the turn and note objects themselves, so a caller can
    # remove a note from its turn.
    for (turn, note), (want_turn, want_note) in zip(
            live, [(t1, h1), (t1, n1), (t2, h2)]):
        assert turn is want_turn and note is want_note


def test_live_notes_follows_the_turns_notes_lists():
    conv = Conversation()
    t = Turn("assistant", "A.", notes=[_note(HEAD_OFF, "H"), _note(1, "N")])
    conv.turns += [Turn("user", "Q"), t]
    assert len(conv.live_notes()) == 2
    t.notes = t.notes[1:]
    assert [n.text for _, n in conv.live_notes()] == ["N"]
