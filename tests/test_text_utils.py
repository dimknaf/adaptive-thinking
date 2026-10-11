"""The string helpers in text.py that decide what text reaches the model.

cut_at_marker keeps transcript markers out of every string the model returns;
_already_written and _at_boundary are two of the gate's guards; _restarted is
the backstop that drops a note when the model starts the answer over;
_whole_sentences, _tail_words and _inline_note build the note a mid-answer
block leaves in the context.

All of them are pure, so every expected value here was derived by reading the
code and is asserted exactly.
"""

import random

import pytest

from adaptive_thinking import chatml, config, prompts
from adaptive_thinking.state import Block
from adaptive_thinking.text import (
    _already_written,
    _at_boundary,
    _inline_note,
    _restarted,
    _tail_words,
    _whole_sentences,
    cut_at_marker,
)

# The closing line _inline_note appends after the monologue.
CLOSING = (prompts.NUDGE_PREFIX
           + " Thinking stays in here; the answer itself gets none of it.")


def _block(monologue):
    return Block(id=0, tail="", crit=None, monologue=monologue)


# ---------------------------------------------------------------------------
# cut_at_marker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", chatml.SPILL)
def test_cut_at_marker_truncates_at_each_spill_marker(marker):
    assert cut_at_marker("Done." + marker + "junk") == ("Done.", marker)
    assert cut_at_marker(marker) == ("", marker)


@pytest.mark.parametrize("text, expected", [
    ("Hello<|im_end|>junk", ("Hello", "<|im_end|>")),
    # The earliest marker wins, whatever its order in chatml.SPILL.
    ("a</think>b<think>c", ("a", "</think>")),
    ("x<think>y<|im_start|>", ("x", "<think>")),
    ("plain", ("plain", None)),
    ("", ("", None)),
    (None, ("", None)),
    # A partial marker is not a marker.
    ("almost <|im_", ("almost <|im_", None)),
])
def test_cut_at_marker_examples(text, expected):
    assert cut_at_marker(text) == expected


def test_the_inline_seed_is_itself_cut_to_nothing():
    # INLINE_SEED contains "<think>". This is why a beam's text holds only the
    # generated body and the seed is prepended when a prompt is built: stored
    # in the beam, the seed would cut the whole thought away on its first step.
    assert "<think>" in config.INLINE_SEED
    assert cut_at_marker(config.INLINE_SEED) == ("", "<think>")


def test_cut_at_marker_reads_the_marker_list_at_call_time(cfg):
    cfg(SPILL=chatml.SPILL + ("@@",))
    assert cut_at_marker("keep@@drop</think>") == ("keep", "@@")


def test_cut_at_marker_property_prefix_before_the_first_marker():
    rng = random.Random(1234)
    alphabet = ["a", "b", " ", ".", "\n", "<", ">", "|", "/"] + list(chatml.SPILL)
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        clean, marker = cut_at_marker(text)
        assert text.startswith(clean)
        assert not any(m in clean for m in chatml.SPILL)
        if marker is None:
            assert clean == text
        else:
            assert marker in chatml.SPILL
            assert text[len(clean):].startswith(marker)


# ---------------------------------------------------------------------------
# _already_written: the gate does not fire inside text it has already written
# ---------------------------------------------------------------------------


def test_already_written_detects_a_repeated_run_up():
    first = "To find the minutes between 09:35 and 13:12, we first split it. "
    vis = first + "Then " + first
    assert len(first) >= 48
    assert _already_written(vis, len(vis)) is True


def test_already_written_is_false_for_new_text():
    vis = ("To find the minutes between 09:35 and 13:12, split it at the hour. "
           "From 09:35 to 13:35 is four hours, and 13:12 is 23 minutes short.")
    assert _already_written(vis, len(vis)) is False


def test_already_written_needs_a_full_span_before_the_position():
    assert _already_written("x" * 47, 47) is False
    # At off == span the run-up is compared against the empty string.
    assert _already_written("a" * 48, 48) is False
    assert _already_written("abcdefghij" * 10, 100) is True


def test_already_written_with_all_distinct_characters():
    vis = "".join(chr(33 + i) for i in range(94))
    assert _already_written(vis, 94) is False


def test_already_written_respects_a_custom_span():
    vis = "abc xyz abc"
    assert _already_written(vis, len(vis), span=3) is True
    assert _already_written(vis, len(vis), span=4) is False


# ---------------------------------------------------------------------------
# _restarted: containment, not prefix
# ---------------------------------------------------------------------------

BEFORE = "It depends on Bruno, whose status we do not know at all."


def test_restarted_catches_a_paraphrased_opening_with_a_verbatim_body():
    new = " We cannot say. " + BEFORE[10:]
    vis = BEFORE + new
    assert len(BEFORE) == 56
    # A prefix check misses this: the new text does not start the way the
    # answer did.
    assert not new.lstrip().startswith(BEFORE[:10])
    assert _restarted(vis, len(BEFORE)) is True


def test_restarted_is_false_for_a_genuine_continuation():
    assert _restarted(BEFORE + " Next, consider Carla.", len(BEFORE)) is False
    assert _restarted(BEFORE + " Next, consider Carla, who is married to Dan.",
                      len(BEFORE)) is False


def test_restarted_is_false_with_nothing_before_the_block():
    assert _restarted(BEFORE, 0) is False


def test_restarted_needs_a_full_span_of_repeated_text():
    body = BEFORE[10:]  # 46 characters, all present before the block
    assert _restarted(BEFORE + body[:39], len(BEFORE)) is False
    assert _restarted(BEFORE + body[:40], len(BEFORE)) is True
    assert _restarted(BEFORE + body[:6], len(BEFORE), span=6) is True


def test_restarted_checks_only_the_most_recent_window():
    # A repeat that has already been followed by 40 new characters is no
    # longer in the window. The engine calls this once per token, so in a real
    # turn the repeat would have been caught while it was the latest window.
    vis = BEFORE + " " + BEFORE[10:50] + " and then something different entirely."
    assert _restarted(vis, len(BEFORE)) is False


def test_restarted_fires_before_the_rewrite_finishes_when_checked_per_token():
    tokens = [" We", " cannot", " say."] + [" " + w for w in BEFORE[10:].split()]
    vis, fired_at = BEFORE, None
    for i, tok in enumerate(tokens):
        vis += tok
        if _restarted(vis, len(BEFORE)):
            fired_at = i
            break
    assert fired_at is not None
    # Not during the paraphrased opening, and before the last word.
    assert 3 <= fired_at < len(tokens) - 1


# ---------------------------------------------------------------------------
# _at_boundary: the gate only fires where a new sentence or clause starts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vis, off, expected", [
    ("", 0, True),
    ("   ", 3, True),
    ("Hello. World", 6, True),
    ("Hello. World", 7, True),
    ("Why? Because", 5, True),
    ("Stop! Now", 6, True),
    ("Step 1: add", 7, True),
    ("a; b", 2, True),
    ("line\nnext", 5, True),
    ("Hello, world", 7, False),
    ("Hello world", 6, False),
    ("Hello world", 3, False),
])
def test_at_boundary(vis, off, expected):
    assert _at_boundary(vis, off) is expected


def test_at_boundary_treats_a_decimal_point_as_a_sentence_end():
    # Documents current behaviour: the check looks at one character, so the
    # "." inside 3.14 counts as a sentence end. No test here relies on it as
    # a design goal.
    assert _at_boundary("pi is 3.14", 8) is True


# ---------------------------------------------------------------------------
# _whole_sentences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("I will add the hours first and then the minutes. So",
     "I will add the hours first and then the minutes."),
    ("First I check the hours, then the minutes carefully.\nThen",
     "First I check the hours, then the minutes carefully."),
    ("Is the total more than four hours in this case? Maybe",
     "Is the total more than four hours in this case?"),
    ("Ends cleanly.", "Ends cleanly."),
    ("  Done?  ", "Done?"),
    ("Wow!", "Wow!"),
    ("no punctuation here at all", "no punctuation here at all"),
    ("", ""),
    (None, ""),
])
def test_whole_sentences(text, expected):
    assert _whole_sentences(text) == expected


def test_whole_sentences_keeps_a_short_complete_part_uncut():
    # The last sentence end must be past character 40 to cut there, so a
    # short note keeps its dangling word rather than shrinking to a stub.
    assert _whole_sentences("Add them. So") == "Add them. So"
    assert _whole_sentences("a" * 40 + ". So") == "a" * 40 + ". So"
    assert _whole_sentences("a" * 41 + ". So") == "a" * 41 + "."


# ---------------------------------------------------------------------------
# _tail_words: the answer's last words, replayed after </think>
# ---------------------------------------------------------------------------


def test_tail_words_cuts_at_a_word_boundary():
    assert _tail_words("abc def ghi", 7) == "ghi"
    assert _tail_words("So the train arrives at", 90) == "the train arrives at"
    assert _tail_words(None, 5) == ""
    assert _tail_words("", 5) == ""


def test_tail_words_drops_the_first_word_even_when_nothing_was_cut():
    # Documents current behaviour: everything up to the first space is
    # dropped unconditionally, so a tail shorter than n still loses its
    # first word. The engine passes the last 240 characters with n = 90, so
    # in practice the tail is almost always longer than n.
    assert _tail_words("First step.", 90) == "step."


def test_tail_words_keeps_a_leading_space_when_the_cut_lands_on_one():
    # Documents current behaviour: a space at index 0 is not treated as a cut
    # point, so the returned tail starts with it.
    assert _tail_words("abc def ghi", 8) == " def ghi"


def test_tail_words_without_a_space_returns_the_last_n_characters():
    # Documents current behaviour: with no space to cut at, the result can
    # start mid-word.
    assert _tail_words("abcdefgh", 3) == "fgh"


def test_tail_words_property_is_a_bounded_suffix():
    rng = random.Random(99)
    words = ["the", "train", "arrives", "at", "13:12,", "so", "4h", "x", ""]
    for _ in range(300):
        tail = " ".join(rng.choice(words) for _ in range(rng.randint(0, 30)))
        n = rng.randint(1, 120)
        out = _tail_words(tail, n)
        assert tail.endswith(out)
        assert len(out) <= n


# ---------------------------------------------------------------------------
# _inline_note: what a mid-answer block leaves in the context
# ---------------------------------------------------------------------------


def test_inline_note_exact_shape_without_a_tail():
    note = _inline_note(_block("Let me check.\n  13:12 minus 09:35 is 3h37m."), "")
    assert note == (
        "<think>Hmm, hold on. Let me check. 13:12 minus 09:35 is 3h37m. "
        "Right - back to the answer. Thinking stays in here; the answer "
        "itself gets none of it.</think>"
    )


def test_inline_note_replays_the_tail_after_the_close():
    mono = "Let me check.\n  13:12 minus 09:35 is 3h37m."
    bare = _inline_note(_block(mono), "")
    note = _inline_note(_block(mono), "So the train arrives at")
    assert note == bare + "the train arrives at"
    # The replay is outside the block: `</think>` means "the answer begins
    # here", and what follows it is the answer already in flight.
    inside, after = note.split("</think>")
    assert "the train arrives at" not in inside
    assert after == "the train arrives at"


def test_inline_note_default_tail_is_empty():
    b = _block("Checked.")
    assert _inline_note(b) == _inline_note(b, "")
    assert _inline_note(b).endswith("</think>")


def test_inline_note_cuts_a_copied_nudge_so_it_appears_once():
    mono = "Fine. " + CLOSING + " extra words after the copy."
    note = _inline_note(_block(mono), "")
    assert note == "<think>Hmm, hold on. Fine. " + CLOSING + "</think>"
    assert note.count(prompts.NUDGE_PREFIX) == 1
    assert "extra words" not in note


def test_inline_note_with_an_empty_monologue():
    expected = "<think>Hmm, hold on. " + CLOSING + "</think>"
    assert _inline_note(_block(""), "") == expected
    assert _inline_note(_block(None), "") == expected


def test_inline_note_trims_a_dangling_sentence_from_the_monologue():
    mono = "First the hours: 13 minus 9 is four hours exactly. Then the minutes are"
    note = _inline_note(_block(mono), "")
    assert "First the hours: 13 minus 9 is four hours exactly. " + CLOSING in note
    assert "the minutes are" not in note


def test_inline_note_reads_its_knobs_from_config(cfg):
    cfg(INLINE_SEED="<think>Wait.", ANCHOR_CHARS=12)
    note = _inline_note(_block("Checked."), "the answer is forty-two")
    assert note.startswith("<think>Wait. Checked. ")
    # The last 12 characters are "is forty-two"; everything up to the first
    # space is dropped.
    assert note.endswith("</think>forty-two")


def test_inline_note_property_one_block_one_line_one_nudge():
    rng = random.Random(7)
    words = ["Let", "me", "check", "13:12", "minus", "09:35.", "So", "yes!",
             "\n", "  ", "Right", "-", "back", "to", "the", "answer."]
    for _ in range(300):
        mono = " ".join(rng.choice(words) for _ in range(rng.randint(0, 40)))
        if rng.random() < 0.3:
            mono += " " + CLOSING + " " + mono
        tail = " ".join(rng.choice(words[:10]) for _ in range(rng.randint(0, 50)))
        note = _inline_note(_block(mono), tail)

        assert note.startswith(config.INLINE_SEED + " ")
        assert note.count("<think>") == 1
        assert note.count("</think>") == 1
        assert note.count(prompts.NUDGE_PREFIX) == 1
        inside, after = note.split("</think>")
        # The block itself is one line: the answer was interrupted mid-line.
        assert "\n" not in inside
        assert inside.endswith(CLOSING)
        assert after == _tail_words(tail, config.ANCHOR_CHARS)
        assert tail.endswith(after)
