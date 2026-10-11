"""The string helpers that decide what text reaches the model, and what the
reader sees.

Pure functions of their arguments, except that they read chatml, prompts and
config at call time: which markers end the answer (cut_at_marker), what the
model sees with the thinking notes put back in (splice, head_note), the
repetition and boundary checks the gate and the resume watch rely on, and the
note a mid-answer block leaves in the context (_inline_note).
"""

from . import chatml, config, prompts

def cut_at_marker(text):
    """Truncate at the first transcript marker. Returns (clean, marker_or_None).

    Every string that comes back from the model goes through this. `stop` alone
    is not enough: vLLM streams the token that triggers a stop before applying
    it, and a marker can arrive inside a chunk that also carries real text.
    """
    hit, at = None, len(text or "")
    for m in chatml.SPILL:
        i = (text or "").find(m)
        if i != -1 and i < at:
            at, hit = i, m
    return (text or "")[:at], hit


def collapse_ws(text):
    """Every run of whitespace as a single space, none at either end."""
    return " ".join(text.split())


def _already_written(vis, off, span=48):
    """Is the text just before `off` a repeat of something earlier?

    The observed loop: a block rewinds, the model restarts its sentence, the
    gate fires on the restatement, blocks again, restarts again. Nothing caps
    the answer any more, so it runs forever. If the run-up to a position
    already appears earlier in the answer, the model is going in circles and
    another block will only deepen it.
    """
    if off < span:
        return False
    tail = vis[off - span : off]
    return tail in vis[: off - span]


def _restarted(vis, off, span=40):
    """Has the text written since `off` started reproducing the answer?

    The measured failure: a block closes, and because `</think>` means "the
    answer begins here" the model begins the ANSWER again. A first version of
    this asked whether the new text was a verbatim prefix of the old, and it
    reported zero while the model was in fact rewriting the same answer four
    times over - because it PARAPHRASES. The openings differed ("It
    depends on Bruno..." then "We cannot determine it.") while the body came
    back nearly word for word.

    So the test is containment, not prefix: once `span` consecutive characters
    of the new text appear anywhere before the block, the model is writing the
    answer again rather than continuing it. Only the most recent window is
    checked, which costs one substring search per token and still catches any
    repeat, because every window becomes the most recent one in turn.
    """
    new = vis[off:]
    return len(new) >= span and new[-span:] in vis[:off]


def _at_boundary(vis, off):
    """Does a new sentence or clause start at this position?

    `</think>` means "the answer begins here" in the Qwen template, so a block
    closing mid-sentence makes the model BEGIN rather than continue - a
    paragraph break and a restatement, which is the gap in the answer. Fire the
    gate only where a new sentence was going to start anyway and the template's
    meaning and ours agree.
    """
    before = vis[:off].rstrip()
    return (not before) or before[-1] in ".!?:;" or vis[:off].endswith("\n")


def head_note(notes):
    """The opening thought among a turn's notes, if it still has one."""
    return next((n for n in notes if n.off < 0), None)


def splice(vis, notes):
    """What the MODEL sees: the visible answer with thinking notes inserted.

    The same function serves generation and replay, which is the point - a turn
    renders identically after it finishes to how it read while it was running.

    The opening thought is skipped: it belongs to the header, not to the body,
    and render() puts it there.
    """
    notes = [n for n in (notes or []) if n.off >= 0]
    if not notes:
        return vis
    parts, prev = [], 0
    for n in sorted(notes, key=lambda n: n.off):
        parts.append(vis[prev : n.off])
        parts.append(n.text)
        prev = n.off
    parts.append(vis[prev:])
    return "".join(parts)


def _whole_sentences(text):
    """Cut back to the last complete sentence.

    A thought can stop part-way through one - the model was still writing when
    its beam left the frontier. Ending a note on a dangling "So" and then
    `</think>` is a broken prompt in the position where the template is
    strictest, and it reads as a truncation everywhere else.

    Both notes use this. The head note did NOT, which is how an opening
    thought ending "...the constraint: not both circles. So" reached the
    context verbatim.
    """
    t = (text or "").strip()
    cut = max(t.rfind(". "), t.rfind("? "), t.rfind("! "),
              t.rfind(".\n"), t.rfind("?\n"), t.rfind("!\n"))
    if t[-1:] in (".", "?", "!"):
        return t
    return t[: cut + 1] if cut > 40 else t


def _inline_note(block, tail=""):
    """The thinking block as the model reads it, mid-sentence.

    Two parts, and only the second one is ours.

    The MONOLOGUE is the model's own - the winning beam of the tree. No
    instruction prompt sits on top of it: the beams are seeded with a couple of
    words and continue from the model's own half-finished answer. Authoring the
    deliberation ourselves would make the block a record of a decision rather
    than the model actually thinking.

    The NUDGE is ours. It exists because this is out of distribution: the Qwen
    template puts <think> straight after the assistant header, never mid-answer,
    so nothing otherwise tells the model its sentence is still in flight. It
    also says where thinking belongs, because of a measured failure - with notes
    in context and no such line, the model kept the deliberating VOICE after the
    block closed, wrote "(Wait, no. let's re-read carefully)" into the answer,
    and ran out of budget before concluding. The markers were gone; the register
    was not.

    The whole thing is one line. The visible text was interrupted mid-line, and
    a multi-line block would leave the model looking at a fresh line start.
    """
    mono = collapse_ws(block.monologue or "")
    # The model can see earlier notes in its context and copies the nudge into
    # its own monologue, so the note would carry it twice. Cut our own template
    # back off before appending the real one.
    i = mono.find(prompts.NUDGE_PREFIX)
    if i != -1:
        mono = mono[:i]
    mono = _whole_sentences(mono)
    if mono:
        mono = mono.rstrip() + " "
    # THE ANCHOR GOES AFTER `</think>`, NOT INSIDE THE BLOCK.
    #
    # `</think>` means "the answer begins here" in this template, so whatever
    # follows it is read as the start of the answer. Leave nothing there and
    # the model begins the answer - the whole answer. Measured: four separate
    # openings of the same answer in one turn, 218 kept tokens for eighty
    # tokens of content.
    #
    # Quoting the tail INSIDE the block was the first attempt and it failed,
    # because an instruction cannot outrank the template. So the tail is
    # replayed after the close instead: the model finds the answer already in
    # flight at exactly the words it stopped on, and continues them. The
    # reader never sees the repeat - it lives in the note, which is context
    # only - so `vis` still holds each word once.
    anchor = _tail_words(tail, config.ANCHOR_CHARS)
    return (
        config.INLINE_SEED
        + " "
        + mono
        + f"{prompts.NUDGE_PREFIX} Thinking stays in here; the answer itself "
        "gets none of it."
        + "</think>"
        + anchor
    )


def _tail_words(tail, n):
    """The last <= n characters of `tail`, cut at a word boundary.

    Cutting mid-word would hand the model half a token sequence to continue,
    which is worse than handing it nothing.
    """
    t = (tail or "")[-n:]
    i = t.find(" ")
    return t[i + 1:] if i > 0 else t
