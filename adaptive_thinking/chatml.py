"""The chat format, built by hand.

We drive /v1/completions with raw text rather than /v1/chat/completions,
because only the completions endpoint accepts a LIST of prompts - and
batching the branches into one request is the entire performance argument
for this design. That means building the chat template ourselves.

ChatML, as Qwen3 uses it. NO_THINK is what the Qwen chat template emits when
enable_thinking=False: an empty thinking slot. With OPEN_THINK on (the
default) the slot is filled by the tree's own opening thought instead (see
think_head), so NO_THINK is only the fallback: for a turn rendered with
think=False, for a past turn whose opening thought has been evicted, and for
every turn when OPEN_THINK is off. Either way the only thinking in the
transcript is what this engine's tree produced.

Every string here is byte-sensitive: generation runs at temperature 0 and
leans on the prefix cache, so a changed character changes every answer after
it. Other modules read these as attributes of this module (chatml.SPILL, not
a copy taken at import), so a test can patch them in one place.
"""

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
NO_THINK = "<think>\n\n</think>\n\n"

# Anything in this list means the model has stopped writing the answer and
# started writing the transcript. None of it may reach the reader.
#
# `<|im_start|>` is here because it was observed in a visible answer: STOP used
# to hold IM_END alone, and the token loop checked for IM_END alone, so a stray
# turn marker flowed straight through. `<think>` is here because once thinking
# blocks are left in the context the model imitates them and opens its own
# mid-answer - that is the reasoning text that spilled into the output.
SPILL = (IM_START, IM_END, "<think>", "</think>")
# `</think>` matters twice over: in the ANSWER it means the model opened a
# thinking block it should not have, and in a BEAM it means the thought is
# complete. Leaving it out meant a beam that had finished kept generating to
# the end of its chunk, so nothing ever closed and no block ever ended.
STOP = [IM_END, IM_START, "<think>", "</think>"]


def turn(role, text):
    return f"{IM_START}{role}\n{text}{IM_END}\n"


def assistant_header(think=False):
    """Open an assistant turn. `think` leaves the thinking slot OPEN.

    With think=False the slot is filled with the empty NO_THINK block, which is
    what the Qwen template emits for enable_thinking=False - the answer starts
    immediately. With think=True the header stops before it, so a <think> seed
    can be appended and the model reasons at its native position.
    """
    return f"{IM_START}assistant\n" + ("" if think else NO_THINK)


def think_head(text):
    """A finished thought sitting in the assistant header, Qwen's own shape.

    ONE source of truth on purpose. This string is built while the turn is
    generating and again every time the turn is re-rendered afterwards; if the
    two ever differ by a character, every later token sits on a shifted prefix
    and the KV cache is discarded for nothing. experiments/context_probe.py
    checks it.
    """
    return "<think>\n" + text.strip() + "\n</think>\n\n"


def closed_question(context, body):
    """Close the assistant turn, ask `body` as the user, open a fresh answer.

    The shape every closed question takes - the gate axes, the judge axes, the
    labelled Choice and the Yes/No probe - so the answer is read at the first
    position of a new assistant turn, with the thinking slot already empty.
    """
    return context + IM_END + "\n" + turn("user", body) + assistant_header()
