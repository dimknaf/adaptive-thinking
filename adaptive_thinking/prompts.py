"""What the engine says to the model: every prompt's wording.

These are MEASURED ARTEFACTS, not prose to tidy. Each axis, level and probe
was worded against an observed failure, and the engine runs at temperature 0
over a prefix cache, so a changed byte changes every result after it. The
comments say what each one was measured against.

Other modules read these as attributes of this module (prompts.AXES, not a
copy taken at import), so a test can patch them in one place.
"""

from . import chatml

# The system turn of a Conversation that is not given one.
DEFAULT_SYSTEM = (
    "You are a careful assistant. Answer directly and concretely. "
    "Do not pad, do not restate the question."
)

# The line that closes a block and hands control back to the answer. It is a
# fixed template, so the model can see earlier copies in its context and imitate
# them - measured, it wrote the nudge into its own monologue and the note ended
# up carrying it twice. This constant is both what we append and the marker used
# to cut a copied one back off.
NUDGE_PREFIX = "Right - back to the answer."

# The two Yes/No probes. Both are answered at a single position with one token,
# so their cost is the prompt only.
#
# The gate probe asks the model about its own next step. It no longer opens
# blocks - the levelled GATE_AXES further down do. The engine only asks it
# while config.GATE_LOG_YESNO is on, and records its answer next to the axis gate's.
GATE_PROBE = (
    "Pause. Looking at your answer so far, is the next thing you are about to "
    "say a point where getting it wrong would matter - a fact, a number, a "
    "name, a judgement call - so that it is worth considering alternatives "
    "before committing? Answer with one word, Yes or No."
)

# The leaf probe is ADVERSARIAL on purpose. It runs on the same weights that
# produced the branch, so asking "are you confident?" invites the model to
# rubber-stamp its own output. Asking it to find the fault gives a signal that
# can actually discriminate. Lower is better.
#
# Not read by the engine at present (see docs/CONFIG.md): thoughts are judged
# on the levelled AXES below instead.
LEAF_PROBE = (
    "Read the continuation above. Does it contain a statement that is wrong, "
    "unsupported, or inconsistent with what came before it? Answer with one "
    "word, Yes or No."
)

YES, NO = "Yes", "No"

# ---------------------------------------------------------------------------
# Judging a thought on axes, rather than picking a letter.
#
# Each axis is one closed question about ONE candidate, answered at a single
# position. Every (candidate, axis) pair rides in one batched request - four
# candidates on four axes is 16 reads - the same round trip a single Choice
# costs, and with no labelled options there is no position to anchor on.
#
# THE LEVELS ARE CONCRETE AND SITUATIONAL ON PURPOSE. An earlier measurement
# found a field scoring 3/100 forward and 98/100 reversed when its levels were
# bare magnitude words ("very poor / poor / fair"). Describing the actual
# situation instead is what made that bias go away, so do not "tidy" these
# into adjectives.
#
# Each axis names a failure that was OBSERVED in this project, not imagined.
# Levels run worst -> best; the score is the expected value over them, 0..1.
# ---------------------------------------------------------------------------
AXES = {
    # Beams were caught regurgitating an earlier block's text; one winner
    # scored -0.01 across 28 tokens doing nothing else.
    "looping": dict(
        weight=1.0,
        question="Read the reasoning above. How much of it is repetition?",
        levels=[
            "it repeats whole sentences it has already written",
            "it restates the same point in different words",
            "it circles back to earlier ground now and then",
            "it revisits a point only where that is genuinely needed",
            "every line adds something not already said",
        ],
    ),
    # Thirty tokens were once spent restating "The user is asking for the
    # number of minutes between 09:35 and 13:12".
    "progress": dict(
        weight=1.3,
        question="How far has this reasoning actually got?",
        levels=[
            "it is restating the question or the setup again",
            "it is circling the problem without committing to an approach",
            "it has set the work up but has not done it",
            "it has done part of the work",
            "it has reached a definite answer",
        ],
    ),
    # A branch produced "17:47 + 1:25 = 18:72" and carried on regardless.
    "confusion": dict(
        weight=1.6,
        question="Is the reasoning above internally sound?",
        levels=[
            "it states something false or impossible and continues anyway",
            "it contradicts something it said earlier",
            "a step does not follow from the one before it",
            "it is sound but one step is asserted without support",
            "every step follows from the one before it",
        ],
    ),
    # Hedging and drifting instead of committing to a step.
    "concreteness": dict(
        weight=0.7,
        question="How specific is this reasoning?",
        levels=[
            "it talks about the problem without naming any specifics",
            "it gestures at a method without applying it",
            "it names the quantities but does not use them",
            "it works with the actual numbers or facts given",
            "it carries the actual values through to a result",
        ],
    ),
}


def _levels(a):
    return "\n".join(f"{chr(65 + i)}: {t}" for i, t in enumerate(a["levels"]))


def _one_letter(question, options):
    """A question over lettered options, to be answered with one letter."""
    return question + "\n\n" + options + "\n\nAnswer with one letter."


def _levelled_body(a):
    """An axis's question with its levels lettered from A, worst first."""
    return _one_letter(a["question"], _levels(a))


def axis_prompt(context, thought, axis):
    """One axis, one candidate, one answer position."""
    a = AXES[axis]
    body = "Here is a piece of reasoning:\n\n" + thought.strip() + "\n\n" + _levelled_body(a)
    return chatml.closed_question(context, body)


# ---------------------------------------------------------------------------
# The gate, scored on levels instead of a coin flip.
#
# Judging a THOUGHT has used concrete levelled axes for a while (AXES above).
# Deciding whether to think AT ALL asked one binary Yes/No and thresholded it,
# which is a single flat number carrying no information about how critical, how
# hard, or how irreversible the next step is. The crude half of the system was
# the half that decides.
#
# It is nearly free to fix: answer_probs batches every (position, axis) pair
# into ONE request, and a request to the endpoint these numbers were measured
# on costs ~2674 ms whatever it carries, so 6 positions x 3 axes is 18 reads in
# the round trip the Yes/No probe already spends.
#
# Levels are CONCRETE AND SITUATIONAL for the same measured reason as AXES: a
# field scored 3/100 forward and 98/100 reversed when its levels were bare
# magnitude words. They run from "not worth stopping for" up to "stop and
# think", so the expected value IS the criticality.
# ---------------------------------------------------------------------------
GATE_AXES = {
    "stakes": dict(
        weight=1.4,
        question=(
            "You are part-way through writing an answer. Thinking about what "
            "you are about to say next: what rests on getting it right?"
        ),
        levels=[
            "it is a connecting phrase; nothing depends on how it is put",
            "it restates something already established above",
            "it adds a detail that could be corrected later at little cost",
            "it is a claim the reader would act on, and a wrong one misleads",
            "the rest of the answer is built on it, so if it is wrong "
            "everything after it is wrong too",
        ],
    ),
    "difficulty": dict(
        weight=1.2,
        question="How hard is the next step actually going to be to get right?",
        levels=[
            "the next words are the obvious continuation of the sentence",
            "it is recall of something standard and well known",
            "it needs a small step of reasoning unlikely to go wrong",
            "it needs a calculation or an inference where a slip is easy",
            "it is the kind of step where the first instinct is usually wrong",
        ],
    ),
    "commitment": dict(
        weight=1.0,
        question="How far does the next step lock the answer in?",
        levels=[
            "it can be rephrased or corrected in the very next sentence",
            "it is one point among several and the others stand without it",
            "it sets a direction that would be awkward but possible to change",
            "it picks one reading of the question over another",
            "it commits to an answer the rest of the reasoning cannot walk back",
        ],
    ),
}


def gate_prompt(context, axis):
    """One gate axis, one position, one answer token.

    Unlike axis_prompt this does NOT quote a piece of reasoning back: the gate
    asks about what comes NEXT, and the context it is asking about is the whole
    answer so far, which is already the prompt.
    """
    return chatml.closed_question(context, _levelled_body(GATE_AXES[axis]))


# The beam-selection question for the labelled Choice (tree.rank), a
# single-prompt ranking inspired by jevfire: every candidate in ONE prompt
# against verified single-token labels, answered at a single position, so the
# model RANKS them against each other. It was introduced because judging each
# candidate in isolation had been measured to give a 0.10-0.30 band across five
# near-identical continuations - almost no discrimination. The engine does not
# call it at present (see config.CHOICE_EVERY); experiments/tree_probe.py's
# label-position check does.
CHOICE_QUESTION = (
    "You are part-way through working something out, and these are the ways the "
    "reasoning could carry on. Which one is the most solid and promising - clean, "
    "correct, and actually getting somewhere, rather than repeating itself, "
    "drifting off, or hedging?"
)


def choice_prompt(context, options, labels, question=None):
    """One prompt, every option labelled, one answer position."""
    lines = "\n".join(f"{lab}: {opt}" for lab, opt in zip(labels, options))
    return chatml.closed_question(context, _one_letter(question or CHOICE_QUESTION, lines))


def probe_prompt(context, question):
    """Close the assistant turn, ask `question`, open a fresh answer position."""
    return chatml.closed_question(context, question)
