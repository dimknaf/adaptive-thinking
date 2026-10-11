"""Every knob in one place. No other module reads os.environ.

Only what the environment can set lives here. The chat format is chatml.py and
the wording of every prompt is prompts.py. Other modules read each knob as an
attribute of this module when they need it (config.BEAM_WIDTH), never as a
copy taken at import, so a knob changed at runtime - server.py's `inline`
switch, a test - is seen everywhere at once.

Nothing here caps how much the model may think. A block ends when its beams
close it. The measurement behind each default is in the comment above it;
docs/CONFIG.md tabulates every knob, its default and its status.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# The .env at the REPOSITORY ROOT, one level above this package: where it sat
# next to config.py before the engine became a package, and where
# .env.example sits. It only fills variables the environment has not already
# set (override=False), so an exported variable always wins. Computed here
# rather than taken from paths.py so that this module imports nothing from
# the package and can be executed on its own. os.path.abspath rather than
# Path.resolve(), as in paths.py, so that ENV_FILE and paths.ROOT agree.
ENV_FILE = Path(os.path.abspath(__file__)).parents[1] / ".env"
load_dotenv(ENV_FILE, override=False)


def _f(name, default):
    return float(os.getenv(name, "") or default)


def _i(name, default):
    return int(os.getenv(name, "") or default)


def _b(name, default):
    return (os.getenv(name, "") or str(default)).strip().lower() in ("1", "true", "yes")


# --- endpoint ---------------------------------------------------------------
BASE_URL = (os.getenv("OPENAI_BASE_URL", "") or "").rstrip("/")
# /tokenize and /metrics live at the server ROOT, not under /v1: some servers
# expose /tokenize at the root only, and a /v1/tokenize request 404s there.
ROOT_URL = BASE_URL[: -len("/v1")] if BASE_URL.endswith("/v1") else BASE_URL
MODEL = os.getenv("OPENAI_MODEL", "") or ""
API_KEY = os.getenv("OPENAI_API_KEY", "") or ""
USER_AGENT = os.getenv("LLM_USER_AGENT", "") or "adaptive-thinking/0.1"
TIMEOUT = _f("LLM_TIMEOUT", 300)
# Transient failures are retried. If the endpoint sits behind a proxy or CDN,
# gateway timeouts such as 524 are transient; one once threw away a turn that
# had already spent 374 seconds on its opening thought. A retry is cheap: the
# prefix the failed request sent is now cached, so the second attempt
# re-prefills almost nothing.
#
# A streaming request is only retried BEFORE it has handed out a token -
# replaying one that has already emitted text would duplicate the answer.
LLM_RETRIES = _i("LLM_RETRIES", 2)
LLM_RETRY_BACKOFF = _f("LLM_RETRY_BACKOFF", 3.0)

# --- tier 1: entropy, as a FLOOR not a gate ---------------------------------
#
# The original design used entropy as a veto: probe only where entropy is high.
# Measured over 668 tokens, P(yes) at high-entropy
# positions was 0.407 against 0.484 at mid-entropy ones - a gap of -0.077, i.e.
# the two signals are independent and entropy is, if anything, slightly
# anti-correlated with what the probe cares about. The strongest probe hit in
# the whole sample sat at H=0.94, below any threshold worth setting.
#
# So entropy survives only as a FLOOR, and for a mechanical reason that does
# not depend on correlation: if the top token holds ~all the mass, all the
# branches are the same string and the block cannot change anything. Below the
# floor, branching is not a bad idea - it is a no-op.
ENTROPY_FLOOR = _f("ENTROPY_FLOOR", 0.4)
MIN_GAP_TOKENS = _i("MIN_GAP_TOKENS", 12)

# No block may open until the answer is past its opening clause.
#
# This is not a heuristic, it is a fix for a measured failure. Qwen's template
# is `<|im_start|>assistant\n<think>...</think>\n\n` followed by the answer, so
# `</think>` MEANS "the answer begins here". Close a block inside the opening
# sentence and "begin the answer" is the same string as "continue the answer",
# so the model begins it again:
#
#   "To find the number of minutes between 09:35 and 13:12, we can
#    To find the number of minutes between 09:35 and 13:12,
#    To find the number of minutes between 09:35 and 13:12, we can break..."
#
# Three blocks in the opener, three restarts. The same prompt set, with three
# blocks fired deep in a structured answer, restarted zero times. Every restart
# observed was in the opening sentence.
#
# It costs nothing: the opener is restatement boilerplate the beams already
# showed is not worth thinking about.
WARMUP_TOKENS = _i("WARMUP_TOKENS", 28)

# --- tier 2: the semantic gate, batched over candidate positions ------------
# Stream this many tokens, then score the best candidates inside the window on
# the gate axes (prompts.GATE_AXES) in ONE request. One round trip per window,
# not per token.
WINDOW_TOKENS = _i("WINDOW_TOKENS", 24)
PROBE_CANDIDATES = _i("PROBE_CANDIDATES", 6)
# The reference threshold for the logged Yes/No probe (GATE_LOG_YESNO). It
# does NOT decide whether a block opens: GATE_THRESHOLD does, on the levelled
# axes further down. The engine never reads this; experiments/gate_audit.py
# compares the Yes/No probe against the axis gate at this threshold, over one
# run.
#
# 0.60 sits between the measured essay median (0.202) and arithmetic median
# (0.622), so the old probe fired on facts and sums and stayed quiet through
# prose.
PROBE_THRESHOLD = _f("PROBE_THRESHOLD", 0.60)
PROBE_ENABLED = _b("PROBE_ENABLED", True)

# --- the thinking block: a lockstep tree, checkpointed per chunk -------------
#
# The block does NOT branch once and let K continuations run to completion.
# That was the first design and it was K one-shot guesses, not a search: the
# whole shape of the exploration was decided by a single token.
#
# Instead the beams advance in lockstep. The block fans out on its first token
# into BEAM_WIDTH conversations; at every checkpoint (every THINK_CHUNK tokens)
# each live beam proposes THINK_BRANCH_K alternatives for its last token,
# giving up to BEAM_WIDTH x THINK_BRANCH_K candidates, which are pruned straight
# back to BEAM_WIDTH. The width never grows: it stays BEAM_WIDTH.
BEAM_WIDTH = _i("BEAM_WIDTH", 4)
# Tokens each beam generates per request. A request to the endpoint these
# numbers were measured on costs ~2674 ms whatever it returns - a one-token
# request costs the same as a fifty-token one. At 1 that whole cost bought a
# single token per beam.
# This is NOT a limit on thinking: the loop keeps calling until a beam closes
# its own block. It only sets how often the tree checkpoints and branches.
THINK_CHUNK = _i("THINK_CHUNK", 500)
THINK_BRANCH_K = _i("THINK_BRANCH_K", 5)
# How many of the width*k candidates survive the free cumulative-logprob cut and
# go into the labelled Choice. Bigger = the Choice sees more, and one prompt
# carries them all, so this is cheap up to the point the prompt gets unwieldy.
# Not read by the engine at present (see docs/CONFIG.md): the Choice is not
# called, and only experiments/tree_probe.py prints this value.
CHOICE_SHORTLIST = _i("CHOICE_SHORTLIST", 8)

# --- keeping the conversations apart -----------------------------------------
#
# Beam search ranked purely by cumulative logprob is a search for the single
# most probable string, so it converges: measured at width 5, the five finished
# beams of a block were five paraphrases of one sentence. These knobs are what
# stop that, and each targets a failure that was observed rather than imagined.
#
# Survivors one chunk may contribute. ONE: the k candidates from a beam are the
# same 500-token text with a different final token, and at temperature 0
# generation is effectively deterministic, so two survivors from one chunk stay
# twins for the rest of the block and one slot does no work. Separation comes
# from the first-token fan-out instead, and from churn refills. _prune
# backfills past this cap only when the frontier cannot otherwise be filled.
MAX_CHILDREN_PER_PARENT = _i("MAX_CHILDREN_PER_PARENT", 1)
# How far behind the block's best thought a live conversation may be before it
# stops counting as a possible challenger. On the RAW 0..1 axis scale, so it
# compares across live and finished alike - a standardised score could not.
# Relative, like RAMBLE_RATIO, and it does nothing until a conversation has
# been judged on its own 500 tokens.
#
# 0.12 IS PROVISIONAL. In the one run it was read from, a normal field of
# contenders sat within 0.02-0.06 of each other and one clearly bad thought sat
# 0.18 behind, so 0.12 fires only on a clearly bad thought rather than on the
# runner-up. It did not fire at all in the zebra runs. Read the spread of
# `raw` off the standings in a real run before trusting it.
CHURN_MARGIN = _f("CHURN_MARGIN", 0.12)
# The margin of safety before slots start closing. A conversation behind the
# leader is REPLACED by a variation of a good branch while the block is still
# finding better thoughts; only once no new best has appeared for this many
# checkpoints does a retirement close the slot instead. At 1 the block would
# wind down on a single quiet round, which is too eager - one checkpoint
# proves nothing.
CHURN_PATIENCE = _i("CHURN_PATIENCE", 2)
# Charged to a candidate whose token an earlier sibling already took this step -
# plain diverse beam search, and free because the ranking is local.
SIBLING_PENALTY = _f("SIBLING_PENALTY", 0.35)
# Charged when a candidate's last COPY_NGRAM characters already appear in the
# context. Reproducing context has near-zero surprisal, so logprob REWARDS it:
# every beam of blocks 1-3 was regurgitating an earlier block's closing line,
# and one winner scored -0.01 across 28 tokens doing it.
COPY_PENALTY = _f("COPY_PENALTY", 0.60)
COPY_NGRAM = _i("COPY_NGRAM", 24)
# Run the labelled Choice (a single-prompt ranking inspired by jevfire) every
# Nth checkpoint. 1 = every one, 0 = never.
#
# Not read by the engine at present (see docs/CONFIG.md): _run_block always
# ranks candidates by cumulative logprob and then judges survivors on the AXES,
# so setting this changes nothing. Only server.py (which reports it) and
# experiments/tree_probe.py (which sets it) touch it.
#
# It was switched OFF on a measurement taken on the earlier one-token-per-step
# engine: the candidates differed by a SINGLE token, so there was nothing to
# compare and the model read the label instead of the text - 4 of 8
# reversed-order probes flipped the winner. It also cost one extra request per
# token, doubling every run. Candidates are now whole 500-token thoughts, so
# that premise no longer holds, and it has not been re-measured at THINK_CHUNK.
CHOICE_EVERY = _i("CHOICE_EVERY", 0)
# For the Choice (Engine._rank): ask it in BOTH option orders and average each
# candidate's probability, which cancels the positional preference above. Free:
# both prompts ride in the same batched request.
CHOICE_DEBIAS = _b("CHOICE_DEBIAS", True)

# How many alternatives are read at each ANSWER position while streaming. This
# feeds the entropy floor and the UI's token tooltip; it is not the beam width.
BRANCH_K = _i("BRANCH_K", 5)
# Leave the thinking block in the context the model reads. ON by default: the
# blocks are what the model carries forward, and they are what eviction later
# reclaims - this is a context-size mechanism, not a display option. With it off
# a block leaves no trace at all, continuation is seamless BY CONSTRUCTION, and
# nothing is being tested.
KEEP_BLOCKS_INLINE = _b("KEEP_BLOCKS_INLINE", True)
# What opens the monologue. Still no instruction on top - this is a seed, not a
# prompt - but it cannot be bare. Measured on this model: "<think>" alone makes
# it RESTART the answer ("To find the number of minutes... we can break the time
# down into two parts: 1."), while "<think>Hmm, hold on." makes it reflect
# ("Let me double-check the subtraction. 13:12 minus 09:35..."). Two words of
# seed is the difference between restarting and thinking.
INLINE_SEED = os.getenv("INLINE_SEED", "") or "<think>Hmm, hold on."
# --- the opening thought ----------------------------------------------------
#
# Qwen's template puts <think> straight after the assistant header, so that is
# where the model's own reasoning belongs and the one position where `</think>`
# meaning "the answer begins here" is simply TRUE. We used to suppress it -
# render() emitted the empty NO_THINK block and the answer started cold, so the
# only thinking in the transcript was the mid-answer blocks.
#
# That had it backwards. Mid-answer is the hard, out-of-distribution position:
# it needs a seed, a nudge, a boundary check and a warmup guard to work at all
# (see INLINE_SEED, NUDGE_PREFIX, _at_boundary, WARMUP_TOKENS). The easy
# position was the one going unused.
#
# So the tree runs at the native position, before the first answer token, every
# time. Same beams, same axes, same cards - it IS the model's own opening
# thought, searched instead of guessed. The triggered blocks carry on after it.
OPEN_THINK = _b("OPEN_THINK", True)
# Bare, unlike INLINE_SEED. Those two extra words exist because mid-answer
# "<think>" alone makes the model RESTART the sentence. Here there is nothing
# to restart, so the seed is the template's own opener and nothing of ours.
OPEN_SEED = os.getenv("OPEN_SEED", "") or "<think>\n"
# How much of the answer-so-far is replayed AFTER a block's `</think>`.
#
# `</think>` means "the answer begins here", so whatever follows it is read as
# the start of the answer. Leave nothing there and the model starts the answer
# over - measured, four separate openings of one answer in a single turn, 218
# kept tokens for eighty tokens of content, and it PARAPHRASED rather than
# copied, so a verbatim-prefix check reported none of it.
#
# Quoting the tail inside the block was tried first and failed: an instruction
# does not outrank the template. Replaying it after the close puts the answer
# already in flight at the words it stopped on. The reader never sees the
# repeat - it lives in the note, which is context only.
ANCHOR_CHARS = _i("ANCHOR_CHARS", 90)
# Tokens after a block whose logprobs are compared against the response mean.
RESUME_WATCH_TOKENS = _i("RESUME_WATCH_TOKENS", 12)
# The model's OWN <think> reasoning needs far more room than an answer, and it
# is the baseline that matters - if Qwen thinking for itself beats a tree of
# beams, the tree is not worth building. Only experiments/bench_hard.py uses
# this, and it caps the native arm alone - the tree arm is uncapped - so that
# comparison is not like for like (see its docstring).
# experiments/zebra/run.py gives the native arm the server's max_model_len,
# less the prompt, instead.
NATIVE_MAX_TOKENS = _i("NATIVE_MAX_TOKENS", 1600)

# --- context management -----------------------------------------------------
# Eviction is by AGE, where a block's age is how many tokens have been
# generated since it was written - not by transcript size. A block stays for
# exactly as long as it is still near the live end of the conversation.
#
# The two watermarks are deliberately far apart. Eviction only TRIGGERS once
# something is EVICT_AGE_TOKENS old, and when it fires it clears the whole band
# down to EVICT_KEEP_TOKENS in one pass. A single threshold would fire
# continuously, and every firing rewrites the prefix and discards the KV cache
# from that point on. One rare bulk rewrite is far cheaper than a steady drip.
EVICT_AGE_TOKENS = _i("EVICT_AGE_TOKENS", 100000)
EVICT_KEEP_TOKENS = _i("EVICT_KEEP_TOKENS", 20000)

# The demo server's port. The code default is 8099; .env.example sets 8123.
PORT = _i("PORT", 8099)

# --- judging thoughts, and when a block may end early -----------------------
# No single axis may decide a candidate on its own. Each axis is standardised
# across the candidates being compared (so an axis that merely VARIES more
# cannot outweigh one that MATTERS more), then clipped to this many units
# before the weighted sum. Range is used rather than standard deviation: with
# a handful of candidates an sd estimate is too noisy to divide by.
AXIS_CLIP = _f("AXIS_CLIP", 2.0)

# When to stop waiting for a better thought. For interchangeable draws the
# chance the next finisher is a new best is 1/(n+1): 1/2, 1/4, 1/6 ... records
# get rare fast. At three finished that is 25%, and if the leader is also clear
# of the runner-up there is little left to find. If it is close, the remaining
# beams are allowed to land. Neither of these is a cap on thinking - they are
# the point at which more thinking stops paying. And the block only stops this
# way once no live beam is still in contention (see scoring._in_contention):
# a conversation still scoring near the best is never cut off.
STOP_AFTER_FINISHED = _i("STOP_AFTER_FINISHED", 3)
STOP_MARGIN = _f("STOP_MARGIN", 0.35)
# A live beam grown this many times longer than the longest FINISHED thought
# may be rambling rather than thinking. Relative to the pool on purpose: it
# does nothing at all until something has finished, so it is not a fixed
# limit. Length alone never removes a beam: one past this ratio is dropped
# only if it is also out of contention on score.
RAMBLE_RATIO = _f("RAMBLE_RATIO", 2.5)

# --- the gate's decision ----------------------------------------------------
# The weighted mean of the gate axes, 0..1, at or above which a block opens.
#
# ABSOLUTE, and deliberately not standardised across the positions in a window
# the way scoring.score_axes standardises across candidates. That
# standardisation is right when a winner must be picked from a set; the gate
# decides WHETHER to fire, and standardising would guarantee that some position
# in every window looks like the best one.
#
# The code default remains the original, uncalibrated 0.55. The measured value,
# 0.80, ships in .env.example (see docs/CONFIG.md): over 20 probed positions
# 0.55 fired on all 20 and 0.80 on 4. Treat 0.80 as cost control rather than
# judgement - the scores were squeezed into 0.746-0.824 because the stakes and
# commitment axes saturate, so it picks roughly the top fifth by score. Running
# without the shipped .env.example therefore gives a gate that fires at nearly
# every candidate position. experiments/gate_audit.py reads the distribution
# back - every probed position is recorded on the window event, fired or not,
# which is what makes the calibration possible.
GATE_THRESHOLD = _f("GATE_THRESHOLD", 0.55)
# Keep the old binary probe running alongside and record it, so one run shows
# how the two signals relate (experiments/gate_audit.py compares them at
# PROBE_THRESHOLD).
# It is one EXTRA batched request per window: the two read different token
# sets (Yes/No against letters) and answer_probs renormalises over one set, so
# they cannot share a request without skewing both. Switching it off saves that
# request and drops `p_yes` from the window events; it does not change which
# positions open a block.
GATE_LOG_YESNO = _b("GATE_LOG_YESNO", True)


def summary():
    """The knobs a front end shows, under the names /config serves them by.

    server.py returns exactly this dict as /config, and static/index.html
    reads it by these names, so the keys and their order are a contract: add
    a knob here only together with the page that shows it, and never rename
    one. cli.py's settings line is drawn from it too. Like every knob, the
    values are read when it is called, not when this module was imported.
    """
    # Imported here, not at the top: this module imports nothing from the
    # package, so that it can be executed on its own (see ENV_FILE).
    from . import prompts

    return {
        "model": MODEL,
        "window_tokens": WINDOW_TOKENS,
        "entropy_floor": ENTROPY_FLOOR,
        "gate_threshold": GATE_THRESHOLD,
        "gate_axes": list(prompts.GATE_AXES),
        "probe_enabled": PROBE_ENABLED,
        "beam_width": BEAM_WIDTH,
        "think_branch_k": THINK_BRANCH_K,
        "think_chunk": THINK_CHUNK,
        "choice_every": CHOICE_EVERY,
        "open_think": OPEN_THINK,
        "keep_blocks_inline": KEEP_BLOCKS_INLINE,
        "evict_age_tokens": EVICT_AGE_TOKENS,
        "evict_keep_tokens": EVICT_KEEP_TOKENS,
    }
