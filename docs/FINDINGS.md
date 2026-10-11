# Findings

This document lists every measurement in this repository, positive and
negative, and records where each one came from. The design changed several
times, and most of those changes happened because a measurement forced them.
Many numbers below were therefore taken on an earlier version of the engine.
Each one says which. Re-running a script today measures the current engine,
which will not necessarily reproduce a historical number.

The negative results matter as much as the positive ones. Two of the most useful
results here say the idea did *not* work: hidden blocks are a no-op
([2.1](#21-a-hidden-block-is-a-no-op-by-construction)), and a labelled Choice
did not transfer to beam selection
([3.5](#35-a-labelled-single-token-choice-did-not-transfer)).

## Reading the provenance

Every measurement was taken on `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`, an NVFP4
quantisation of Qwen3.8-27B, served by vLLM at temperature 0.

**Engine eras.** The engine was rebuilt twice during development. Each result
below names the era it was measured on.

| era | what the engine was |
|---|---|
| v1 prototype | A binary Yes/No gate probe over an entropy floor. A block branched once on the top 5 tokens and ran five independent 48-token continuations, and the winner was committed into the answer. A late v1 revision added notes that stay in place and age-based eviction. |
| v2 engine | Token-synchronous beam search: width 5, five alternatives per beam, one token per step, up to 32 thinking steps per block, the same binary gate. The answer was capped at 512 tokens and 8 blocks. Nothing was committed into the answer. |
| current engine | An opening thought every turn, a gate scored on three levelled axes, and a chunked tree: 500-token checkpoints, width 4, five alternatives. Also a first-token fan-out, a four-axis judge and churn. No cap on thinking or answer length. |

**Scripts.**

- **Live measurement scripts** in `experiments/`:
  `experiments/tree_probe.py`, `experiments/context_probe.py`,
  `experiments/gate_audit.py`, `experiments/bench_hard.py`,
  `experiments/zebra/run.py`. The first four sat in the repository root, and
  the zebra experiment in `zebra/`, when they produced the numbers below;
  `tree_probe.py` and `context_probe.py` were then named `test_tree.py` and
  `test_context.py`, and `degraded_check.py`, an offline scenario, was
  `test_degraded.py`. Several have been updated since they
  produced a historical number. `tree_probe.py` and `gate_audit.py` both last
  changed their measurement with the current engine.
- **Archived scripts** in `experiments/archive/`: `calibrate.py`,
  `probe_check.py`, `bench.py`, `inspect_inline.py`. They are kept unchanged
  and target an earlier engine, so they do not run as intended against the
  current code; see
  [experiments/archive/README.md](../experiments/archive/README.md).
- A number marked **recorded at the time** was written down when it was
  measured. No script in this repository reproduces it on its own.
- A claim marked **informal observation** was seen during development but
  never measured systematically, so treat it as weaker evidence.

**Determinism.** Temperature 0 makes repeated runs *nearly* identical, not
identical. Batched decoding on the server varies from run to run: identical
input to the current engine has taken 28 and 43 checkpoints on two runs. Early
notes treated repeats as fully deterministic. The practical conclusion still
holds: repeating a task measures little, and breadth has to come from more
tasks.

---

## 1. The gate

### 1.1 Entropy is not a gate

*v1 prototype · `experiments/archive/calibrate.py` (668 generated tokens),
`experiments/archive/probe_check.py` (116 positions)*

The original design used entropy as a veto: probe only where the model looks
unsure. The highest-entropy positions turned out to be almost entirely
**synonym choices**:

```
1.69  ' chose'        <- ' chose' | ' opted' | ' relied' | ' used'
1.67  ' inexpensive'  <- ' inexpensive' | ' cheap' | ' significantly'
```

Meanwhile the arithmetic prompts, where a wrong commitment cannot be recovered,
never reached the top of the ranking. The scale matters too. With top-5
logprobs, truncated entropy is bounded by ln 6 ≈ 1.79 nats, so a threshold that
looks reasonable, such as 2.0, can never fire.

The gate probe asked "is the next thing you are about to say worth considering
alternatives for?", and its answer, P(yes), came out independent of entropy:

| group | median P(yes) |
|---|---|
| high-entropy positions | 0.407 |
| mid-entropy positions | **0.484** |
| arithmetic prompts | **0.622** |
| open-ended essay prompt | 0.202 |

Across all positions P(yes) spread from 0.076 to 0.924. The gap between high-
and mid-entropy positions is **−0.077**: the two signals are independent, and
entropy is if anything slightly anti-correlated with what the probe responds to.
The strongest probe hit in the sample sat at H = 0.94.

**Consequence.** Entropy survives only as a floor (`ENTROPY_FLOOR = 0.4`), for a
mechanical reason that does not depend on correlation: if the top token holds
nearly all the mass, every branch is the same string and a block cannot change
anything. The binary probe these scripts measured no longer decides anything
([1.3](#13-the-levelled-gate-and-what-gate_threshold-actually-does)). The floor
is still in use.

### 1.2 The binary probe discriminated by task, much less by token

*v1 `experiments/archive/probe_check.py`; v2 `experiments/archive/bench.py` and
`experiments/gate_audit.py` (as it was then)*

- **By task, it worked.** Arithmetic had a median P(yes) of 0.622, against 0.202
  for essay prose (v1). Across 14 bench tasks it fired on every
  time-arithmetic problem with a carry and stayed silent on every
  single-operation one (v2).
- **Live audit (v2).** A prose prompt fired **0 of 16** times, with a median of
  0.072. Reasoning prompts had medians of 0.55 to 0.80.
- **By token, it was rougher.** 44% of firings landed on a number or a name,
  and 33% on punctuation or a bare function word. The gate also clustered: it
  once stopped three times in one clause to think about the word "the"
  (an informal observation).

`experiments/gate_audit.py` has since been updated to audit the levelled gate.
It still compares against the old probe, which is logged alongside while
`GATE_LOG_YESNO` is on.

### 1.3 The levelled gate, and what GATE_THRESHOLD actually does

*current engine · recorded at the time;
`adaptive_thinking/prompts.py` (`GATE_AXES`) and `adaptive_thinking/config.py`
(`GATE_THRESHOLD`); `experiments/gate_audit.py`*

The binary probe returns one flat number. It says nothing about how critical,
how hard or how irreversible the next step is. It was replaced by three axes,
each a closed question with five concrete, situational levels, read at a single
answer position:

| axis | weight | lowest level | highest level |
|---|---|---|---|
| stakes | 1.4 | a connecting phrase; nothing depends on how it is put | the rest of the answer is built on it |
| difficulty | 1.2 | the obvious continuation of the sentence | the first instinct is usually wrong |
| commitment | 1.0 | can be corrected in the very next sentence | commits to an answer that cannot be walked back |

`crit` is the weighted mean of the expected level on each axis, an absolute
number from 0 to 1. It is deliberately *not* standardised across positions:
standardising would guarantee that some position in every window looks like the
best one. All (position, axis) pairs go into one batched request, so 6 positions
on 3 axes is 18 reads in the round trip the binary probe already spent.

The levels describe situations rather than using bare magnitude words ("very
poor / poor / fair"). The reason is an earlier measurement, made outside this
repository, in which a bare-adjective scale scored 3/100 in one option order and
98/100 in the reverse. That effect has not been re-measured here.

**First comparison.** On the first position measured, crit was 0.78 where
P(yes) was 0.037, at the word the whole answer turned on.

**The axes are saturated.** Over 14 probed positions, 12 fired. The median crit
of those that fired was 0.794, against 0.774 for those that did not, and every
position scored above 0.70. Stakes (0.81–0.93) and commitment (0.87–0.97) were
saturated, and difficulty (0.53–0.68) was the only axis with any spread.

**Setting the threshold.** Over 20 probed positions:

| threshold | fired |
|---|---|
| ≥ 0.55 (the original guess, still the code default) | 20/20 (100%) |
| ≥ 0.78 | 11/20 (55%) |
| ≥ 0.80 | 4/20 (20%) |
| ≥ 0.82 | 1/20 (5%) |

The whole distribution sat between 0.746 and 0.824. By axis: stakes 0.834–0.920,
commitment 0.888–0.964, difficulty 0.489–0.592. On one logic puzzle, the
threshold changed the run completely:

- **At 0.80:** six windows stayed quiet (best crit 0.750–0.795), and the turn
  had one block instead of nine. It finished in 319 s with a correct answer.
- **At 0.55:** the same turn had run for more than 19 minutes and was still
  going.

**Reading.** 0.80 is cost control, not judgement. It works here because it
happens to sit just above this answer's band, not because the gate has become
discriminating. On another question it may fire constantly again. The axes
have to separate before the number can mean more. `.env.example` ships 0.80,
and `adaptive_thinking/config.py` still defaults to the uncalibrated 0.55.
`experiments/gate_audit.py`'s threshold table stops at 0.75, so it does not
show the shipped value.

### 1.4 Where a block may open

*current engine · comments in `adaptive_thinking/config.py` (`WARMUP_TOKENS`),
`adaptive_thinking/gate.py` (`_eligible`) and `adaptive_thinking/text.py`;
recorded at the time*

- **Not inside the opening clause** (`WARMUP_TOKENS = 28`). Three blocks fired
  in an answer's opening sentence caused three restarts:

  ```
  To find the number of minutes between 09:35 and 13:12, we can
  To find the number of minutes between 09:35 and 13:12,
  To find the number of minutes between 09:35 and 13:12, we can break...
  ```

  The same prompt set, with three blocks fired deep in a structured answer,
  restarted zero times. Every restart observed was in the opening sentence.
- **Only at a sentence or clause boundary.** A new sentence was going to start
  there anyway, so `</think>` meaning "the answer begins"
  ([2.4](#24-think-means-the-answer-begins-here)) costs nothing.
- **Not inside text the answer has already written.** The observed loop went
  like this: a block rewinds, the model restarts its sentence, the gate fires
  on the restatement, and another block opens. With no cap on answer length,
  that never ends.
- **At least `MIN_GAP_TOKENS = 12` after the previous block.**

As a result, an answer shorter than `WARMUP_TOKENS` can never fire the gate.
Separately, it was observed that a single-operation question such
as `246 ÷ 3` never fires it, while time arithmetic with a carry does.

---

## 2. Keeping the thinking in context

### 2.1 A hidden block is a no-op by construction

*v2 engine · `experiments/archive/bench.py`, 14 arithmetic word problems with
checkable answers*

| mode | correct | time (s) | × | decode tokens | × | blocks per response | changed next token per response |
|---|---|---|---|---|---|---|---|
| plain | 12/14 | 14.1 | 1.0 | 178 | 1.0 | 0 | n/a |
| adaptive (block hidden) | 13/14 | 34.0 | 2.4 | 292 | 1.6 | 1.2 | **0.0** |
| inline (block in context) | 13/14 | 33.3 | 2.4 | 282 | 1.6 | 1.2 | **1.1** |

Across all 14 tasks (about 17 blocks), a hidden block never changed the next
token. This is not a tuning problem. With the block removed, the context after
it is byte-identical to the context before it, so at temperature 0 the model
writes what it was about to write. The tree runs, costs its tokens, and is
thrown away. So the hidden mode's 13/14 is **not** evidence that thinking
helped: it differs from plain only through run-to-run variation in batched
decoding. That mode is a control, not a treatment, and `KEEP_BLOCKS_INLINE`
defaults to true.

Inline blocks do change the next token, about once per response. But the
duplicated-span rate rose sharply:

```
repetition   plain 0.0025   adaptive 0.0023   inline 0.0338
```

That is fourteen times plain. And at n = 14 the accuracy difference is one task
either way. Inline got the train task wrong where the others got it right, and
the meeting task right where the others got it wrong. 13/14 against 12/14 is
not a result.

An earlier 6-task bench on the v1 prototype (recorded at the time) scored
plain 5/6, adaptive 6/6 and inline 6/6, at 2.6× and 3.2× plain's wall-clock
time. Its post-block logprob delta of −0.010 / −0.046 suggested the seam was
invisible to the model. The larger bench above, and the restarts in
[2.4](#24-think-means-the-answer-begins-here), showed that it was not.

### 2.2 A bare `<think>` restarts the answer

*v1 prototype · recorded at the time;
`experiments/archive/inspect_inline.py` is the diagnostic that shows the
model-facing context*

The model writes its own deliberation. No instruction prompt sits on top: the
input is just its half-finished answer and a seed. The seed cannot be bare:

| seed | what the model wrote |
|---|---|
| `<think>` | *restarts the answer:* "To find the number of minutes… we can break the time down into two parts: 1." |
| `<think>Hmm, hold on.` | "Let me double-check the subtraction. 13:12 minus 09:35. First, convert both to minutes since midnight…" |
| `<think>Actually, wait -` | "I need to be careful. Let me recalculate." |

Two words of seed make the difference between restarting and reflecting. The
mid-answer seed is `<think>Hmm, hold on.` (`INLINE_SEED`). The opening thought
uses the template's bare `<think>` plus a newline, because at the top of a turn
there is nothing to restart.

A related trap: `INLINE_SEED` contains `<think>`, which is one of the transcript
markers the engine cuts at. A beam's stored text must therefore hold only the
generated body. When the seed was kept in it, the marker cut retired every beam
on step 0 and blanked the monologue (v2).

### 2.3 The model copies what it can see, and the register bleeds

*v1 prototype and v2 engine · recorded at the time*

- **The closing line leaked into the answer.** With a bare seed, the model
  copied the block's closing line into the visible answer word for word.
  `<think>` is an ordinary token in this vocabulary, so nothing stops the model
  emitting it. A stray `<|im_start|>` also reached a visible answer, because
  the stop list held only `<|im_end|>`. Every string the model returns is now
  cut at the first transcript marker (`<|im_start|>`, `<|im_end|>`, `<think>`,
  `</think>`), and these cuts are reported as `spills`.
- **Then it leaked into the monologue.** With that fixed, the model copied the
  closing line into its own monologue, so the note carried it twice.
  `NUDGE_PREFIX` is now both the line that gets appended and the marker used to
  cut a copied one back off.
- **The register bled through.** With notes in context and no line saying where
  thinking belongs, the answer filled with "(Wait, no. … let's re-read
  carefully)" and never concluded. The note's closing line now ends "Thinking
  stays in here; the answer itself gets none of it.", and answers conclude
  again.
- **One residue remains (v2):** a long answer can still end with a
  `*(Self-Correction: …)*` parenthetical after its conclusion.
- **Banning strings does not work** (an informal observation).
  Blocking phrases at generation time was tried: banning both spellings of a
  word made the model emit nothing. The
  fixes above change the mechanism instead of suppressing symptoms.

### 2.4 `</think>` means "the answer begins here"

*current engine · recorded at the time; `adaptive_thinking/text.py`
(`_inline_note`, `_restarted`)*

In Qwen's template, `<think>…</think>` sits straight after the assistant header,
so `</think>` means "the answer begins here". A mid-answer block also closes
with `</think>`, and the model read that as a cue to begin the *whole* answer
again. The block's closing line said literally "Continuing from where I
stopped, no restart", and lost.

Two attempts were made, and the difference between them is the lesson:

1. **Quoting the answer's last words inside the block did nothing.** An
   instruction does not outrank the template.
2. **Replaying them after `</think>` works** (`ANCHOR_CHARS = 90`). What sits in
   the "answer begins here" slot is now the answer itself, already in flight at
   the word it stopped on.

Measured: before the change, one turn opened the same answer four separate
times, keeping 218 tokens for about eighty tokens of content. After it, the
turn kept 101 tokens and a single clean copy. The reader never sees the replay:
it lives in the note, which is context only.

**The model paraphrases rather than copies.** The openings differed ("It
depends on Bruno…", then "We cannot determine it.…") while the bodies came back
nearly word for word. A verbatim-prefix detector reported 0 restarts through all
four rewrites. `_restarted` now tests containment instead. Once 40 consecutive
characters of new text appear anywhere before the block, the answer is being
rewritten; the engine rolls back, drops the note and continues without it. It is
a backstop, not the fix.

**Still open.** Inline repetition was 0.0338 against plain's 0.0025 (v2, 2.1),
and a short run can still visibly duplicate an opening clause across the seam.

### 2.5 The opening thought belongs at the native position

*current engine · recorded at the time*

- **Before:** earlier versions filled Qwen's thinking slot, straight after the
  assistant header, with the empty block the template emits when thinking is
  disabled. The only thinking in a transcript was therefore mid-answer, the one
  position that is out of distribution. It needs a seed, a closing nudge, a
  boundary check and a warmup to work at all.
- **Now:** the same tree runs at the native position on every turn, before any
  answer token, seeded with the template's bare `<think>`. There is nothing to
  rewind or resume and no seam to repair. The result sits in the assistant
  header as a note like any other, and it ages out with the rest.
- **It costs time.** One opening thought took 374 s (recorded at the time).
  There is no recorded comparison of answers with `OPEN_THINK` on and off.
- **A note must end on a whole sentence.** An opening thought once ended
  "...the constraint: not both circles. So", immediately followed by
  `</think>`. That is a broken prompt in the position where the template is
  strictest. Both kinds of note are now cut back to the last complete sentence.

### 2.6 Blocks stay where they happened; eviction is by age

*v1 prototype for the design; v2 engine ·
`experiments/context_probe.py` for the numbers*

**Placement.** An earlier version dropped blocks at the end of a turn and
re-rendered them as a short line in a header at the top. The model then read
one string while generating and a different one afterwards, so every later
token sat on a different prefix and the KV cache was discarded for no reason.
Notes now stay at the character offset they were written at, and the same
`splice()` renders them during generation and on replay. The opening thought's
header is built by a single function (`think_head`) for the same reason.

**Eviction** is by age, and it deletes rather than summarises.
`experiments/context_probe.py` with small test watermarks (trigger at 260
tokens, keep 40):

```
turn 1: clock=110  notes=1
turn 2: clock=184  notes=1
>> EVICTED 2 blocks, oldest was 282 behind, 723 -> 570 tokens, cache was 78.0%
turn 4: clock=385  notes=3   all in place: OK
```

The gap between the two watermarks is the point. Eviction only *triggers* once
a note is `EVICT_AGE_TOKENS` old, and then it clears everything older than
`EVICT_KEEP_TOKENS` in one pass. A single threshold would fire continuously,
and every firing rewrites the prefix from that point on. The re-prefill is real
and is reported alongside the prefix-cache hit rate. Defaults: 100,000 and
20,000 tokens.

---

## 3. The tree, as built on the v2 engine

### 3.1 The first design was not a tree

*v1 prototype · recorded at the time; v2 engine ·
`experiments/tree_probe.py`*

The v1 block branched once on the top 5 tokens at the gate position, ran five
independent 48-token continuations and kept the best finished string. That is
five one-shot guesses. Nothing advanced together, nothing was ever pruned, and
a single token decided the whole shape of the search.

It also mostly explored typography. The first working run reported that 7 of 8
blocks "changed the output", and the changes were `' ('` against
`' including'` and `' $$'` against `' $'`. Adversarial scores for the branches
sat in a narrow 0.10–0.30 band. Once a branch had to normalise to different text
*and* beat the greedy choice by a margin, only 2 of 8 blocks on the same prompt
changed anything.

The v2 engine rebuilt the block as token-synchronous beam search. One request
advanced all five beams by one token, each proposed five continuations, and the
25 candidates were pruned straight back to 5. From
`experiments/tree_probe.py`:

```
lineages born 27, discontinued 41, splits 41
distinct parents surviving per step: 1:2  3:11  4:19
widest frontier seen: 5
```

### 3.2 An assertion that passes under collapse

*v2 engine · `experiments/tree_probe.py`*

The first check that this was a tree asserted that two beams share a parent. It
passed on 31 of 32 steps. That condition also holds when the frontier has
collapsed onto one lineage, and that is what was happening: the five
conversations were converging into paraphrases of one sentence.
`experiments/tree_probe.py` now reports pairwise frontier diversity, lineage births, deaths and splits, and
whether any beam carries the engine's own boilerplate. The lesson: measure
frontier diversity, not structure.

### 3.3 Cumulative logprob rewards copying

*v2 engine · `experiments/tree_probe.py`*

Before the fix, every surviving beam in blocks 1–3 contained "Right - back to
the answer", the closing line of an earlier block's note, which was sitting in
the context. Reproducing context costs almost no surprisal, so a
maximum-likelihood beam search *prefers* it. One winner scored **−0.01 across 28
tokens** doing nothing else, and the whole 32-step thinking budget went on
restating the question.

Three changes followed, each aimed at a failure that was actually seen:

- length-normalise the ranking score, so a short beam cannot hold the frontier
  on a near-zero total;
- charge a penalty when a candidate's tail already appears in the context
  (`COPY_PENALTY = 0.60`, over the last `COPY_NGRAM = 24` characters);
- retire a beam the moment it starts reproducing the closing line, instead of
  paying for it and truncating it afterwards.

Afterwards no beam carried the closing line, and scores sat in a believable
range of −0.02 to −6.1 instead of pinned near zero. All three measures remain in
the current engine.

### 3.4 Diversity has to be forced, and it still decays

*v2 engine · `experiments/tree_probe.py`*

Without a cap, the children of the highest-scoring lineage swept the frontier:
in a 124-step sample, six steps had collapsed to a single parent. Two standard
counter-pressures were added. `MAX_CHILDREN_PER_PARENT` (then 2) stops one
lineage taking every slot, and a sibling-token penalty applies diverse beam
search.

With both in place, frontier diversity across one block:

```
first step 0.760    quarter 0.293    middle 0.749    last step 0.159
```

Diversity oscillates and **still ends converged**. Three of the five beams
finished as "Let me break it down step by step. I need to find the time / the
number / how many". Beam search is a search for the single most probable
string, so convergence is its objective, not a bug in it. This is reported, not
tuned away.

**On the current engine:**

- **The cap is 1.** The k candidates from one 500-token chunk share all but
  their last token, and at temperature 0 they stay near-twins.
- **Separation is meant to come from elsewhere:** the first-token fan-out
  ([4.3](#43-fan-out-on-the-first-token)) and churn refills
  ([4.9](#49-churn_margin-and-the-spread-of-raw)). The prune backfill takes
  extra siblings only when the frontier cannot otherwise be filled, and the
  engine reports those slots as `backfilled` because they are near-twins.
- **`SIBLING_PENALTY` never applies** as the code stands. One beam's
  alternatives are always distinct tokens, and the set of seen tokens is reset
  for each beam.
- **Diversity has not been re-measured** on the chunked engine.

### 3.5 A labelled single-token Choice did not transfer

*v2 engine · `experiments/tree_probe.py`*

The technique was inspired by jevfire. It puts every option in one prompt, each
against a verified single-token label, and reads the answer at a single
position, so that the model compares the options instead of judging each one
alone. It was applied to ranking beam candidates on the train question (correct
answer 18:25):

| pruning | requests | seconds | answer |
|---|---|---|---|
| Choice every step | 72 | 52.7 | **18:12, wrong** |
| Choice every 4 steps | 48 | 36.0 | 18:25 |
| **cumulative logprob only** | **40** | **30.6** | 18:25 |

On 8 probes that listed the options in reverse order, the raw Choice picked a
**different winner 4 times**. It was reading the label position as much as the
text. Asking in both orders and averaging (`CHOICE_DEBIAS`) cancels that bias,
and both prompts ride in the same request.

**Why it failed.** The primitive works on semantically distinct options, such as
brake / coast / accelerate. Beam candidates at one step of a per-token tree
differ by a **single token**. There is nothing to choose between, so the model
falls back on the letter. Separately, judging each candidate in isolation with
an adversarial yes/no question gave a 0.10–0.30 band across five near-identical
continuations (v1), which is almost no discrimination.

**Status.** `CHOICE_EVERY` was set to 0 on this measurement. The chunked engine
changed the premise: candidates are now whole 500-token thoughts. The Choice has
not been re-measured there, and the current engine does not read `CHOICE_EVERY`
at all. Candidates are ranked on length-normalised logprob, and the survivors
are judged by per-candidate axis questions
([4.8](#48-two-scores-deliberately)), which have no labelled options spanning
candidates. As a result, `experiments/tree_probe.py`'s three Choice
configurations now use
the same settings, so comparing them measures only run-to-run variation. Its
reversed-order probe still calls the ranking function directly.

---

## 4. The chunked tree (current engine)

### 4.1 Checkpoints, not tokens

*current engine · `adaptive_thinking/config.py` comment (`THINK_CHUNK`),
recorded at the time*

A request to the test endpoint cost about 2,674 ms whatever it returned: a
one-token request cost the same as a fifty-token one. At one token per step,
that whole cost bought one token per beam. The tree now advances each beam by
up to `THINK_CHUNK = 500` tokens per request and branches at the end of the
chunk. `THINK_CHUNK` is not a cap on thinking. The loop keeps going until the
beams close their own thoughts. The default width went from 5 to 4 in the same
change.

### 4.2 Branching at the end of a chunk duplicated a word

*current engine · recorded at the time; `adaptive_thinking/tree.py`
(`_expand`)*

The top-k list comes back *at* the chunk's final token: it holds the
alternatives to the word just written, not candidates for the next one.
Appending an alternative to the whole chunk therefore duplicated that word.
Every beam of one run ended "...looking at at", "...person. IfIf", "...Carla is
married married". The stem is now the chunk minus its last token, and each
alternative replaces that token. Candidate 0 is "carry on exactly as you were".

### 4.3 Fan out on the first token

*current engine · recorded at the time; `adaptive_thinking/tree.py`
(`run_block`, `fan_out`)*

With 500-token chunks, most thoughts close inside their first chunk, so the
branch point at the end of a chunk was never reached. The result was four
one-shot guesses, judged once, and no search at all. Measured: the opening
thought on a "bat and ball" question finished in chunk 0, one card appeared,
and three lineages never existed.

Every block now starts with a one-token request that reads the distribution
over the first thinking token. It opens `BEAM_WIDTH` conversations on the best
tokens that are not markers, whitespace or punctuation. At temperature 0, two
beams that share a prefix generate the same text, so beams not separated at the
start never separate later.

### 4.4 A top-k read can come back degenerate

*current engine · the `n_top` counts and the `'!'` stub were recorded at the
time; the full token lists are an informal observation*

Four byte-identical one-token requests, seconds apart:

```
trial 0: n_top=2   ['Let', '!']
trial 1: n_top=20  ['Let','The','We','This','Thinking']
trial 2: n_top=20  ['Let','The','We','This','Anna']
trial 3: n_top=20  ['Let','The','We','This','Anna']
```

One read in four came back truncated. The cause could not be determined from
the client side; concurrent batching on the server is one possibility.
`_fan_out` therefore reads again when the first read yields fewer openings than
the width, and keeps the better read. No single top-k read should be assumed to
be well formed.

### 4.5 Choosing the winner: three rules in turn

*current engine · recorded at the time;
`adaptive_thinking/tree.py` (`_pick_winner`)*

1. **The first finished thought won.** The `'!'` stub from
   [4.4](#44-a-top-k-read-can-come-back-degenerate) closed itself immediately,
   was the only finished thought, and won its block with raw 0.536 against
   three live thoughts at 0.739–0.781. The block's monologue was one character.
2. **Then the best raw score won, with finishing only a tie-break.** A live beam
   at 0.79 then beat a finished thought at 0.78, and its truncated text became
   the monologue, ending mid-sentence: "...not both circles. So".
3. **Now, a complete thought beats a half-written one, subject to a
   substance floor.** A finished thought counts only if it is at least a tenth
   as long as the longest thought in the block. The floor is relative, like
   `RAMBLE_RATIO`: "!" against 9,450 characters fails, while four comparable
   thoughts all pass. Among the finished thoughts that qualify, the best raw
   score wins. If none qualifies, the best raw score among all contenders wins.

### 4.6 Only a bad score removes a live conversation

*current engine · recorded at the time*

Three kept monologues from one run all ended mid-sentence:

```
block 0 (9450 chars)  "...The constraint: not both circles. So"
block 1 (5708 chars)  "...I'll go with **Square** ... but I acknowledge"
block 2 (1849 chars)  "...But wait - the question is phrased as "What shape"
```

Four rules had removed conversations for reasons other than scoring badly:

- **A stall exit.** The block ended after `2 × CHURN_PATIENCE` checkpoints
  without a better score, even while beams were still writing. Block 0's five
  checkpoints match it exactly.
- **The winner rule** in step 2 of [4.5](#45-choosing-the-winner-three-rules-in-turn).
- **A records rule.** It ended the block outright once three thoughts had
  finished and the leader was clear, killing every live beam, possibly
  including the best one.
- **A ramble rule.** It dropped any beam longer than 2.5× the longest finished
  thought. On a hard question, that is the beam doing the work, cut because a
  shallow one finished first.

The stall exit was removed: a thought ends when the model ends it. The winner
rule became step 3 of [4.5](#45-choosing-the-winner-three-rules-in-turn). The
records rule and the ramble rule now defer to `_in_contention(beam, leader)`,
the one test used wherever a live beam might be dropped. A live conversation
is kept while it is within `CHURN_MARGIN` of the best raw score in the block,
however long it has run and however stale the block around it has become. The
stop rule (at least `STOP_AFTER_FINISHED = 3` finished thoughts and a lead of
`STOP_MARGIN = 0.35`) and the ramble rule (`RAMBLE_RATIO = 2.5`) apply only to
beams that are out of contention.

### 4.7 A drifting score cannot drive a stopping rule

*current engine · `adaptive_thinking/tree.py` (`_update_leader`)*

`raw` rises simply because a longer thought scores better on "progress". So the
question "did anything beat the leader?" kept being reset by length alone, and
the block never wound down. One run went 20 minutes with nothing to show
(an informal observation). A new best must now clear the previous
one by 0.01. Because `raw` is bounded to 0..1, that also bounds how many times
the streak can reset.

### 4.8 Two scores, deliberately

*current engine · `adaptive_thinking/scoring.py` (`score_axes`)*

Each contender is judged on four axes: looping (weight 1.0), progress (1.3),
confusion (1.6) and concreteness (0.7). Each axis is a closed question with five
concrete levels, and every (candidate, axis) pair rides in one batched request.
Each axis is anchored in a failure that was actually seen. For example, the
confusion axis exists because a branch wrote "17:47 + 1:25 = 18:72" and carried
on regardless. Two numbers come out:

- **`overall`.** Each axis is standardised across the group (by range, not
  standard deviation, since a handful of candidates is too few for a stable
  estimate), clipped to ±`AXIS_CLIP` (2.0), then weighted and summed. It ranks
  like against like and means nothing across groups. Live beams and finished
  thoughts are standardised separately: a half-written thought scores badly on
  progress just for being half-written, and mixing the groups would retire live
  beams for being alive.
- **`raw`.** The plain weighted mean of the axis scores, from 0 to 1, and
  comparable across groups. Only `raw` can answer "can this live beam catch
  that finished thought?", so the contention test, the churn and the winner
  rule all use it.

The two should not be merged.

### 4.9 CHURN_MARGIN and the spread of raw

*current engine · `adaptive_thinking/tree.py` (`churn`); recorded at
the time. The spread below is an informal observation.*

The `raw` scores of the contenders in one run, over four checkpoints:

```
0.735  0.713  0.743  0.563     spread 0.18   <- one clearly bad thought
0.794  0.774  0.731  0.743     spread 0.06
0.851  0.852  0.816  0.789     spread 0.06
0.832  0.827  0.823  0.816     spread 0.02   <- converged, nobody should go
```

A normal field spreads 0.02–0.06. `CHURN_MARGIN = 0.12` was set as a guess
when the churn was added, before this spread was recorded. Against
this spread it fires only on a field like the first row: a conversation leaves
when it is clearly not a challenger, not merely because it is last. Lowering
the margin towards 0.06 would start retiring the runner-up.

**How the churn works.** If the lowest-scoring live beam is `CHURN_MARGIN` or
more behind the leader, it retires. While the block is still finding
better thoughts, its slot goes to a variation of the best lineage: a sibling
candidate that the checkpoint would otherwise have discarded. After
`CHURN_PATIENCE = 2` checkpoints without a new best, the slot closes instead.
At most one beam leaves per checkpoint, never the last live beam, and never one
that has not been judged.

**Status.** One run's spread is not a calibration. `adaptive_thinking/config.py`
calls 0.12
provisional, and the churn had not been observed to fire in the recorded runs
up to and including the zebra experiment.

### 4.10 Surviving endpoint failures

*current engine · recorded at the time; `experiments/degraded_check.py`*

A gateway timeout (HTTP 524) killed a browser turn that had already spent 374 s
on its opening thought. The finished thinking was lost, and the whole turn
surfaced as an error. The response has two layers:

- **Retry** (`adaptive_thinking/llm.py`). Statuses 429, 500, 502, 503, 504 and
  524, and transport
  errors, are retried twice with a linear backoff. A retry is cheap because the
  failed request's prefix is now cached on the server. A streaming request is
  retried only before it has produced a token, because replaying one that has
  would write the answer twice.
- **Degrade** (`adaptive_thinking/engine.py`, `Engine.respond`). If the answer
  stream still fails, the turn
  finishes with what it has: the opening thought in the header and the partial
  answer in the transcript. The stats and the UI mark the answer incomplete. An
  error that is not transient still raises: being wrong quietly is worse than
  failing loudly.

`experiments/degraded_check.py` runs this scenario offline against a stub
endpoint, and `tests/test_degraded_turn.py` asserts it in the test suite. One gap
remains: a failure *during* the opening thought is outside the degrade path, so
once the retries are exhausted the turn still ends with an error.

---

## 5. The zebra comparison and `reasoning_effort`

*current engine · `experiments/zebra/run.py`, `experiments/zebra/results.json`,
recorded at the time*

On the one 5x5 zebra puzzle that Qwen's own thinking failed to solve (seed 1,
19 clues), the tree passed twice, in 1,620 s and 3,703 s. The native run failed
0/25 after 1,562 s. It ended its turn mid-word, still inside `<think>`, having
used about 25k tokens of a 261,122-token budget. That figure is an estimate:
the run recorded 89,395 characters, not a token count. Native solved the other
five puzzles. The tree was run only on the one native failed, so this is n = 1
and selected on failure.
[experiments/zebra/README.md](../experiments/zebra/README.md) has the full
table and caveats.

**Why the tree won, stated honestly.** It did not think deeper. The beams share
no partial results, so this is closer to parallel sampling with quality
selection than to search. In one of the passing runs, 14 thoughts completed in
the opening block, and most were probably no better than native's single
thread; only one had to get through. The raw native stream
(`experiments/zebra/native_5x5_s1_raw.txt`) shows genuine case-splitting,
contradiction after contradiction, never converging.

**`reasoning_effort` is only a sentence in the system prompt.** It was
recovered by posting chat messages to `/tokenize` and reading what the chat
template rendered:

| effort | what the template adds before the system prompt |
|---|---|
| low | "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration." |
| medium | nothing |
| xhigh | "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer." (the default on the server used) |

This engine drives raw `/v1/completions`, so no chat template runs and no
sentence is added unless the caller adds it. That is how
`experiments/zebra/run.py` gives both arms the identical dial. Runs made before
the dial existed carried no sentence, which is equivalent to medium, not to the
xhigh default.

The dial did nothing for a model that was stuck. Native produced *more* text at
low than without the sentence. That uninstructed native run on this puzzle is
not in `results.json`. For the tree, low changed the shape of the search: beams
finished far more readily (14 and 30 finished thoughts per block, against 1;
these counts are informal observations). But it cost more
overall, 3,703 s against 1,620 s, and in that three-block run one mid-answer
block took 51% of the time re-deriving a question the opening thought had
already settled.

---

## 6. Methodology lessons

*Sources: records made at the time; comments in
the engine (then `thinking.py`, now the modules of `adaptive_thinking/`),
`adaptive_thinking/server.py`, `experiments/bench_hard.py` and
`experiments/zebra/run.py`. The rest (the
earlier token caps of 4,000 and 16, the 65-character run, the corrupted JSONL
file and the banned-strings result) are informal observations.*

Each of these was learned by getting it wrong once.

- **Both arms of a comparison get identical limits.** A comparison that gave the
  native arm a 1,800 s timeout and the tree a 3,600 s timeout, in the same
  command line, reported native as failing. It was invalid. Token caps of
  1,600 and 4,000, and an accidental cap of 16, contaminated earlier attempts at
  the same comparison. If a limit applies to one arm, it applies to both or to
  neither. `experiments/zebra/run.py` gives native the server's
  `max_model_len`, less the prompt, and neither arm a wall-clock limit.
  `experiments/bench_hard.py` still caps only the native arm; this is a known
  issue.
- **Omitting `max_tokens` does not mean unlimited.** The completions API
  defaults it to 16. A run that "removed the limit" that way gave Qwen 65
  characters. For a genuinely uncapped run, read `max_model_len` from
  `/v1/models`: a number from the server, not a guess.
- **Write the raw stream to disk as it arrives.** Two native failures could not
  be inspected, because only character counts had been logged, so a stuck model
  could not be told apart from a broken harness. `experiments/zebra/run.py`
  now writes the stream to `experiments/zebra/native_raw.txt` as it goes.
- **Streamed usage never arrives on an aborted stream.** The engine aborts
  streams on purpose to open blocks, so decode tokens are counted locally. The
  server's `prompt_tokens` covers batched calls only and is not a total.
- **Check the server's queue before a long run.** `num_requests_waiting` on
  `/metrics` once reached 79 while a benchmark from this repository was
  running. Run one measurement at a time: two runs appending to the same JSONL
  file at once corrupted it.
- **Measure what the property means, not a structural proxy for it.** "Two beams
  share a parent" held 31/32 under total collapse
  ([3.2](#32-an-assertion-that-passes-under-collapse)).
- **Fix the mechanism, not the symptom.** Banned strings made the model emit
  nothing ([2.3](#23-the-model-copies-what-it-can-see-and-the-register-bleeds)).
- **Record which marker, and where.** A `<|im_end|>` that arrives in the same
  chunk as the answer's last word is the model ending its turn, and it is
  harmless. One that arrives mid-answer is a real spill. `spill_markers` in the
  stats records both, which a bare count could not.
- **The tree must receive the whole prompt.** For a while the block was given
  only the answer fragment, without the system turn, the question or the
  assistant header. The beams then deliberated over a bare fragment, and the
  judge appended `<|im_end|>` to it and built malformed ChatML.
- **Do not spread an event into a function with named parameters.** A logging
  helper took `(kind, **fields)`, so the day events gained their own `kind`
  field every turn failed with a `TypeError`. It takes a dict now.
- **`append()` moves an existing DOM node to the end.** Appending only the live
  cards sent them behind every discontinued one. Live cards are now placed
  first, in frontier order.

---

## 7. Open problems and what has not been measured

- **Diversity decays** across a block (0.760 → 0.159 on the v2 engine). Beam
  search converges by design. The current engine separates lineages at the
  fan-out and relies on churn refills for any later separation. Whether that
  holds diversity has not been measured.
- **Resumption still restarts sometimes**
  ([2.4](#24-think-means-the-answer-begins-here)).
- **The gate clusters, and its axes are saturated**
  ([1.2](#12-the-binary-probe-discriminated-by-task-much-less-by-token),
  [1.3](#13-the-levelled-gate-and-what-gate_threshold-actually-does)). The
  threshold mostly controls cost.
- **The churn is unvalidated** ([4.9](#49-churn_margin-and-the-spread-of-raw)).
- **Latency, not compute, is the cost.** Requests share a long cached prefix,
  and each checkpoint is two dependent round trips: one to advance the beams,
  one to judge them. Running
  the tree inside the inference engine rather than over HTTP should remove most
  of that overhead. It has not been tried.
- **Not measured on the current engine:**
  - an accuracy benchmark with checkable answers
    (`experiments/archive/bench.py` targets the v2 engine, and
    `experiments/bench_hard.py` gives the two arms different limits);
  - answers with the opening thought on against off;
  - the labelled Choice on 500-token candidates;
  - the tree on the five zebra puzzles native solved.
