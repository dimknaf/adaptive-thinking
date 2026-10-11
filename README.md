# adaptive-thinking

A normal reasoning model thinks once, up front, and then answers. This project
asks whether a model can also decide *while it is writing* that a particular
next step is worth stopping for. Every turn opens with a thought in the model's
own `<think>` slot. Then, as the answer streams, a gate scores upcoming
positions. Where a position is judged critical enough, generation rewinds to it
and a small tree of candidate continuations (several parallel lines of thinking)
is explored and judged. The winning deliberation stays in the context at the
spot where it happened, and the model resumes and picks its own next token.
Everything runs against an OpenAI-compatible vLLM endpoint over HTTP: no changes
to the inference engine, no fine-tuning, no custom kernels. The server does need
a few specific request features; see [Requirements](#requirements).

**Status: research prototype.** It has been measured on one model, the sample
sizes are small, and it is slow: minutes for a hard answer, and up to an hour
on the zebra puzzle below. The results include the negative ones, which are
some of the most useful things here.

## How it works

```
EVERY TURN, before the first answer token
  the opening thought: the tree runs in the model's own <think> slot,
  the template's native position, so </think> means what the template
  says it means and there is no seam to repair                   OPEN_THINK

THEN the answer streams, one window at a time                  WINDOW_TOKENS
  │
  ├─ candidates: positions with entropy above a FLOOR and at least two
  │    distinct alternatives, past the opening clause, at a sentence or
  │    clause boundary                                         ENTROPY_FLOOR
  │
  ├─ ONE batched request scores the candidates on three levelled axes,
  │    stakes / difficulty / commitment -> crit, absolute 0..1      the gate
  │
  └─ best candidate with crit >= GATE_THRESHOLD opens a block
        rewind to that position
        ┌─ fan out: BEAM_WIDTH (4) different first tokens, ONE request
        │    (read a second time only if the first read is degenerate)
        ├─ every checkpoint (THINK_CHUNK = 500 tokens):
        │    ONE request advances every live beam by up to 500 tokens
        │    each beam proposes THINK_BRANCH_K (5) alternatives for its
        │    last token -> up to 20 candidates, pruned back to the width
        │    ONE request judges the contenders on four axes
        │    (looping / progress / confusion / concreteness)
        │    churn: a conversation clearly behind the leader retires; a
        │    variation of a good branch takes its slot, or the slot closes
        └─ the winning deliberation becomes the block's note, kept in context
        close, resume: the model picks its own next token
```

Invariants. Each one was violated by an earlier version, and each is enforced
now:

- **The width never grows.** Four beams with five alternatives each give 20
  candidates, and they are pruned straight back to at most four. The width
  never becomes 20, then 100.
- **The frontier shrinks only by finishing.** A thought that closes itself
  keeps its place in the pool and can still win. The only other way a slot
  closes is the churn retiring a conversation that scores clearly below the
  leader.
- **Only a bad score removes a live conversation.** A conversation is never
  removed for its length, for other thoughts finishing, or for the block
  going quiet. `_in_contention` is the single test, and it asks only about
  score.
- **A block commits no answer token.** It produces thinking. Whether the
  model's next token then differs from the one it was about to write is the
  measurement (`changed_next_token`).

Notes stay at the offset where they were written, so a turn renders the same
afterwards as it did while running. They are reclaimed later by age
(`EVICT_AGE_TOKENS` / `EVICT_KEEP_TOKENS`), and nothing is summarised.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) walks through the code.

## Headline result: one zebra puzzle

Of six certified zebra puzzles, Qwen's own thinking solved five. On the sixth,
a 5x5 puzzle, it failed, and the tree solved it on both of its runs. The puzzle
has 19 clues, pared down until no clue can be removed without losing
uniqueness, and a CSP solver certifies that it has exactly one solution.
Grading is an exact match on the whole grid. Neither arm was capped below the
model's context length or given a wall-clock limit. The native run and the
tree's effort=low run got the identical system prompt.

| 5x5, seed 1 | result | time | notes |
|---|---|---|---|
| native Qwen thinking, effort=low | **FAIL**, 0/25 cells | 1,562 s | ended its own turn mid-word, still inside `<think>`, after 89,395 characters (about 25k tokens by a character-count estimate, roughly 90% of a 261,122-token budget unused) |
| this engine, effort=low | **PASS**, 25/25 | 3,703 s | 3 blocks, 58,725 tokens explored |
| this engine, no effort sentence | **PASS**, 25/25 | 1,620 s | 1 block, 32,648 tokens explored |

Read it with these caveats:

- **n = 1.** One puzzle, two passing runs of the tree.
- **The puzzle was selected because native failed on it.** Native solved the
  other five puzzles (3x3 twice, 4x4 twice, and 5x5 seed 2 in 506 s, all
  without an effort sentence). The tree was not run on those five.
- **The effort settings do not line up.** The only recorded native run on
  this puzzle used effort=low. The tree's faster pass used no effort sentence.
- **Runs vary a lot.** Identical input at temperature 0 has taken 28 and 43
  checkpoints on different runs.
- **The win is not deeper search.** The beams share no partial results, so
  this is closer to parallel sampling with quality selection. Many thoughts
  complete and only one has to be right. Native has a single thread, and once
  that thread fell into a backtracking loop the whole run went with it.

The generator, solver, puzzles, runner, recorded results and the raw failing
native stream are in [experiments/zebra/](experiments/zebra/README.md). From
the repository root, with an endpoint configured (see
[Quickstart](#quickstart)):

```bash
python -m experiments.zebra.generate 5 1        # make and certify the puzzle (CPU only, several minutes)
python -m experiments.zebra.run native 5 1 low  # Qwen's own thinking
python -m experiments.zebra.run ours   5 1 low  # this engine
```

The committed puzzle is loaded if it exists, so the first line is only needed
to check the certification.

## What was measured

Full detail, with provenance for every number, is in
[docs/FINDINGS.md](docs/FINDINGS.md). Several results come from earlier versions
of the engine and are labelled as such. The eras are the v1 prototype (a binary
gate probe and one-shot branching), the v2 engine (a per-token tree of width 5),
and the current engine (the chunked tree described above).

- **Entropy is not a gate.** The highest-entropy positions were synonym
  choices (`' chose' | ' opted' | ' relied'`). The gate probe's P(yes) was no
  higher there than at mid-entropy positions (0.407 against 0.484, a gap of
  −0.077). Entropy survives only as a floor, because below it every branch is
  the same string. *(v1; `experiments/archive/calibrate.py`,
  `probe_check.py`)*
- **The gate discriminates by task, much less by token.** The old binary probe
  gave arithmetic a median P(yes) of 0.622 against 0.202 for essay prose, and
  on a prose prompt it fired at 0 of 16 probed positions. But 33% of its
  firings landed on punctuation or a bare function word. The current levelled
  gate is saturated on two of its three axes: over 20 probed positions, every
  score fell between 0.746 and 0.824. So `GATE_THRESHOLD=0.80` is cost control, not judgement.
  *(v1, `experiments/archive/probe_check.py`; v2, `experiments/gate_audit.py`
  as it was then; current engine)*
- **A block hidden from the model is a no-op by construction.** Remove the
  block from the context and the context after it is byte-identical to the
  context before it, so at temperature 0 the model writes the same next token.
  Across 14 tasks, hidden blocks changed the next token 0 times. Blocks kept
  in context changed it about once per response. Blocks must stay in context.
  *(v2; `experiments/archive/bench.py`)*
- **A bare `<think>` restarts the answer; two words of seed make the model
  reflect.** `<think>` alone produced "To find the number of minutes… two
  parts: 1.", while `<think>Hmm, hold on.` produced "Let me double-check the
  subtraction." *(v1 prototype)*
- **`</think>` means "the answer begins here", and it outranks any
  instruction.** After a mid-answer block the model started the whole answer
  again. Quoting the answer's last words inside the block did nothing.
  Replaying them after `</think>` fixed it: 218 kept tokens for about 80 tokens
  of content, down to 101. The model paraphrases rather than copies, so
  restarts are detected by containment rather than by prefix. *(current engine)*
- **Cumulative logprob rewards copying.** Reproducing text already in the
  context costs almost no surprisal, so beams regurgitated an earlier block's
  closing line, and one winning beam had a cumulative logprob of −0.01 across
  28 tokens. Length normalisation, a copy penalty and early retirement fixed
  it. *(v2; `experiments/tree_probe.py`)*
- **A labelled single-token Choice, inspired by jevfire, did not transfer to
  beam selection.** Ranking the candidates in one labelled prompt at every
  step cost 72 requests and 52.7 s and gave a wrong answer. Plain logprob
  ranking took 40 requests and 30.6 s and answered correctly. On 8
  reversed-order probes the winner flipped 4 times: the candidates differed by
  a single token, so the model read the label. Not re-measured on the chunked
  engine. *(v2; `experiments/tree_probe.py`)*
- **Diversity decays.** Frontier diversity fell from 0.760 at a block's first
  step to 0.159 at its last, and three of five beams finished as variants of
  one sentence. Beam search converges by design. This is reported, not
  fixed. *(v2; `experiments/tree_probe.py`)*

## Requirements

- **Python** 3.9 or later (developed and tested on 3.11).
- **An OpenAI-compatible vLLM server** whose `/v1/completions`:
  - accepts a **list** of prompts in one request (every batched step depends
    on this);
  - returns top-k logprobs up to 20 (vLLM's default `--max-logprobs`);
  - supports the `logprob_token_ids` request extension, used together with
    `return_tokens_as_token_ids`. `adaptive_thinking/llm.py` raises a clear
    error if the server ignores it.
- **`/tokenize` at the server root** (the base URL without `/v1`). Required.
  **`/metrics`** at the root is optional and only feeds prefix-cache
  reporting. `experiments/zebra/run.py`'s native arm also reads
  `max_model_len` from `/v1/models`.
- **A Qwen3-style ChatML model with `<think>`.** The chat template is built by
  hand in `adaptive_thinking/chatml.py`, because only the completions endpoint
  takes a list of prompts. Other model families need changes there.

All recorded measurements were taken on
`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`, an NVFP4 quantisation of Qwen3.8-27B.

## Quickstart

```bash
cp .env.example .env         # set OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_API_KEY
pip install -r requirements.txt

python -m adaptive_thinking.llm   # smoke test: tokenize, streamed logprobs,
                                  # logprob_token_ids, a batched list of prompts
python cli.py "A meeting starts at 11:50 and runs for 2 hours 45 minutes. When does it end? Show your working."
python server.py             # local browser UI, http://127.0.0.1:8123
python -m experiments.zebra.run ours 5 1 low   # the zebra puzzle above; can take an hour
```

`.env.example` carries the measured values, which differ from some code
defaults: `GATE_THRESHOLD=0.80` against the uncalibrated 0.55 in
`adaptive_thinking/config.py`, and `PORT=8123` against 8099. Copy it to get the
documented behaviour.
[docs/CONFIG.md](docs/CONFIG.md) lists every knob. Any knob can also be set for
a single run from the environment (for example
`OPEN_THINK=false python cli.py "..."`). `python cli.py --json "..."` prints the
raw event stream. `python -m adaptive_thinking.llm` prints your endpoint URL,
so redact it before sharing the output.

Prompts must be hard enough to fire the gate. Time arithmetic with a carry
fires, as in the example above. A single division such as 246 ÷ 3 never does.
The opening thought runs on every turn regardless.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
pytest              # the offline suite
pytest -m slow      # only the CPU-heavy tests, e.g. the 5x5 zebra regeneration (minutes)
```

The offline suite in `tests/` drives the engine against a fake endpoint and
needs no server, no key and no network. Golden recordings in `tests/golden/`
pin every request and event byte for byte, so a restructure that changes
behaviour fails. The live measurement scripts in `experiments/`
(`tree_probe`, `context_probe`, `gate_audit --run`, `bench_hard`,
`zebra.run`) call a real endpoint and are never collected by pytest. Run them
by hand, from the repository root and one at a time:
`python -m experiments.tree_probe`.
[CONTRIBUTING.md](CONTRIBUTING.md) has the details, including the
[structure and rules](CONTRIBUTING.md#structure-and-rules) the code is kept
to.

## Cost

Wall-clock time is the cost, not GPU work. Each request is a small batch, a
handful of beams or a set of `max_tokens=1` reads, over a long prefix the
server has already cached. What adds up is the chain of dependent round trips:

- **per answer window:** one streaming request; when the window has candidate
  positions, one gate request, plus one more while `GATE_LOG_YESNO` is on (the
  default);
- **per block:** one or two fan-out requests, then two per checkpoint (advance
  the beams, judge the contenders);
- **per turn:** the opening thought, which runs even for trivial questions.

On the endpoint used for these measurements, a request cost about 2.7 s
whether it returned one token or fifty, which is why the tree checkpoints
every 500 tokens rather than every token. Recorded durations: 374 s for one
opening thought; 319 s for a logic puzzle answer at `GATE_THRESHOLD=0.80`,
against more than 19 minutes and still running at 0.55; 1,620 s and 3,703 s
for the zebra passes. Running the tree inside the inference engine rather
than over HTTP should remove most of this overhead. That has not been built.

## Repository layout

The engine is the `adaptive_thinking` package. Its modules are listed from the
bottom layer up, and each imports only from its own row or the rows above it
([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#module-map-and-layering) has the
full map and the rule).

| path | what it is |
|---|---|
| `adaptive_thinking/config.py` | Every knob, read from the environment and the `.env` at the repository root. The comments cite the measurement behind each default. |
| `adaptive_thinking/paths.py`, `events_log.py` | Where things live on disk (the repository root, the UI, `events.jsonl`), and appending to and reading `events.jsonl`. |
| `adaptive_thinking/chatml.py`, `prompts.py` | The hand-built ChatML template, and all prompt wording: the gate and judge axes, their levels, the probes. Measured wording, byte-sensitive. |
| `adaptive_thinking/llm.py` | The vLLM client: `stream` (per-token top-k logprobs), `top_next` (a list of prompts in one streamed request), `answer_probs` (`logprob_token_ids` read at one position), plus `labels`, `tokenize`, `prefix_cache_stats`, retries and usage metering. `python -m adaptive_thinking.llm` is the smoke test. |
| `adaptive_thinking/text.py`, `state.py`, `events.py` | The string helpers (`splice`, `_inline_note`, `_restarted`), the beams, blocks, notes and conversation, and the event builders with `EVENT_TYPES`. |
| `adaptive_thinking/scoring.py` | The levelled reads behind the gate and the judge, the two judge scores, and `_in_contention`. |
| `adaptive_thinking/gate.py`, `tree.py` | The gate (`choose`, `gate_scores`), and the thinking tree (`run_block` and its steps: `fan_out`, `prune`, `churn`, the winner). |
| `adaptive_thinking/engine.py` | The `Engine`: one turn (`respond` and its phases) and eviction (`maybe_evict`). It keeps each step of the gate and the tree as a method (`_choose`, `_run_block`, ...). |
| `adaptive_thinking/cli.py`, `server.py`, `static/index.html` | The front ends: a terminal client (`--json` prints raw events), and a local demo, a FastAPI SSE server with a single-file browser UI. They add no logic of their own. The server appends to `events.jsonl` at the repository root. |
| `cli.py`, `server.py` | Launchers for the two front ends, so `python cli.py` and `python server.py` work from the repository root. |
| `experiments/` | Measurement scripts, run as modules from the repository root (`python -m experiments.gate_audit`). `tree_probe`, `context_probe`, `gate_audit` and `bench_hard` are live measurements, not unit tests; `gate_audit` without `--run` only reads `events.jsonl`. `degraded_check` is an offline scenario: a stub endpoint fails after the opening thought, and the turn must keep its thinking. Shared helpers are in `experiments/_common.py`. The package never imports anything from here. |
| `experiments/zebra/` | The zebra experiment. See [experiments/zebra/README.md](experiments/zebra/README.md). |
| `experiments/archive/` | Four measurement scripts for an earlier engine, kept as provenance. See [experiments/archive/README.md](experiments/archive/README.md). |
| `tests/`, `conftest.py` | The offline pytest suite: a fake endpoint (`tests/fake_endpoint.py`), the golden recordings and their harness (`tests/golden/`), and at the root the configuration that keeps the scripts out of collection. `.github/workflows/tests.yml` runs it in CI, after `ruff check .`. |
| `pyproject.toml` | Project metadata, the package declaration (`pip install -e .`), and the ruff and pytest settings. |
| `docs/` | [ARCHITECTURE.md](docs/ARCHITECTURE.md), [CONFIG.md](docs/CONFIG.md), [FINDINGS.md](docs/FINDINGS.md). |
| `.env.example` | Endpoint settings and the measured knob values. |
| `requirements.txt`, `requirements-dev.txt` | Runtime and test dependencies. |

## Limitations and known issues

None of these is fixed yet.

- **Diversity decays across a block.** On the v2 engine, late-block beams
  were near paraphrases. The current engine's first-token fan-out and
  one-survivor-per-parent pruning are meant to counter this, but frontier
  diversity has not been re-measured on it.
- **Resumption can still restart a clause.** The `</think>` anchor fixed
  whole-answer restarts in the recorded runs. Clause-level duplication across
  the seam was seen before the anchor existed and has not been re-measured
  since. `_restarted` is the backstop: it drops the note and resumes without
  it.
- **The gate can cluster.** On the v2 engine it once stopped three times in
  one clause to think about the word "the". `MIN_GAP_TOKENS` spaces firings
  out but does not stop the gate circling the same spot, and the current gate
  has not been shown to avoid it.
- **The gate axes are saturated.** Stakes and commitment score high almost
  everywhere, so only difficulty varies and the threshold mostly controls
  cost.
- **The churn is unvalidated.** It was not observed to fire in any recorded
  run, the zebra runs included. `CHURN_MARGIN=0.12` is provisional: it was
  read off the standings of a single run.
- **`PROBE_ENABLED=false` breaks the browser UI.** In that mode the engine puts
  `NaN` into the event JSON, which the browser cannot parse.
- **`SIBLING_PENALTY` never applies.** One beam's alternatives are always
  distinct tokens, so the penalty never triggers.
- **`CHOICE_EVERY` is not read by the engine.** As a result, the three
  Choice configurations that `experiments/tree_probe.py` compares are the
  same run.
- **`experiments/bench_hard.py` is not a fair comparison.** It caps the
  native arm at `NATIVE_MAX_TOKENS` (1600) and leaves the tree uncapped, and
  its grader matches substrings. Do not quote its results.
  `experiments/zebra/run.py` caps neither arm below the model's context
  length.
- **An endpoint failure during the opening thought ends the turn with an
  error** once the client's retries are exhausted. Failures later in the turn
  degrade gracefully: the thinking and the partial answer are kept and marked
  incomplete.
- **The demo server is local and single-user.** It binds 127.0.0.1 and has no
  authentication. The UI's inline toggle changes a process-wide setting for
  every session, and every query is logged to `events.jsonl`. The UI also
  inserts some model text via `innerHTML`. That is fine for a local tool, but
  do not expose it. See [SECURITY.md](SECURITY.md).
- **Code defaults differ from `.env.example`** for `GATE_THRESHOLD` (0.55 in
  code against the measured 0.80) and `PORT` (8099 against 8123).

## Contributing and licence

Contributions: [CONTRIBUTING.md](CONTRIBUTING.md). Security:
[SECURITY.md](SECURITY.md).

Licensed under the Apache License 2.0; see [LICENSE](LICENSE).
