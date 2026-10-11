# Configuration

Every tunable in the engine is an environment variable read by
`adaptive_thinking/config.py`.
No other module reads `os.environ`. This page lists all 43 of them, with the
default in the code, the value the shipped `.env.example` sets, what each one
does, and what its value rests on.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together and the
[README](../README.md) for setup.

## How settings are read

`adaptive_thinking/config.py` calls `load_dotenv()` on the `.env` file at the
repository root (one level above the package), once, at import. It does not
depend on the working directory. The precedence is:

1. **The process environment.** `load_dotenv` runs with `override=False`, so
   a variable already set in the shell wins over `.env`.
2. **`.env`.** Copy it from `.env.example`.
3. **The code default** in `adaptive_thinking/config.py`.

Some details matter in practice:

- **An empty value falls back to the code default, not to `.env`.** The
  helpers use `os.getenv(name, "") or default`. If the shell holds
  `GATE_THRESHOLD=` (set but empty), `load_dotenv` leaves it alone because it
  is already set, and the engine gets `0.55`, not the `0.80` in `.env`. An
  empty line in `.env` (`GATE_THRESHOLD=`) has the same effect.
- **Booleans:** after trimming whitespace and ignoring case, only `1`, `true`
  and `yes` count as true. Anything else that is not empty is false. That
  includes `on` and `y`.
- **Integers** go through `int()`. A value such as `BEAM_WIDTH=4.0` raises
  `ValueError` when `config` is imported, and every module fails with it.
- **Values are read from the environment once, at import, and looked up at
  call time.** Every module reads `config.X` as a module attribute and never
  uses `from .config import X`, so an assignment to `config.X` at runtime takes
  effect at once. Three things rely on this: the server's `?inline=`
  parameter, `experiments/tree_probe.py`, and tests that use `monkeypatch`.
  Editing `.env` needs a restart.
- **Seeds that contain a newline.** python-dotenv reads an unquoted `\n` as a
  literal backslash and `n`, and a bare `<think>` loses the newline. Only a
  double-quoted value (`"<think>\n"`) gives a real newline. `OPEN_SEED` is
  therefore left out of `.env.example` and kept at its code default.

Per-run overrides, without editing any file:

```bash
# POSIX shells: applies to this one process only
OPEN_THINK=false BEAM_WIDTH=2 python cli.py "A meeting starts at 11:50 and runs for 2 hours 45 minutes. When does it end?"
```

```powershell
# PowerShell: stays set for the session, so remove it afterwards
$env:BEAM_WIDTH = '2'; python cli.py "..."; Remove-Item Env:BEAM_WIDTH
```

To check what a process actually runs with, look at the header line `cli.py`
prints (`entropy floor … window … gate>=… beam WxK @ chunk …`) or at
`GET /config` on the demo server, which reports a subset of the knobs.

Adding, changing or removing a knob touches several files at once;
[CONTRIBUTING.md](../CONTRIBUTING.md#structure-and-rules) lists them.

## Two defaults differ from `.env.example`

| knob | code default | `.env.example` | which to trust |
|---|---|---|---|
| `GATE_THRESHOLD` | `0.55` | `0.80` | **0.80.** `config.py` calls 0.55 the original, uncalibrated value. Over 20 probed positions, 0.55 fired on all 20. 0.80 fired on 4 and was set from that distribution (see the comment in `.env.example`). A run without a `.env` gets 0.55, and the gate then fires in almost every window that has a candidate. |
| `PORT` | `8099` | `8123` | Either works. The shipped `.env.example` gives 8123; with no `.env` the server listens on 8099. |

## Knobs that do little or nothing at present

These are all still read from the environment. They are listed so that
changing one is not mistaken for an experiment.

- **`CHOICE_EVERY`**: the engine does not read it. `_run_block` always ranks
  candidates on length-normalised logprob (`think_step.ranked_by` is always
  `"logprob"`). The value is still shown by `GET /config` and set by
  `experiments/tree_probe.py`, so that script's "Choice every step / every
  4 / never" comparison now runs the same configuration three times.
- **`CHOICE_SHORTLIST`**: the engine does not read it.
  `experiments/tree_probe.py` prints it.
- **`CHOICE_DEBIAS`**: read only by `Engine._rank`, the labelled Choice.
  Only `experiments/tree_probe.py` calls `_rank`.
- **`PROBE_THRESHOLD`**: the engine does not read it. The only comparison
  against it is in `experiments/gate_audit.py`, which checks the logged
  Yes/No probe (`p_yes`) against the gate's decision.
- **`SIBLING_PENALTY`**: read, but as the code stands it never applies.
  `seen_tokens` is reset for each parent beam, and one beam's alternatives
  come from the keys of a dict, so they are always distinct.
- **`NATIVE_MAX_TOKENS`**: used only by the native arm of
  `experiments/bench_hard.py`. The engine arm there has no cap, so that
  script does **not** give the two arms the same limit.
  `experiments/zebra/run.py` does not use this knob: it gives the native arm
  the server's `max_model_len`.

Two others change more than their names suggest:

- **`PROBE_ENABLED=false`** is the entropy-only ablation. At every window that
  has a candidate, the highest-entropy candidate opens a block and no gate
  request is made. That block's `crit` is `float("nan")`, which `json.dumps`
  writes as a bare `NaN`. The browser's `JSON.parse` rejects that, so the demo
  UI breaks in this mode. The terminal view of `cli.py` still works, but its
  `--json` output contains the same bare `NaN`.
- **`GATE_LOG_YESNO=true`** (the default) adds **one batched request per
  window** in which the gate scored anything. It re-asks the old binary
  "worth more thought?" probe, only to record `p_yes` next to the axis score.

`adaptive_thinking/prompts.py` also defines a `LEAF_PROBE` prompt that the
engine no longer uses. It is a constant, not an environment variable.

## Removed knobs

`THINK_ALOUD` has been removed: nothing read it, since the model always writes
the deliberation now. A `.env` that still sets it is harmless; the variable is
simply ignored. A knob that nothing reads is deleted rather than kept, so
this list is where a removed one is recorded.

## Every knob

Legend for the **basis** column:

- **measured**: the value was set from a recorded distribution.
- **measured failure**: the mechanism exists because of an observed failure,
  but the number itself was not tuned.
- **historical**: measured on an earlier engine (per-token tree, binary probe).
  The scripts are kept in `experiments/archive/` for reference and do not run
  against the current engine.
- **uncalibrated** / **provisional**: `config.py` itself marks the value as
  not yet calibrated, or as set from too little data to trust.
- **unmeasured**: a working value with no measurement recorded.

"not set" means the variable is absent from `.env.example`, or present only as
a commented-out line showing the code default.

### Endpoint

| knob | code default | `.env.example` | what it does | basis |
|---|---|---|---|---|
| `OPENAI_BASE_URL` | `""` | `https://your-vllm-host/v1` | Base URL, including `/v1`. `ROOT_URL` for `/tokenize` and `/metrics` is this URL with a trailing `/v1` removed. `VLLM()` raises if it is empty. | – |
| `OPENAI_MODEL` | `""` | `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | Model id sent with every request. `VLLM()` raises if it is empty. The chat format in `adaptive_thinking/chatml.py` assumes Qwen3 ChatML with `<think>`. | – |
| `OPENAI_API_KEY` | `""` | empty | Sent as `Authorization: Bearer …`. | – |
| `LLM_USER_AGENT` | `adaptive-thinking/0.1` | `adaptive-thinking/0.1` | `User-Agent` header. `experiments/zebra/run.py` also sends it. | – |
| `LLM_TIMEOUT` | `300` | `300` | httpx timeout in seconds. The connect timeout is fixed at 30 s. | unmeasured |
| `LLM_RETRIES` | `2` | not set | Retries after a transient failure: HTTP 429/500/502/503/504/524, a connection error, a connect/read/write/pool timeout, or a remote protocol error. A stream is retried only before it has produced a token. | measured failure (a 524 discarded a 374 s opening thought); value unmeasured |
| `LLM_RETRY_BACKOFF` | `3.0` | not set | Seconds to wait before retry *n*: `backoff × n`. | unmeasured |

### The gate: where to stop and think

| knob | code default | `.env.example` | what it does | basis |
|---|---|---|---|---|
| `ENTROPY_FLOOR` | `0.4` | `0.4` | Minimum truncated entropy, in nats, for an answer position to be a gate candidate. The position also needs at least two distinct alternatives. It is a floor, not a gate: below it, every branch would be the same string. | floor rationale historical (P(yes) 0.407 at high entropy vs 0.484 at mid entropy, i.e. independent); value unmeasured |
| `MIN_GAP_TOKENS` | `12` | `12` | Answer tokens that must pass after a block before another can open. | unmeasured |
| `WARMUP_TOKENS` | `28` | not set | No gate block may open before this token index of the answer. | measured failure (every observed restart came from a block in the opening sentence); value unmeasured |
| `WINDOW_TOKENS` | `24` | `24` | Answer tokens per stream request. The gate runs once per window. | unmeasured |
| `PROBE_CANDIDATES` | `6` | `6` | Most candidate positions scored per window, highest entropy first. | unmeasured |
| `GATE_THRESHOLD` | `0.55` | `0.80` | A block opens at the best candidate if its `crit`, the weighted mean of the three gate axes rounded to 4 places, is at or above this. | code default: uncalibrated; `.env.example`: measured (20 positions; the distribution sat in 0.746–0.824) |
| `GATE_LOG_YESNO` | `true` | not set | Also runs the old Yes/No probe on the same positions and adds `p_yes` to `window.probed`. One extra request per scored window. | calibration aid |
| `PROBE_THRESHOLD` | `0.60` | `0.60` | Not read by the engine. `experiments/gate_audit.py` compares `p_yes` against it. | historical (between the essay median of 0.202 and the arithmetic median of 0.622) |
| `PROBE_ENABLED` | `true` | `true` | `false` is the entropy-only ablation described above. | ablation switch |

### The thinking tree

| knob | code default | `.env.example` | what it does | basis |
|---|---|---|---|---|
| `BEAM_WIDTH` | `4` | `4` | Most live conversations in a block, and how many openings the fan-out tries to start. The width never grows. | unmeasured |
| `THINK_BRANCH_K` | `5` | `5` | Alternatives each beam proposes for its last token at a checkpoint (`logprobs` on `top_next`). The fan-out reads `min(20, BEAM_WIDTH × THINK_BRANCH_K)`. Adds no requests. | unmeasured |
| `THINK_CHUNK` | `500` | `500` | `max_tokens` per checkpoint request: how often the tree branches and is judged. **Not a cap on thinking.** A block ends when its beams close it. | measured motivation (a request cost about 2.7 s whatever it generated); value unmeasured |
| `MAX_CHILDREN_PER_PARENT` | `1` | not set | Most survivors one beam's chunk may contribute to the next frontier. `_prune` backfills past it only when the frontier cannot otherwise be filled. | measured failure (without a cap, the frontier collapsed to one parent in 6 of 124 steps, historical) |
| `COPY_PENALTY` | `0.60` | not set | Subtracted from `rank_score` when a candidate's last `COPY_NGRAM` characters already appear in the context. | measured failure (a winner scored −0.01 over 28 tokens of copied text); value unmeasured |
| `COPY_NGRAM` | `24` | not set | Length of that tail, in characters. | unmeasured |
| `SIBLING_PENALTY` | `0.35` | not set | Meant for a token an earlier sibling already took. Currently never applies (see above). | unmeasured |
| `CHURN_MARGIN` | `0.12` | `0.12` | `_in_contention`: a judged live beam is still a challenger while `leader − raw < CHURN_MARGIN`. It decides retirement, the stop rule and the ramble rule. | provisional (one run's standings spread 0.02–0.06 in a normal field and 0.18 with one clearly bad thought; the churn was not observed to fire in any recorded run) |
| `CHURN_PATIENCE` | `2` | `2` | Checkpoints without a new best (`dry`) before a retirement closes the slot instead of refilling it with a variation of a good branch. | unmeasured |
| `AXIS_CLIP` | `2.0` | not set | The judge standardises each axis within a group of candidates and clips it to ±this before the weighted sum that makes `overall`. | unmeasured |
| `STOP_AFTER_FINISHED` | `3` | not set | The block stops early once at least this many thoughts have finished, the best finished thought leads the next by `STOP_MARGIN`, and no live beam is in contention. | reasoned (for interchangeable draws a new record has chance 1/(n+1)); value unmeasured |
| `STOP_MARGIN` | `0.35` | not set | That lead, on the standardised `overall` score. | unmeasured |
| `RAMBLE_RATIO` | `2.5` | not set | A live beam longer than this multiple of the longest finished thought is dropped, but only if it is also out of contention. | unmeasured |
| `CHOICE_EVERY` | `0` | `0` | Not read by the engine (see above). | historical (a labelled single-token Choice, inspired by jevfire, cost 72 requests against 40, gave a wrong answer, and flipped its winner on 4 of 8 reversed-order probes) |
| `CHOICE_SHORTLIST` | `8` | `8` | Not read by the engine. | – |
| `CHOICE_DEBIAS` | `true` | `true` | `_rank` asks the Choice in both option orders and averages. Only `experiments/tree_probe.py` uses it. | historical (the same 4-of-8 flips) |

### The answer, and the note a block leaves in it

| knob | code default | `.env.example` | what it does | basis |
|---|---|---|---|---|
| `BRANCH_K` | `5` | `5` | Top-k logprobs read at each **answer** position. They feed the entropy floor and the `token.top` list in the UI. Not the beam width. | unmeasured |
| `KEEP_BLOCKS_INLINE` | `true` | `true` | Keep each block's note in the context the model reads. Off is the control: the context after a block is byte-identical to the context before it. The server's `?inline=0/1` changes this **for the whole process**. | historical (with it off, blocks changed the next token 0 times across 14 tasks) |
| `INLINE_SEED` | `<think>Hmm, hold on.` | `<think>Hmm, hold on.` | Opens every mid-answer monologue, and every note. | measured (a bare `<think>` made the model restart the answer; two words of seed made it reflect) |
| `ANCHOR_CHARS` | `90` | not set | How many characters of the answer so far a note replays after its `</think>`, cut at a word boundary, so the model finds the sentence already in flight. | measured failure (four restarts of one answer in a turn; 218 kept tokens for about 80 tokens of content before, 101 after) |
| `RESUME_WATCH_TOKENS` | `12` | `12` | Answer tokens after a block whose logprobs make up `resume_logprob`. | unmeasured |

### The opening thought

| knob | code default | `.env.example` | what it does | basis |
|---|---|---|---|---|
| `OPEN_THINK` | `true` | `true` | Run the tree before the first answer token, in Qwen's own `<think>` slot, on every turn. Off means a cold start with the empty `NO_THINK` block. This is the largest cost lever. | design choice |
| `OPEN_SEED` | `"<think>\n"` | not set | Seed for the opening thought: the template's own opener. Leave it at the default (see the note on newlines above). | – |

### Baselines

| knob | code default | `.env.example` | what it does | basis |
|---|---|---|---|---|
| `NATIVE_MAX_TOKENS` | `1600` | not set | Token cap on the native arm of `experiments/bench_hard.py` only. | – |

### Context management

| knob | code default | `.env.example` | what it does | basis |
|---|---|---|---|---|
| `EVICT_AGE_TOKENS` | `100000` | `100000` | Eviction triggers once any note is this many generated tokens old. | unmeasured |
| `EVICT_KEEP_TOKENS` | `20000` | `20000` | When it triggers, every note older than this is removed in one pass. The gap between the two watermarks keeps prefix rewrites rare. | unmeasured |

### Server

| knob | code default | `.env.example` | what it does | basis |
|---|---|---|---|---|
| `PORT` | `8099` | `8123` | Port the demo server binds on `127.0.0.1`. | – |

## Fixed in code

These shape behaviour but are not configurable. They are listed so nobody
hunts for a variable that does not exist.

- `MAX_LOGPROBS = 20` in `adaptive_thinking/tree.py`: vLLM's default
  `--max-logprobs`. It caps how many openings a fan-out can see.
  `Engine.MAX_LOGPROBS` is the same constant, exposed on the engine; the
  fan-out reads it from `tree.py`.
- The fan-out reads the opening distribution at most **twice**, and does the
  second read only if the first gave fewer than `BEAM_WIDTH` usable openings.
- A new best must beat the previous leader by more than **0.01** raw to reset
  `dry`.
- The winner's substance floor: a finished thought counts only if its text is
  at least **1/10** as long as the longest thought in the block.
- `_already_written` compares a **48**-character run-up. `_restarted` uses a
  **40**-character window (both in `adaptive_thinking/text.py`), watched for
  **400** characters after a block (`Engine._stream_window`).
- A block's `tail` is the last **70** characters of the answer. A note's
  anchor is chosen from the last **240** (both in `Engine._think_at`).
- A thought's length in tokens is an estimate: each chunk adds a quarter of
  its characters (at least one), and a branch's alternative token one more
  (`tree._expand`). It feeds `rank_score` and the ramble rule.
- `RETRY_STATUS = {429, 500, 502, 503, 504, 524}` in `adaptive_thinking/llm.py`.
- `NUDGE_PREFIX` and every prompt's wording (the gate axes, judge axes and
  their levels) are in `adaptive_thinking/prompts.py`, and the chat-format
  constants in `adaptive_thinking/chatml.py`. They are
  byte-sensitive: see [ARCHITECTURE.md](ARCHITECTURE.md#design-invariants).

## What costs requests

Nothing caps the answer, the thinking or the number of blocks. That is by
design, so cost follows from the knobs above. For one turn:

```
requests ~ answer windows
         + windows with a gate candidate x (1 + GATE_LOG_YESNO)
         + for each block: fan-out (1-2) + 2 x checkpoints
```

- Each checkpoint is one `top_next` request, which advances all live beams by
  up to `THINK_CHUNK` tokens. It is followed by one `answer_probs` request that
  judges every contender on the four axes.
- `OPEN_THINK` adds a full block to every turn.
- `GATE_THRESHOLD` decides how many mid-answer blocks open.
- `ENTROPY_FLOOR`, `WARMUP_TOKENS`, `MIN_GAP_TOKENS` and `PROBE_CANDIDATES`
  decide how many positions are scored at all.
- `THINK_CHUNK` is a latency trade-off, not a limit. A smaller value means
  more round trips for the same thought.
- `stats.requests` does not count `/tokenize` or `/metrics` calls:
  - 26 calls resolve the single-token letter labels, once per client.
  - 2–4 calls resolve the Yes/No ids when `GATE_LOG_YESNO` is on, also once
    per client.
  - Up to 2 `/tokenize` calls and 1 `/metrics` call happen per eviction.
