# Architecture

This page describes how the engine is put together: the vocabulary, the
modules and the layering rule they follow, the flow of one turn, the events it
emits, and the design rules the code depends on.
[README](../README.md) covers what the project is and how to run it.
[CONFIG.md](CONFIG.md) lists every knob.
[CONTRIBUTING.md](../CONTRIBUTING.md#structure-and-rules) has the rules for
changing any of it.

## The idea in one paragraph

A reasoning model normally thinks once, up front, and then answers. This
engine can also stop **while writing the answer**, at a position a gate judges
worth thinking about. There it runs a small tree search over the model's own
thinking: several conversations advance in parallel, weak ones are retired,
and the best finished thought is kept. It then resumes the answer with that
thought left in the context, at the place it happened. Every turn also opens
with the same search in the model's native `<think>` slot. A response
therefore reads like this:

```
[opening thought] answer tokens ... [thinking block] ... answer tokens ... [thinking block] ... done
```

Everything runs against an OpenAI-compatible vLLM server (the requirements are
listed below), using a handful of request shapes. There are no changes to the
inference server, no fine-tuning and no custom kernels.

## Glossary

The code grew through several engines, and some names outlived the design
they were coined for. This is what each term means in the current code.

**The search**

| term | meaning | in the code |
|---|---|---|
| block | One run of the thinking tree, from fan-out to winner. Either the **opening thought**, run in the model's own `<think>` slot before the answer, or a **gate block**, opened mid-answer where the gate fired. | `Block`, `kind` `"open"` or `"gate"` |
| beam | One conversation inside a block's tree: its generated text so far, its logprob and its judge scores. | `Beam` |
| conversation | In prose and in the UI, the same thing as a beam. Not to be confused with `Conversation`, the whole chat transcript the engine renders into a prompt. | `Beam`; `Conversation` is the transcript |
| lineage | A conversation followed across checkpoints. It keeps its `uid` while it is carried forward; a second survivor from one lineage, or a churn refill, is a new lineage whose `from_uid` names the one it split from. The UI draws one card per lineage. | `Beam.uid`, `Beam.from_uid` |
| slot | A position on the frontier at one checkpoint. Slots are reused every checkpoint, so a slot is not an identity; `parent` names the slot a candidate came from. | `Beam.index`, `Beam.parent` |
| frontier | The live beams, at most `BEAM_WIDTH`. | `beams` in `tree.run_block` |
| thought | A beam's text: the generated body only, never the seed. A **finished** thought is one that ended: its chunk reached `</think>` (or another marker) or a stop, it began writing the nudge line, or its chunk came back with no alternatives. It leaves the frontier for the **pool**, keeps being judged and can still win. | `Beam.text`, `Beam.done`, `finished` |
| candidate | Two different things. In the gate: an answer position that might open a block (`gate._Cand`). In the tree: a beam proposed at a checkpoint, one per alternative for a live beam's last token, before pruning (`cands` in `tree.run_block`). | |
| fan-out | The first step of every block: one one-token read of the distribution over the thought's first token, whose best usable tokens start the conversations. | `tree.fan_out` |
| checkpoint | One iteration of a block's loop: advance every live beam by up to `THINK_CHUNK` tokens in one request, expand into candidates, prune, churn, judge in one request. | see *checkpoint and step* below |
| churn | Retiring a live conversation that is out of contention, and either refilling its slot with a variation of the best lineage or closing the slot. | `tree.churn`, `tree._apply_churn` |
| in contention | A live beam whose `raw` is within `CHURN_MARGIN` of `leader`; one not yet judged counts as in contention. The single test for whether a live conversation may be removed. | `scoring._in_contention` |
| leader, dry | The best `raw` seen anywhere in the block so far, and the number of checkpoints since it last improved by more than 0.01. | `tree._update_leader` |
| winner | The thought that becomes the block's monologue. | `tree._pick_winner` |

**What a block leaves behind**

| term | meaning | in the code |
|---|---|---|
| monologue | The winning thought's text, as the block produced it. | `Block.monologue`, the `monologue` event |
| note | What stays in the model's context after a block, at the place it happened: `off` (a character offset into the turn's visible answer, or `HEAD_OFF` for the opening thought), `text` and `born`. The opening thought's note is its monologue cut to whole sentences and rendered in the assistant header by `think_head`. A gate block's note is the inline note below. | `Note` |
| inline note | `INLINE_SEED`, the monologue on one line, the nudge line, `</think>`, then the anchor. Built by `text._inline_note`. | |
| seed | What opens a thought: `OPEN_SEED` (`<think>` and a newline) for the opening thought, `INLINE_SEED` (`<think>Hmm, hold on.`) mid-answer. Prepended when a prompt is built, never stored in a beam. | `config.OPEN_SEED`, `config.INLINE_SEED` |
| nudge | The fixed closing line of an inline note, starting with `NUDGE_PREFIX`. Also the marker used to cut a copied nudge out of a monologue. | `prompts.NUDGE_PREFIX` |
| anchor | Up to `ANCHOR_CHARS` of the answer's last words, replayed after `</think>` so the model resumes mid-sentence. | `text._tail_words` |
| visible text | The answer the reader sees, without notes. `splice` puts the notes back in to build what the model sees. | `vis`, `Turn.text`, `text.splice` |
| clock, age | `Conversation.clock` counts tokens generated in the conversation. A note's age is `clock − born`; eviction runs on age. | |

**The answer stream and the gate**

| term | meaning | in the code |
|---|---|---|
| window | `WINDOW_TOKENS` answer tokens streamed in one request. The gate runs once per window. | `Engine._stream_window` |
| gate | Chooses whether, and where in the last window, to open a block. | `gate.choose`, `gate.gate_scores` |
| spill | A transcript marker (`SPILL`) cut out of model output. One at the answer's final token is the model ending its turn. | `stats.spill_markers` |
| rewind | Dropping the answer tokens written after the chosen position, before a gate block runs. | the `rewind` event |
| restart | A note made the model begin the answer again; the engine rolls back and drops the note. | `text._restarted`, `Engine._undo_restart` |
| degraded | The endpoint failed mid-answer after its retries; the turn finishes with what it has. | the `degraded` event |

**Checkpoint and step.** They are the same thing. The per-token engine this
grew out of advanced one token per *step*, and the names stayed when the step
became a 500-token checkpoint: the `think_step` event and its `step` field,
`Block.steps`, `block_close.steps`, and `think_steps` in `done.stats` all count
checkpoints. Some labels in the browser UI predate the change as well:
"candidates per token" means candidates per checkpoint, "thought for N tokens"
shows N checkpoints, and "tokens per beam" is `THINK_CHUNK`, the tokens per
beam per checkpoint. The names are part of the event schema and the UI, so
they have been kept.

**The scores.** "Score" means several different numbers:

| number | what it is | where it appears |
|---|---|---|
| `Beam.score` | cumulative logprob of the beam's tokens, before penalties | `total` in the beam objects of `frontier` and `think_step` |
| `rank_score` | `score / max(1, tokens) − penalty`: length-normalised, with the copy (and sibling) penalties. What pruning sorts on. | `score` in the beam objects of `frontier` and `think_step` |
| `raw` | the judge's weighted mean of the four raw axes, absolute, 0..1. The only score comparable across live beams and finished thoughts. | `standings.ranked[].raw`; `winner_score` is the winner's `raw` |
| `overall` | the judge's axes standardised within a group, clipped, weighted and summed. Relative, centred on 0, meaningless across groups. | `standings.ranked[].overall`, `think_step.retired[].overall`; the figure on a UI card |
| `crit` | the gate's criticality: the weighted mean of the three gate axes, absolute, 0..1, rounded to 4 places | `window.probed[].crit`, `block_open.crit` |
| `p` | the labelled Choice's probability, only when `rank` ran | `p` in the beam objects |

[Two scores](#two-scores) explains why `raw` and `overall` both exist.

## Why raw `/v1/completions`, not the chat API

- **Batching.** Only `/v1/completions` accepts a **list** of prompts in one
  request. The design depends on this: every live beam of a checkpoint, and
  every candidate × axis pair a judge scores, goes out in one round trip.
- **Control of the template.** The engine has to leave the assistant's
  thinking slot open, put a finished thought into it, and insert thinking
  notes in the middle of an answer. None of that can be expressed through a
  chat template the server applies. `chatml.py` therefore builds Qwen3 ChatML
  (`<|im_start|>`, `<|im_end|>`, `<think>…</think>`) itself.

What the server must provide:

| requirement | used by | if missing |
|---|---|---|
| `POST /v1/completions` accepting a list of prompts, with SSE streaming | everything | nothing works |
| `logprobs` up to 20 (vLLM's default `--max-logprobs`) | answer stream, beam checkpoints, fan-out | HTTP 400 |
| `logprob_token_ids` + `return_tokens_as_token_ids` (a vLLM extension that returns the logprob of specific token ids) | the gate, the judge, the Yes/No probe | `RuntimeError` naming the parameter |
| `POST /tokenize` at the server **root** (not under `/v1`) | answer ids, letter labels, eviction size | traceback on the first gate or judge request |
| `GET /metrics` at the server root | eviction's cache-hit report | the field is `null`; nothing else changes |
| `GET /v1/models` with `max_model_len` | `experiments/zebra/run.py` only | that script fails |

## Module map and layering

The engine is the `adaptive_thinking` package. Its modules form layers, and a
module imports only from the layers below its own:

```
config  <-  chatml / prompts  <-  llm  <-  text / state  <-  scoring  <-  gate / tree  <-  engine  <-  cli / server / UI
```

Module by module, bottom layer first, with what each imports from the
package:

```
layer                   module              imports from the package
knobs                   config              nothing at import; summary() (the /config
                                            dict) reads prompts.GATE_AXES when called
files on disk           paths               nothing: ROOT, STATIC, EVENTS
                        events_log          nothing: appending to and reading events.jsonl
chat format, wording    chatml              nothing
                        prompts             chatml
client                  llm                 config, chatml, prompts   (+ httpx)
text and state          text                config, chatml, prompts
                        state               chatml, prompts, text
                        events              config, prompts
judgement               scoring             config, prompts
gate and tree           gate                config, prompts, scoring, text
                        tree                config, prompts, events, llm, scoring, state, text
one turn                engine              config, chatml, events, gate, llm, scoring, state, text, tree
front ends              cli                 config, prompts, engine, llm, state, text
                        server              config, events_log, paths, engine, llm, state
                                            (+ fastapi, uvicorn); serves static/index.html
                        static/index.html   GET /config, GET /chat (SSE), POST /reset
```

Outside the package:

- `cli.py` and `server.py` at the repository root are launchers for
  `adaptive_thinking.cli` and `adaptive_thinking.server`, so that
  `python cli.py` and `python server.py` work from a checkout.
- `experiments/` holds the measurement scripts. They import the package, and
  `experiments/_common.py` holds what several of them share.
- `tests/` is the offline suite.

### The layering rule

- **Imports point down.** A module may use modules in lower layers, and a
  module in the same layer only where the table shows it (`prompts` uses
  `chatml`, `state` uses `text`). There are no cycles. The one reference upwards is
  `config.summary()`, which imports `prompts` inside the function, so that
  `config.py` itself imports nothing from the package and can be executed on
  its own.
- **The package never imports `experiments/` or `tests/`.** Experiments depend
  on the package, never the reverse. A script that needs something the engine
  does not offer gets a helper in `experiments/_common.py`, not a hook in the
  engine.
- **Knobs and prompt constants are read at call time**, as attributes of
  their module (`config.BEAM_WIDTH`, `chatml.SPILL`, `prompts.AXES`), never as
  a copy taken at import with `from .config import X`. The server's
  `?inline=` switch, `experiments/tree_probe.py` and the tests all assign to
  them at runtime.
- **New code goes in the lowest layer that can hold it.** A string helper
  belongs in `text.py`, a request shape in `llm.py`, a new event's builder in
  `events.py`, a step of the search in `tree.py` (as a pure function of the
  beams where it can be), the gate's rules in `gate.py`, and anything that
  sequences a turn in `engine.py`. The front ends render events; they add no
  logic of their own.

`tests/test_package.py` checks the parts of this that can be checked
mechanically: no module in the package binds a knob or a prompt constant with
`from ... import`, `config.py` imports nothing from the package at import,
and `EVENT_TYPES` lists every event type the engine yields. The direction of
the other imports is kept by review.

### Why the code is held to these rules

- **The prompts are measured artefacts.** The engine decodes at temperature 0
  over a prefix cache, so one changed byte in a prompt, a seed or the order of
  a batch changes every token after it. The wording of each axis and level was
  set against an observed failure. Tidying it is a behaviour change, and has
  to be measured like one.
- **Behaviour is pinned byte for byte.** The goldens in `tests/golden/` record
  every request and every event of scripted turns. A restructure leaves them
  unchanged; a behaviour change regenerates them in a commit that says what
  changed.
- **Events are a contract.** The CLI, the browser UI, `events.jsonl`, the
  measurement scripts and the goldens all read them, so an event changes only
  together with every reader (see [Event vocabulary](#event-vocabulary)).
- **The record includes what did not work.** Several of the most useful
  results here are negative ([FINDINGS.md](FINDINGS.md)). A comment explains
  why the code is the way it is and cites the measurement, so that a default
  is not "tidied up" by someone who has not seen the failure it prevents.

### `config.py`, `chatml.py`, `prompts.py`: knobs, chat format, prompts

Three modules, one job each:

1. **Knobs** (`config.py`). Every environment variable, read through `_i`,
   `_f`, `_b` and `os.getenv`. See [CONFIG.md](CONFIG.md).
2. **The chat format** (`chatml.py`).
   - `IM_START` and `IM_END`.
   - `NO_THINK = "<think>\n\n</think>\n\n"`: what Qwen's template emits when
     thinking is disabled.
   - `SPILL`: markers that must never reach the reader.
   - `STOP`: the stop list sent with generation requests.
   - `turn(role, text)`.
   - `assistant_header(think=False)`. With `think=True` it stops before the
     thinking slot.
   - `think_head(text)`: a finished thought in the header. It is the single
     source for this string, both while generating and when re-rendering.
   - `closed_question(context, body)`: closes the assistant turn with
     `<|im_end|>`, asks `body` as a user turn, and opens a fresh assistant
     header, so the answer is read at a single position.
3. **Prompt wording and builders** (`prompts.py`). Measured wording: see
   [why the code is held to these rules](#why-the-code-is-held-to-these-rules).
   - `DEFAULT_SYSTEM`, and `NUDGE_PREFIX`, the fixed line that closes a
     mid-answer note.
   - The judge axes `AXES`: looping, progress, confusion, concreteness.
   - The gate axes `GATE_AXES`: stakes, difficulty, commitment. Every axis
     has five concrete, situational levels, from worst to best.
   - `axis_prompt`, `gate_prompt`, `choice_prompt` and `probe_prompt`, each
     built on `chatml.closed_question`.
   - `GATE_PROBE` (the Yes/No probe logged beside the gate), `YES` and `NO`,
     `CHOICE_QUESTION`, and `LEAF_PROBE`, a probe the engine no longer
     reads.

### `llm.py`: the request shapes

The `VLLM` class holds one `httpx.AsyncClient`. Its base URL is
`OPENAI_BASE_URL`, and it sends the headers `Authorization: Bearer …`,
`User-Agent` and `Content-Type: application/json`. The timeout is
`LLM_TIMEOUT`, with a 30 s connect timeout.

`transient(exc)` decides what gets retried: HTTP 429/500/502/503/504/524,
connection errors, timeouts and remote protocol errors. `_give_up` is the one
retry rule every request shape shares, and `_backoff` counts the retry and
waits `LLM_RETRY_BACKOFF × n` seconds before retry *n*.

| method | request | payload keys | returns | callers |
|---|---|---|---|---|
| `stream(prompt, max_tokens, top_k=None, temperature=0.0, stop=None)` | `POST /completions`, SSE | `model`, `prompt` (str), `max_tokens`, `temperature`, `logprobs` (= `BRANCH_K`), `stop` (= `STOP` unless overridden), `stream: true` | async generator of `Step` (token text, logprob, sorted top-k, finish). Sets `last_finish`. | the answer windows (`Engine._stream_window`) |
| `top_next(prompts, k=None, max_tokens=None)` | `POST /completions`, SSE | `model`, `prompt` (list), `max_tokens` (= `THINK_CHUNK`), `temperature: 0`, `logprobs` (= `THINK_BRANCH_K`), `stop`, `stream: true` | yields `("tok", i, text)` as tokens arrive, then `("end", [(text, logprob_sum, top_k_at_last_token, stopped)])` with one tuple per prompt | checkpoints (`tree.advance`); the fan-out (`tree.fan_out`) with `max_tokens=1` |
| `answer_probs(prompts, ids=None)` | `POST /completions` | `model`, `prompt` (list), `max_tokens: 1`, `temperature: 0`, `logprobs: 0`, `logprob_token_ids`, `return_tokens_as_token_ids: true` | per prompt, probabilities over `ids` renormalised to sum to 1, read from `top_logprobs[0]["token_id:<id>"]` | `scoring.levelled_read` (the gate and the judge), `tree.rank`, `p_yes` |
| `p_yes(prompts)` | via `answer_probs` with the Yes/No ids | – | `[P(Yes)]` | `gate.choose`, when `GATE_LOG_YESNO` is on |
| `complete(prompts, max_tokens, temperature=0.0, stop=None)` | `POST /completions` | `model`, `prompt` (list), `max_tokens`, `temperature`, `stop` | list of continuation texts | only the `python -m adaptive_thinking.llm` smoke test |
| `labels(n)` | `/tokenize` once per letter A–Z, cached | – | `n` verified single-token labels `[(letter, id)]` | `scoring.levelled_read`, `tree.rank` |
| `tokenize(text)` | `POST {ROOT_URL}/tokenize` | `model`, `prompt`, `add_special_tokens: false` | token ids | `answer_ids`, `labels`, eviction's `_size` |
| `answer_ids()` | tokenizes `Yes`/` Yes` and `No`/` No`, cached | – | `(yes_id, no_id)`, requiring each to be a single token | `answer_probs` when `ids` is omitted |
| `prefix_cache_stats()` | `GET {ROOT_URL}/metrics` | – | `prefix_cache_hits_total / prefix_cache_queries_total`, or `None` | `maybe_evict` |
| `usage_snapshot()` | – | – | `{requests, prompt_tokens, completion_tokens, retries}` | `UsageMeter` |

Two helpers sit beside the class. `UsageMeter(client)` takes the clock and a
usage snapshot when it is made; `secs()` and `used()` give the elapsed time
and the usage deltas since then. `respond` meters a turn with one and
`run_block` a block. `identity_headers()` returns the `Authorization` and
`User-Agent` headers, so that `experiments/zebra/run.py`'s native arm, which
uses its own HTTP client, presents the same identity as the engine.

Notes:

- **Retries.** `_post` (used by `answer_probs` and `complete`) retries a
  transient failure in full. The two streaming calls retry only **before** the
  first token. A token already handed to the caller cannot be taken back.
- **Usage accounting.** `completion_tokens` is counted **locally** for
  streams, because the engine aborts streams on purpose and an aborted stream
  never receives the server's final usage chunk. `prompt_tokens` comes from
  the server, so it covers batched calls only and is not a total.
- **Labels.** Each label must be exactly one token. The labelled question is
  inspired by jevfire. `labels()` verifies the letters A-Z one by one through
  `/tokenize`, so the labels come from the served model's own tokenizer and
  nothing is loaded locally.
- **Entropy.** `Step.entropy` is a lower bound: only the top-k is on the wire,
  and the rest of the mass is treated as a single bucket.

### The engine: `text.py`, `state.py`, `scoring.py`, `gate.py`, `tree.py`, `events.py`, `engine.py`

**Text helpers** (`text.py`), pure functions of their arguments and of the
`chatml`, `prompts` and `config` values they read at call time:

- `cut_at_marker(text)` truncates at the earliest `SPILL` marker. Every model
  output passes through it, because vLLM streams the triggering token before
  it applies `stop`.
- `splice(vis, notes)` inserts the body notes (`off >= 0`) into the visible
  answer at their character offsets. The same function builds the prompt while
  generating and when replaying.
- `head_note(notes)` finds the opening thought (`off == HEAD_OFF == -1`).
- `_at_boundary(vis, off)`: a new sentence or clause starts here.
- `_already_written(vis, off)`: the 48 characters before `off` already appear
  earlier.
- `_restarted(vis, off)`: tests whether the model is writing the answer
  again. It checks **containment**, not prefix, because the model paraphrases
  its openings.
- `_whole_sentences(text)`: cut back to the last complete sentence.
- `collapse_ws(text)`: every run of whitespace as one space.
- `_inline_note(block, tail)` builds a mid-answer note, in this order:
  - `INLINE_SEED`,
  - the winning monologue, on one line and cut to whole sentences,
  - `NUDGE_PREFIX` and a fixed line,
  - `</think>`,
  - up to `ANCHOR_CHARS` of the answer's last words, replayed so the model
    resumes mid-sentence.
- `_tail_words(tail, n)`.

`scoring.py` holds `_in_contention(beam, leader)`, the single "may this
conversation go on?" test, beside the levelled reads (see the judge below).

**Data model** (`state.py`)

| class | holds |
|---|---|
| `Beam` | One conversation in the tree. See the field list below. |
| `Block` | One search: `id`, `kind` (`"open"` or `"gate"`), `tail`, `crit` and `gate` (the gate's score; `None` for the opening thought), `beams`, `monologue` (the winner's text), `winner_score` (the winner's raw), `steps` (checkpoints), `requests`, `changed_next_token`, `finished_count`, `explored_tokens`, `secs`. |
| `Note` | A block as it sits in the context: `off` (character offset into the turn's visible text, or `HEAD_OFF` for the opening thought), `text`, `born` (value of `Conversation.clock` when it was written). |
| `Turn` | `role`, `text` (visible), `blocks`, `notes`. |
| `Conversation` | `system`, `turns`, and `clock` (tokens generated so far). `render(pending_user=None, think=False)` builds the raw prompt, and `render_assistant(turn)` one past assistant turn. `live_notes()` lists every note still in context. |

`Beam` fields:

- `text`: the generated body only. The seed is never stored in it (see the
  invariants below).
- `score`: cumulative logprob.
- `parent`: the slot it came from in the previous step.
- `uid` and `from_uid`: the lineage's identity, and the lineage it split from.
- `index`: its slot in the current step.
- `tokens`: estimated length.
- `penalty`, `last`, `p`, `done`, `medal`.
- The judge's output: `axes`, `raw` and `overall`.
- The derived property `rank_score = score / max(1, tokens) - penalty`.

`render()` writes each past assistant turn as:

```
<|im_start|>assistant\n + (think_head(opening thought) or NO_THINK) + splice(text, notes) + <|im_end|>\n
```

**`Engine`** (`engine.py`)

- `respond(conv, user_text)`: an async generator of events. It runs the
  whole turn in phases that share one `_TurnState`: `_opening_thought`, then
  per window `_stream_window`, the gate (`_choose`), `_think_at` where the
  gate fires and `_undo_restart` where a note made the answer start over;
  last, `_commit_turn` writes the turn into the conversation, `maybe_evict`
  runs, and `events._done_stats` builds the `done` event's stats.
- **The gate** (`gate.py`):
  - `_choose(window, …)` (`gate.choose`) filters and ranks candidate
    positions.
  - `_gate_scores(ctxs)` (`gate.gate_scores`) scores them on the gate axes
    in one request.
- **The tree** (`tree.py`):
  - `_run_block(block, ctx, seed=None)` (`tree.run_block`) runs the whole
    search.
  - `_fan_out(ctx, seed)` (`tree.fan_out`) sets up the first frontier.
  - `_prune(ranked, width)` and `_churn(...)` (`tree.prune`, `tree.churn`)
    are static methods that make no requests. `_churn` does relabel the beam
    it brings in (`uid`, `from_uid`).
- **Judging** (`scoring.py`):
  - `_score_axes(ctx, *groups)` (`scoring.score_axes`) scores contenders on
    the judge axes in one request.
  - `_rank(...)` (`tree.rank`) is the labelled Choice. The engine does not
    call it; only `experiments/tree_probe.py` does.
- **Eviction:** `maybe_evict(conv)` and `_size(conv)`.

Each method in the gate, tree and judging groups is a thin delegate to the
module function named beside it. The ones that make requests pass the
engine's client as the function's first argument; `_prune`, `_churn` and
`_finalise_block` are the module functions themselves, as static methods.
The methods are kept so that every step can still be driven through an
`Engine`, as the tests and `experiments/tree_probe.py` do. The event
builders, and `EVENT_TYPES`, the list of every event type the engine yields,
are in `events.py`.

### `cli.py`

`adaptive_thinking/cli.py`, started by `python cli.py` at the root (or
`python -m adaptive_thinking.cli`). Runs one turn in a terminal and renders
each event with ANSI colour. `--json` prints the raw events, one JSON object
per line. In both modes it first prints a line showing the settings in effect
(drawn from `config.summary()`, plus the churn knobs) and then the question,
each followed by a blank line, so `--json` output starts with four lines that
are not JSON. With no question on the command line it asks a built-in
time-arithmetic one.

### `server.py`

`adaptive_thinking/server.py`, started by `python server.py` at the root (or
`python -m adaptive_thinking.server`; `uvicorn server:app` also works). A
FastAPI app with no logic of its own. It forwards the engine's events over
Server-Sent Events.

| route | what it does |
|---|---|
| `GET /` | serves `adaptive_thinking/static/index.html` |
| `GET /config` | `config.summary()`: a subset of knobs, with these keys in this order: `model`, `window_tokens`, `entropy_floor`, `gate_threshold`, `gate_axes`, `probe_enabled`, `beam_width`, `think_branch_k`, `think_chunk`, `choice_every`, `open_think`, `keep_blocks_inline`, `evict_age_tokens`, `evict_keep_tokens`. The page reads them by name. Makes no LLM call. |
| `POST /reset` | body `{"session": …}`; forgets that conversation |
| `GET /chat?session=…&q=…&inline=…` | runs one turn and streams `data: {event}` lines. If `inline` is given (`0` or `1`), it sets `config.KEEP_BLOCKS_INLINE` **for the whole process**. |

Other behaviour:

- Conversations are kept in memory, keyed by session.
- The server binds `127.0.0.1:PORT`.
- A selection of events is appended to `events.jsonl` at the repository root
  (see below), through `events_log.py`, so a browser session can be audited
  afterwards.

### `static/index.html`

`adaptive_thinking/static/index.html`. Vanilla JavaScript, with no build step:

- It reads `/config`, opens an `EventSource` on `/chat`, and handles each
  event type in one `switch`.
- The answer's tokens are tinted by entropy, and gate decisions appear on
  their tooltips.
- Each block becomes a panel with one card per lineage, showing live text,
  scores, medals, retirements and births.
- One place inserts model text with `innerHTML`: the gate summary line. See
  [SECURITY.md](../SECURITY.md).

### Measurement scripts

They live in `experiments/` and run from the repository root as modules
(`python -m experiments.gate_audit`). All of them talk to a live endpoint
except `degraded_check`, `gate_audit` without `--run` (it only reads
`events.jsonl`) and the zebra generator. None of them is a unit test, and
pytest never collects them. The offline test suite is under `tests/`; see
[CONTRIBUTING.md](../CONTRIBUTING.md).

| script | measures |
|---|---|
| `experiments/gate_audit.py` | Where the gate fires and why, read from `events.jsonl`. `--run` generates a prompt set first. |
| `experiments/bench_hard.py` | Trap questions: the model's own thinking against the engine. Its native arm is capped by `NATIVE_MAX_TOKENS` and the engine arm is not. Writes `bench_hard.jsonl` at the repository root. |
| `experiments/tree_probe.py` | Tree shape, frontier diversity, and the label-position bias of the labelled Choice. Formerly `test_tree.py`. |
| `experiments/context_probe.py` | That notes stay at their offsets, that age-based eviction works, and that a turn renders byte-identically afterwards. Use small watermarks. Formerly `test_context.py`. |
| `experiments/degraded_check.py` | Offline, with a stub endpoint: a 524 after the opening thought keeps the thinking. It prints OK/FAIL lines and exits with status 1 if any check failed; `tests/test_degraded_turn.py` covers the same scenario with assertions. Formerly `test_degraded.py`. |
| `experiments/_common.py` | What the scripts share: UTF-8 console output, running one turn to its `done` event, splitting a native-thinking completion at `</think>`, and the gate's `window` event as rows. |
| `experiments/zebra/` | Puzzle generator (`generate.py`, formerly `zebra.py`), a CSP solver that certifies a unique solution, the committed puzzles, and a runner that grades by exact grid match. Formerly `zebra/` at the repository root. |
| `experiments/archive/` | Scripts from an earlier engine (a per-token tree and a binary probe). Kept for reference; they do not run against the current code. |

## The flow of one turn

```
render prompt ─► [opening thought] ─► loop: stream a window ─► gate ─► (block?) ─► ... ─► finish ─► evict? ─► done
```

The phases are the `Engine` methods named in the steps below; `respond` runs
them in this order.

1. **Prompt.** `prefix = conv.render(pending_user=q, think=OPEN_THINK)`. With
   the opening thought on, the header stops right after
   `<|im_start|>assistant\n`. The engine emits `start`.
2. **Opening thought** (if `OPEN_THINK`; `_opening_thought`).
   - A `Block(kind="open")` runs `_run_block(prefix, seed=OPEN_SEED)`.
   - The winner, cut to whole sentences, becomes a `Note(off=HEAD_OFF)`.
   - `prefix += think_head(text)`: exactly the bytes `render()` will produce
     for this turn later.
   - The engine emits `block_close` with `kind: "open"`.
   - Nothing gates this block, and nothing is rewound.
3. **Answer loop.** It runs until the model ends its turn, with no cap on
   length.
   1. **Stream a window** (`_stream_window`). `stream(prefix + splice(vis,
      notes), max_tokens=WINDOW_TOKENS)`. For each token:
      - Cut any `SPILL` marker and record it in `spills`.
      - Append the token to `vis`.
      - If this is the first token after a gate block, emit `resumed`. It
        records what the model was going to write and what it wrote.
      - Emit `token`.
      - While a block is fresh, watch for `_restarted`.
      - The stream is closed explicitly on an early break.
   2. **Restart backstop** (`_undo_restart`). If the model began the answer
      again after a note, roll back to the block and drop that note. Emit
      `restart` and continue. A window that produced no token at all ends
      the loop.
   3. **Gate: `_choose`.**
      - Candidates are the window positions with entropy ≥ `ENTROPY_FLOOR`,
        at least two distinct alternatives, `idx ≥ WARMUP_TOKENS`, at least
        `MIN_GAP_TOKENS` since the last block, not inside already-written
        text, and at a sentence or clause boundary.
      - Keep the top `PROBE_CANDIDATES` by entropy.
      - `_gate_scores` builds, for each candidate, the full prompt as it read
        at that position. It asks every (position × gate axis) pair in **one**
        `answer_probs` request and turns each answer into an expected level,
        0..1. `crit` is the weighted mean.
      - If `GATE_LOG_YESNO` is on, one more request records `p_yes`.
      - The best candidate opens a block if `crit ≥ GATE_THRESHOLD`.
      - The engine emits `window` whenever at least one position passed the
        filters above (`gate._eligible`), whether or not a block opens.
   4. **No block.** If the model had finished, leave the loop. Otherwise
      stream the next window.
   5. **Rewind** to the chosen position (`_think_at`, which also runs steps 6
      and 7). Emit `rewind` if any tokens were dropped.
   6. **Block.** A `Block(kind="gate")` runs
      `_run_block(prefix + splice(vis, notes))` with `INLINE_SEED`.
   7. **Note.** If `KEEP_BLOCKS_INLINE` is on, `_inline_note` is inserted at
      `off = len(vis)` with `born = clock + idx`. Emit `block_close`.
      Generation resumes at the **same** position, and the model picks its own
      next token.
4. **Endpoint failure.** A transient failure inside the answer loop ends the
   loop and emits `degraded`. The thinking and the partial answer are kept.
   Any other exception propagates. A failure during the opening thought is
   outside this handler and propagates too.
5. **Finish** (`_commit_turn`).
   - A last `cut_at_marker` pass runs over the answer.
   - Note offsets are clamped to the final text.
   - Both turns are appended to the conversation, and `clock += idx`.
6. **Evict** (`maybe_evict`). If any note is at least `EVICT_AGE_TOKENS` old,
   delete every note that is at least `EVICT_KEEP_TOKENS` old, and emit
   `evict` if anything was deleted.
7. **Done.** Emit `done` with the text and the stats (`events._done_stats`).

### Inside a block (`_run_block`)

`tree.run_block` runs these steps. Each is a function of its own, named in
brackets, in `tree.py` unless another module is given. Within a checkpoint
only the advance and the judge make requests; the other steps work on the
beams handed to them and nothing else.

1. Emit `block_open`.
2. **Fan out** (`fan_out`). One `top_next` request with `max_tokens=1` reads the
   distribution over the first thinking token
   (`k = min(20, BEAM_WIDTH × THINK_BRANCH_K)`).
   - The first `BEAM_WIDTH` tokens that contain a letter or digit and no
     marker become beams.
   - If the read is degenerate (fewer than `BEAM_WIDTH` usable openings), it
     is read once more and the better of the two reads is kept. If neither
     read yields a usable opening, the block runs as a single conversation.
   - This is the only point where beams are made to differ by construction.
     At temperature 0, two beams that share a prefix never separate.
   - Emit `frontier`.
3. **Checkpoint loop.** It runs until no live beam remains. Each checkpoint
   costs two requests.
   1. **Advance** (`advance`). One `top_next` request, with prompts
      `ctx + seed + beam.text` for every live beam, advances each by up to
      `THINK_CHUNK` tokens. Emit `think_token` as the text streams.
   2. **Expand** (`_expand`).
      - A beam whose chunk ended with a marker or a stop, or came back with
        no top-k, is **finished**. It moves to the pool, where it is still
        scored and can still win.
      - Every other beam proposes `THINK_BRANCH_K` candidates. Each one
        **replaces** the chunk's last token with an alternative, so the stem
        is not duplicated.
      - A candidate that copies the context pays `COPY_PENALTY`. One that
        starts writing the engine's own nudge line is cut there and finished.
   3. **Prune** (`prune`). Candidates are sorted by `rank_score`.
      - `width = max(1, BEAM_WIDTH − finished − closed)`.
      - `_prune` keeps at most `MAX_CHILDREN_PER_PARENT` per parent, and
        backfills only if the frontier cannot otherwise be filled.
   4. **Churn** (`churn`, then `_apply_churn`). `_churn` looks at the worst
      judged beam. If it is out of contention against the block's best raw
      score (`leader`):
      - While the block is still improving (`dry < CHURN_PATIENCE`), the beam
        is replaced by the best discarded sibling of the best lineage, if
        there is one.
      - Otherwise its slot closes.

      A lineage that now holds two slots has split, and the second gets a
      uid of its own (`_assign_split_uids`). Emit `think_step`.
   5. **Judge** (`scoring.score_axes`, `_update_leader`, `_award_medals`).
      One `answer_probs` request asks every contender (live beams and
      finished thoughts) about every judge axis.
      - `raw` and `overall` are written onto each beam (see
        [two scores](#two-scores)).
      - `dry` resets to 0 only when the best raw score beats `leader` by
        more than 0.01. Otherwise `dry` increments, and `leader` still takes
        any smaller gain.
      - Medals go to the top three by `overall`.
      - Emit `standings`.
   6. **End checks.**
      - If no live candidate is left, stop.
      - **Stop rule** (`_should_stop`): stop if at least
        `STOP_AFTER_FINISHED` thoughts have finished, the best leads the next
        by `STOP_MARGIN`, and no live beam is in contention.
      - **Ramble rule** (`_drop_ramblers`): a live beam longer than
        `RAMBLE_RATIO` × the longest finished thought is dropped, but only if
        it is also out of contention.
4. **Winner** (`finalise_block`, `_pick_winner`).
   - The best `raw` among finished thoughts that are at least 1/10 as long as
     the longest thought in the block.
   - If no finished thought qualifies, the best `raw` among all contenders.
   - Emit `monologue` with the winner's text.

## Event vocabulary

`Engine.respond` yields plain dicts, and `cli.py --json` prints them one per
line. The server forwards each one as an SSE `data:` line and adds `error` and
`end`. Type names, field names, whether a key is present at all, key order and
the rounding of each number are a contract with `cli.py`,
`static/index.html`, the measurement scripts, the goldens and existing
`events.jsonl` files. `events.EVENT_TYPES` lists the sixteen types the engine
yields, in the order a turn meets them, and the builders for the events with
a non-trivial shape are in `events.py`.

**Changing an event** (a new type, a field added, renamed or removed, a key
that is sometimes absent) updates all of these in the same commit:

- `adaptive_thinking/events.py`: the builder, and `EVENT_TYPES` for a new
  type (`tests/test_package.py` checks that every `"type"` the engine writes
  is listed);
- `adaptive_thinking/cli.py` and `adaptive_thinking/static/index.html`, which
  render it or deliberately ignore it;
- `adaptive_thinking/server.py`, if the type should be logged to
  `events.jsonl`;
- this section of this page;
- the goldens in `tests/golden/`, regenerated in that commit, whose message
  says the schema changed.

Axis names are spread into `window.probed[]` as keys, so renaming a gate axis
in `prompts.GATE_AXES` renames event fields too.

### Emitted by the engine

| type | fields | when |
|---|---|---|
| `start` | `user`, `prefix_chars` | Once, at the start of a turn. No consumer renders it. |
| `block_open` | `id`, `kind` (`"open"` or `"gate"`), `width`, `branch_k`, `tail`, `crit`, `gate`, `seed` | A block starts. For the opening thought, `tail` is `""` and `crit` and `gate` are `null`. For a gate block, `gate` is the chosen candidate's score dict (`crit`, `stakes`, `difficulty`, `commitment`, plus `p_yes` if logged). With `PROBE_ENABLED=false`, `crit` is NaN and `gate` is `{"crit": NaN}`. |
| `frontier` | `block`, `started`, `width`, `beams` | After the fan-out. `started < width` means the frontier started narrow. |
| `think_token` | `block`, `uid`, `text` | Streamed text of one conversation during a checkpoint. |
| `think_step` | `block`, `step`, `considered`, `ranked_by`, `beams`, `pool`, `ended`, `retired`, `entered`, `leader`, `dry`, `closed`, `dropped`, `distinct`, `backfilled` | After a checkpoint's prune and churn. See below. |
| `standings` | `block`, `ranked`, `live`, `finished` | After the checkpoint is judged. `ranked[]` is `{uid, rank, medal, done, overall, raw, axes}`; `axes` holds the four raw judge axes. |
| `monologue` | `block`, `text` | The winning thought, once per block. |
| `block_close` | `id`, `kind` (**opening thought only**), `width`, `steps`, `requests`, `winner_score`, `finished`, `explored_tokens`, `secs`, `was_going_to_write` | A block ends. For the opening thought, `kind` is `"open"` and `was_going_to_write` is `null`. A gate block's `block_close` has **no** `kind` key. |
| `window` | `probed`, `opened` | At least one position in the window passed the gate's filters. `probed[]` is `{pos, tok, h, crit, stakes, difficulty, commitment[, p_yes]}`, sorted by `crit` descending; `p_yes` is present only while `GATE_LOG_YESNO` is on. With the probe disabled it is `{pos, tok, h, crit: null}` in entropy order. `opened` says whether a block opens. |
| `rewind` | `tokens` | Answer tokens dropped to reach the chosen position. Only emitted when that number is above 0. |
| `token` | `text`, `h`, `lp`, `resume`, `top` | One answer token: entropy and logprob (3 decimal places), whether it falls in the resume-watch window, and `top[]` as `{t, p}` (at most `BRANCH_K` entries, `p` to 4 places). |
| `resumed` | `block`, `was`, `now`, `changed` | The first token after a gate block, emitted just before its `token` event. `changed` is the measurement: did the thinking change what the model wrote? |
| `restart` | `at`, `rewrote` | The note made the model begin the answer again. The engine rolled back and dropped the note. |
| `degraded` | `error`, `kept` | The endpoint failed mid-answer after its retries. The turn finishes with what it has. |
| `evict` | `notes`, `remaining`, `oldest_age`, `keep_newer_than`, `before_tokens`, `after_tokens`, `cache_hit_before` | Notes were evicted by age. `cache_hit_before` is `null` when `/metrics` is unavailable. |
| `done` | `text`, `stats` | End of turn. The stats are listed below. |

The beam objects in `frontier.beams`, `think_step.beams` and
`think_step.pool` all have the same shape:

```
{i, uid, from_uid, parent, tok, text, score, total, p, done}
```

- `score` is `rank_score` (3 decimal places).
- `total` is the cumulative logprob (2 decimal places).
- `p` is `null` unless `_rank` ran (4 decimal places).
- `tok` is the token that extended the beam. For a finished beam it is the
  last 12 characters of its chunk.
- `text` never includes the seed.

The other `think_step` fields:

- `considered`: live candidates this checkpoint.
- `ranked_by`: always `"logprob"`.
- `ended`: uids that were alive and are not carried forward, excluding those
  that finished.
- `retired`: `[{uid, overall, replaced}]`.
- `entered`: `[{uid, from_uid, tok}]`.
- `leader`: the best raw score so far (3 decimal places), and `dry`: the
  checkpoints without a new best. Both as they stood before this checkpoint
  was judged; the judge runs after `think_step` is emitted.
- `closed`: slots shut by churn.
- `dropped`: `considered − len(beams)`.
- `distinct`: slots holding a chunk no other slot holds.
- `backfilled`: `len(beams) − distinct`.

`done.stats`, with its keys in this order:

| key | meaning |
|---|---|
| `secs` | wall time of the turn |
| `kept_tokens` | answer tokens kept |
| `rewound_tokens` | tokens generated and then rewound away |
| `blocks` | all blocks, including the opening thought |
| `opening_thought` | `OPEN_THINK` |
| `blocks_that_changed_next_token` | gate blocks whose `resumed.changed` was true |
| `think_steps` | checkpoints across all blocks |
| `explored_tokens` | decode tokens spent inside blocks |
| `spills`, `spill_markers` | markers cut from the answer: `[{marker, at, kept, empty}]`. One at the final token is the model ending its turn; one mid-answer is not. |
| `restarts`, `restart_detail` | `[{at, rewrote}]` |
| `degraded` | error text, or `null` |
| `retries` | retries during the turn |
| `notes_in_context` | notes still in the conversation after eviction |
| `requests` | counted requests. Excludes `/tokenize` and `/metrics`. |
| `prompt_tokens` | server-reported; covers batched calls only |
| `completion_tokens` | decode tokens |
| `inline_blocks` | `KEEP_BLOCKS_INLINE` |
| `logprob` | mean answer logprob |
| `resume_logprob` | mean logprob over the resume-watch tokens |
| `resume_logprob_delta` | `resume_logprob − logprob` |
| `resume_samples` | how many tokens `resume_logprob` averages |

### Added by the server

| type | fields | when |
|---|---|---|
| `turn` | `session`, `q`, `inline` | Written to `events.jsonl` at the start of each `/chat` turn. **Not** streamed to the browser. |
| `error` | `message` (`"<ExceptionType>: <text>"`) | Streamed if `respond` raises. Not written to `events.jsonl`. |
| `end` | – | Always the last line of every `/chat` stream. |

`events.jsonl` receives one JSON object per line, written with
`ensure_ascii=False` and stamped with `t`, the unix time to 2 decimal places.
An engine event is written as `{"t": …, **event, "session": …}`, for the
types `window`, `block_open`, `block_close`, `resumed`, `monologue`, `evict`
and `done`. The `turn` record is `{"t": …, "type": "turn", "session": …,
"q": …, "inline": …}`. The high-volume `token` and `think_token` events are
not logged, and neither are `frontier`, `think_step`, `standings`, `rewind`,
`restart`, `degraded` or `start`.

### Order within a turn

```
start
[opening thought]  block_open  frontier  (think_token* think_step standings?)+  monologue  block_close
( token* | resumed token | restart | window [rewind] [block_open ... block_close] )*
[degraded]
[evict]
done
(server) end   or   error end
```

## Design invariants

Each of these rules is here because an earlier version broke it and a
measurement showed the damage. Changing one needs a new measurement, not a
tidier implementation.

1. **The beam width never grows.** Each checkpoint prunes the candidates to
   `max(1, BEAM_WIDTH − finished − closed)`. A frontier of 4 stays at most 4;
   it never becomes `BEAM_WIDTH × THINK_BRANCH_K`.
2. **The frontier shrinks only by finishing,** or by a retirement that closes
   a slot. A finished thought is not an empty slot. It is held by a completed
   conversation that is still scored and can still win, and slots filling up
   that way is how a block ends.
3. **A retirement is matched by an arrival, or it closes the slot.** A beam is
   never lost silently. `_churn` returns `(retired, entered)`. A closed slot is
   counted in `closed`, so the next checkpoint's backfill cannot reopen it.
4. **Only a bad score removes a live conversation.** `_in_contention(beam,
   leader)` is the single test. It is used by the churn, the stop rule and the
   ramble rule, and it asks only about `raw` score. Earlier versions also
   dropped beams for being long, for running many dry checkpoints, or because
   other thoughts had finished. Every one of those turned half-written text
   into a block's thought. Do not add another reason. (Pruning is a different
   step: it chooses among the `k` candidates a live beam proposes, on
   `rank_score`.)
5. **Nothing cuts off a thought that is still in contention.** A block ends
   when its beams close themselves, or when the stop rule finds no live beam
   in contention. A block has no step, token or wall-clock limit. A retired
   or dropped beam never becomes the block's thought, and a finished thought
   that passes the substance floor beats a live one.
6. **A block commits no answer token.** A block produces *thinking*, a note in
   the context. Generation resumes at the same position and the model picks
   its own next token. Whether that token differs from the one it was about to
   write (`resumed.changed`, `blocks_that_changed_next_token`) is the
   measurement. With `KEEP_BLOCKS_INLINE` off, the context after a block is
   byte-identical to the context before it, so at temperature 0 nothing can
   change. That mode is a control.
7. **Blocks stay where they were written.**
   - A note keeps its character offset in the answer for the life of the
     conversation.
   - The opening thought stays in the header.
   - `splice` and `think_head` are the single sources of these strings, so a
     turn renders byte-identically while generating and afterwards.
   - Notes are reclaimed only by **age**, using two watermarks.
     `EVICT_AGE_TOKENS` triggers a pass, which removes everything older than
     `EVICT_KEEP_TOKENS` at once. That keeps prefix rewrites, and the KV-cache
     losses they cause, rare.
8. **Prompts are byte-sensitive.** Temperature-0 decoding and prefix-cache
   hits both depend on the exact bytes. That covers:
   - every prompt builder and every seed,
   - the `think_head`/`NO_THINK` strings,
   - the axis wording and levels, which were measured and should not be
     tidied into adjectives,
   - the order of prompts inside a batch. The code finds each result by the
     arithmetic `candidate × len(axes) + axis`.
9. **A beam's `text` is the generated body only.** The seed is prepended when
   a prompt is built and never stored. `INLINE_SEED` contains `<think>`, which
   is a `SPILL` marker, so storing it would make `cut_at_marker` finish every
   beam on the first checkpoint.
10. **Config is read at call time.** Always `config.X` (and `chatml.X`,
    `prompts.X`), never `from .config import X`. The server's `?inline=`
    parameter, `experiments/tree_probe.py` and the tests all rely on runtime
    assignment.
11. **A stream is never replayed after its first token.** Only transient
    endpoint failures are retried, and only failures inside the answer loop
    become `degraded`.
12. **No single top-k read is trusted.** The endpoint has been observed to
    return a truncated top-k on some reads, so the fan-out reads twice when
    the first read is degenerate.

## Two scores

The judge produces two numbers for every contender, on purpose. They answer
different questions.

| score | what it is | range | used for |
|---|---|---|---|
| `raw` | the plain weighted mean of the four raw judge axes | absolute, 0..1 | the only number comparable **across** groups (live beams against finished thoughts): `_in_contention`, the churn, `leader`/`dry`, the stop and ramble rules, and choosing the winner. The block's `winner_score` is the winner's `raw`. |
| `overall` | each axis standardised **within its group** (by the group's range, clipped to ±`AXIS_CLIP`), then weighted and summed | relative, centred on 0 | ranking like against like: medals, the order of `standings`, the stop rule's lead between the best two finished thoughts, and the figure shown on a retired card |

Live beams and finished thoughts are standardised in separate groups. A
half-written thought scores low on "progress" simply for being half-written,
and mixing the groups would retire live beams for being alive. `overall`
therefore means nothing across groups, and `raw` has to answer "can this live
beam still catch that finished thought?" Do not merge them.

Two other numbers are unrelated to the judge:

- **`rank_score`** is `cumulative logprob / length − penalties`. It is the
  cheap signal that prunes candidates before any model judgement.
  Length-normalising it and adding a copy penalty stopped the search from
  preferring regurgitated context.
- **`crit`** is the gate's absolute 0..1 criticality. It is deliberately
  **not** standardised across positions: standardising would guarantee that
  some position in every window looks worth stopping for.

The judge weights are looping 1.0, progress 1.3, confusion 1.6 and
concreteness 0.7. The gate weights are stakes 1.4, difficulty 1.2 and
commitment 1.0. Each axis has five levels, labelled A–E from worst to best,
and the score is the expected level, scaled to 0..1.

## Known gaps

These are current behaviours, recorded so nobody rediscovers them the hard way:

- A transient endpoint failure **during the opening thought** propagates as
  an exception. Only the answer loop turns failures into `degraded`.
- `PROBE_ENABLED=false` puts `NaN` into `block_open`, which breaks the
  browser UI (see [CONFIG.md](CONFIG.md)).
- `CHOICE_EVERY` is not read, and `_rank` is reachable only from
  `experiments/tree_probe.py`.
- `SIBLING_PENALTY` never applies.
- `?inline=` on `/chat` changes a process-global setting. Sessions live in
  memory with no limit.
- Temperature 0 is not bit-for-bit reproducible on a batching server. Two
  runs on identical input have taken 28 and 43 checkpoints.
