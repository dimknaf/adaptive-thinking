# Contributing

This is a research prototype. The measurement scripts matter as much as the
engine does: most of the design exists because a measurement showed that an
earlier version did not work. Contributions are welcome. So are well-measured
reports that something here does not help.

Before changing the engine, read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
especially the design invariants, and [Structure and rules](#structure-and-rules)
below. [docs/CONFIG.md](docs/CONFIG.md) lists every knob.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

The code is developed and tested on Python 3.11, which is also what CI uses.

Only live runs need an endpoint. For those, `cp .env.example .env` and fill in
`OPENAI_BASE_URL`, `OPENAI_MODEL` and `OPENAI_API_KEY`. The tests never read
`.env`.

## Running the offline tests

```bash
ruff check .        # lint; CI runs it before the tests
pytest              # the offline suite: no endpoint, no .env, no network
pytest -m slow      # the CPU-heavy tests, e.g. regenerating the 5x5 zebra puzzle (about 6 minutes)
```

- `pyproject.toml` configures both. Its `[tool.ruff]` section excludes
  `experiments/archive/`, which holds historical scripts kept unchanged.
  Its `[tool.pytest.ini_options]` section sets `testpaths = tests` and
  deselects the `slow` and `live` markers by default. A `-m` given on the
  command line replaces that default.
- `tests/conftest.py` makes the suite hermetic before anything imports
  `adaptive_thinking.config`:
  - it stops `.env` from loading,
  - it clears every variable `adaptive_thinking/config.py` reads, so the
    code defaults apply,
  - it points the endpoint at a host that does not exist,
  - and a per-test guard fails any real HTTP request.
- The engine is driven by a scripted fake endpoint, `tests/fake_endpoint.py`:
  `FakeVLLM` stands in for the client at the engine level, and its wire-level
  helpers drive the real client over `httpx.MockTransport`.
- The root `conftest.py` keeps the launchers, the package and `experiments/`
  out of collection, so `pytest .` or an editor's test discovery never
  imports a live measurement script.

### Golden (characterisation) tests

`tests/golden/` holds recorded request logs and event streams, produced by
running the real client and engine (and the CLI and the server) against a
scripted fake vLLM server. `harness.py` is the machinery and the fake server,
`scenarios.py` the scripted situations, and `<scenario>.json` the stored
recordings; `tests/test_golden.py` compares a fresh recording with each. They
pin today's behaviour byte for byte:

- every prompt the engine sends, in order, with its payload keys and types;
- every event it emits, with its fields and their order;
- the conversation it leaves behind, the CLI's output and the server's
  responses;
- the knobs each recording ran under.

A refactor must leave them unchanged.

Regenerate them only on purpose, in a commit whose message states the
behaviour change it intends:

```bash
UPDATE_GOLDENS=1 pytest tests/test_golden.py
```

Review the diff of `tests/golden/` as carefully as a code diff. It *is* the
behaviour change.

## Live measurement scripts are not tests

These need a real endpoint. A run takes minutes to an hour, and the output is
a measurement for a person to read, not a pass/fail result:

| script | what it measures |
|---|---|
| `python -m adaptive_thinking.llm` | smoke test: endpoint, logprobs, `logprob_token_ids`, list-of-prompts batching |
| `python cli.py "question"` | one turn of the engine, rendered in the terminal |
| `python -m experiments.gate_audit [--run]` | where the gate fires, and whether for good reasons |
| `python -m experiments.bench_hard` | trap questions: the model's own thinking against the engine (the two arms do **not** get the same limit; see [CONFIG.md](docs/CONFIG.md)) |
| `python -m experiments.tree_probe` | tree shape, frontier diversity, label-position bias |
| `python -m experiments.context_probe` | notes stay in place and eviction works; set small `EVICT_*` watermarks |
| `python -m experiments.zebra.run native\|ours N SEED [effort]` | one certified zebra puzzle, graded by exact grid match |

The scripts in `experiments/` are modules of that package: run them from the
repository root with `python -m`, as above. What they share is in
`experiments/_common.py`. Generating a zebra puzzle
(`python -m experiments.zebra.generate N SEED`) needs no endpoint, only CPU.

`python -m experiments.degraded_check` is the exception: it runs offline
against a stub. It is still a script, not part of the pytest suite. It
prints OK/FAIL lines and exits with status 1 if any check failed;
`tests/test_degraded_turn.py` covers the same scenario with assertions.

## Working principles

- **Measure before claiming.** A number in a doc or a comment should come
  from a run someone can point to. State the model, the settings and n.
- **Report negative results.** Two of the most useful findings here are
  negative. With the blocks hidden from the context, the mechanism changed
  nothing. A labelled single-token Choice, inspired by jevfire, did not
  transfer to beam selection. Record what did not work as carefully as what
  did.
- **Comments explain why, and cite the measurement** where one exists. A
  default without its reason is a default someone will "tidy up".
- **Never give two arms of a comparison different limits.** A token cap, a
  timeout or an effort setting applied to one arm goes in front of both, or
  the comparison is not reported. Omitting `max_tokens` does not mean
  unlimited: `/v1/completions` defaults it to 16. For an uncapped run, read
  `max_model_len` from `/v1/models`.
- **Run one measurement at a time against an endpoint.** If others use the
  same server, check its queue first (`num_requests_waiting` on vLLM's
  `/metrics`). Concurrent runs distort each other's timing and can write
  over each other's output files.
- **Write raw streams to disk as they arrive.** A run that fails with only
  summary counts logged cannot be diagnosed, because a stuck model and a
  broken harness look the same.
- **Temperature 0 means a repeat re-measures the same answer.** Get breadth
  from more tasks, not more runs. A batching server is not bit-exact, so
  repeats can still differ. That is noise, not a second sample.
- **Fix mechanisms, not symptoms.** Do not use banned-string lists,
  `bad_words`, or phrase-specific patches for model behaviour. Banning both
  spellings of one word once made the model emit nothing.
- **Keep the invariants.** The rules in
  [ARCHITECTURE.md](docs/ARCHITECTURE.md#design-invariants) are there because
  each was broken once and the damage was measured. Changing one needs a new
  measurement.

## Structure and rules

The code is arranged so that a change can be checked, and these rules keep it
that way. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#module-map-and-layering)
has the module map and the reasoning behind them.

### Where code goes

- **The package is layered, and imports point down:**

  ```
  config  <-  chatml / prompts  <-  llm  <-  text / state  <-  scoring  <-  gate / tree  <-  engine  <-  cli / server / UI
  ```

  A module uses only the layers below its own. `paths.py` and
  `events_log.py` sit at the bottom beside `config.py` and import nothing
  from the package; `events.py` (the event builders) sits with `text` and
  `state`. New code goes in the lowest layer that can hold it: a string
  helper in `text.py`, a request shape in `llm.py`, a step of the search in
  `tree.py`, a gate rule in `gate.py`, the sequencing of a turn in
  `engine.py`.
- **Experiments depend on the package, never the reverse.** Nothing in
  `adaptive_thinking/` imports `experiments/` or `tests/`. What several
  scripts need goes in `experiments/_common.py`, not into the engine.
- **The front ends add no logic.** `cli.py`, `server.py` and
  `static/index.html` render the engine's events. Anything the UI shows is
  something the CLI can print too.

### Rules for the code

- **Prompts are measured, byte-sensitive artefacts.** The engine decodes at
  temperature 0 over a prefix cache, so the wording in
  `adaptive_thinking/prompts.py`, the chat-format strings in
  `adaptive_thinking/chatml.py`, the seeds, and the order of prompts in a
  batch all change behaviour, however cosmetic an edit looks. Never tidy
  their wording. A change to them is a behaviour change: measure it, and
  regenerate the goldens in that commit.
- **Read config at call time.** Always `config.X` (and `chatml.X`,
  `prompts.X`), never `from .config import X`: a copy taken at import misses
  every runtime assignment, and the server's `?inline=` switch,
  `experiments/tree_probe.py` and the tests all assign. `tests/test_package.py`
  rejects such an import anywhere in the package.
- **A knob that nothing reads is deleted, not kept.** Nothing means no
  module of the engine, no front end and no measurement script. Knobs the
  engine does not read but a script or `/config` does are listed as such in
  [docs/CONFIG.md](docs/CONFIG.md), and removed knobs are recorded there.
- **Adding a knob** touches, in the same commit:
  - `adaptive_thinking/config.py`, read through `_i`, `_f`, `_b` or
    `os.getenv`, with a comment giving its reason and the measurement behind
    its default;
  - `.env.example`, set or commented out at its default (`tests/test_config.py`
    fails if `config.py` and `.env.example` disagree in either direction);
  - `docs/CONFIG.md`: a row in the right table, and the count at the top;
  - `tests/golden/harness.py`: the knob pinned in `KNOBS`
    (`tests/test_golden.py` fails otherwise). The goldens record the knobs
    they ran under, so this commit regenerates them and says why;
  - if a front end shows it: `config.summary()`, which is `/config`, and
    `adaptive_thinking/static/index.html`.

  **Removing a knob** deletes it from the same places, and lists it in
  `RETIRED_KNOBS` in `tests/golden/harness.py` so that the stored goldens stay
  byte-identical until they are next regenerated on purpose.
- **The event schema is a contract.** A change to an event's type, its fields,
  their order or whether a key is present updates, in the same commit:
  - `adaptive_thinking/events.py` (and `EVENT_TYPES`, for a new type);
  - `adaptive_thinking/cli.py`;
  - `adaptive_thinking/static/index.html`;
  - `docs/ARCHITECTURE.md`;
  - the goldens.
- **Comments explain why, and cite the measurement** where one exists. A
  default without its reason is a default someone will "tidy up".

### Rules for tests

- **Tests are offline and hermetic.** The suite in `tests/` reads no `.env`,
  makes no network request and needs no endpoint; a guard fails any test that
  tries. Live measurement scripts live in `experiments/` and are never
  collected by pytest. A test that needs a live endpoint is marked `live` and
  is opt-in only.
- **Goldens change only on purpose.** A refactor leaves `tests/golden/*.json`
  byte-identical; if one differs, the refactor changed behaviour and is fixed,
  not the golden. Goldens are regenerated only in a commit whose message
  states the behaviour change it intends.
- **Keep behaviour changes and refactors in separate commits,** and run the
  whole offline suite and `ruff check .` before each one.

### Rules for measurements

- **Negative results are reported, not tuned away.** See
  [Working principles](#working-principles).
- **Both arms of a comparison get identical limits:** the same token cap, the
  same timeout, the same effort setting, or none for either.
- **One live run at a time against an endpoint.** If others use the same
  server, check its queue first.

## Pull requests

- Keep behaviour changes and refactors in separate commits. A refactor leaves
  the goldens unchanged.
- Describe what was measured, on what model and settings, and include results
  that went the wrong way.
- CI runs `ruff check .` and the offline suite on Python 3.11. It needs no
  secrets and no endpoint.
