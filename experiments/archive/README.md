# Archived measurement scripts

These four scripts produced some of the historical numbers in
[docs/FINDINGS.md](../../docs/FINDINGS.md). They target an earlier version of
the engine and are kept **unchanged**, as provenance rather than as working
tools.

| script | what it measured | engine |
|---|---|---|
| `calibrate.py` | The entropy distribution of generated tokens over a spread of prompts, and which tokens sit above each candidate threshold. It showed that the highest-entropy positions are synonym choices, not decisions. Writes `calibration.jsonl`. | v1 prototype |
| `probe_check.py` | Rebuilds the context at every position recorded by `calibrate.py` and asks the binary Yes/No gate probe. It measured P(yes) against entropy and against task type: 0.407 against 0.484 for high- and mid-entropy positions, and 0.622 for arithmetic against 0.202 for essay prose. | v1 prototype |
| `bench.py` | Plain against hidden-block against inline-block generation on 14 arithmetic word problems with checkable answers. It measured the hidden-block no-op (the next token changed 0 times) and the inline repetition rate. | v2 engine (per-token tree, width 5) |
| `inspect_inline.py` | Prints the raw model-facing context around every block, the monologue, and the tokens produced right after it. It is the diagnostic used to tell model imitation apart from a splicing bug. | v1 prototype |

`calibrate.py`, `probe_check.py` and `inspect_inline.py` were written for the v1
prototype and never changed before their removal. `bench.py` last changed with
the v2 engine.

## They do not run against the current engine

They target the earlier engine:

- **`bench.py`** reads `config.MAX_TOKENS`, which no longer exists, so it
  raises `AttributeError`.
- **`inspect_inline.py`** sets `config.MAX_BLOCKS` and `config.MAX_TOKENS`.
  Nothing reads those any more, so the limits it relies on are silently
  ignored.
- **`calibrate.py` and `probe_check.py`** measure the binary P(yes) probe
  (`PROBE_THRESHOLD`). That probe no longer decides anything: the current gate
  scores levelled axes against `GATE_THRESHOLD`. Their numbers describe a
  signal the engine no longer uses.

All four import the engine as top-level modules (`config` and `llm`, and
`bench.py` and `inspect_inline.py` also import `thinking`). Those modules sat
at the repository root, which is not on their import path from this
directory, and they no longer exist under those names: the engine is now the
`adaptive_thinking` package. All four also call a live endpoint.
They are not part of the test suite, pytest never collects them, and the
linter skips this directory (`pyproject.toml`), so they stay byte-for-byte as
they were.

To see what they measured, read [docs/FINDINGS.md](../../docs/FINDINGS.md),
where each number is labelled with its engine and script. To measure the
current engine, use the scripts one level up, in `experiments/`, run from the
repository root as modules (`python -m experiments.tree_probe`); see
[CONTRIBUTING.md](../../CONTRIBUTING.md#live-measurement-scripts-are-not-tests).
