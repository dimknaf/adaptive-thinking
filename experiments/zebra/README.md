# Zebra puzzles: native thinking against the tree

This is a self-contained experiment comparing two ways of solving a zebra
(logic-grid) puzzle. In the **native** arm, Qwen answers with its own `<think>`
reasoning. In the **tree** arm, this repository's engine answers. Every puzzle
is certified by a solver to have exactly one solution, and grading is an exact
match on the whole grid, not a keyword search. The runner drives the engine
through the `adaptive_thinking` package, and everything here runs from the
repository root as modules of the `experiments.zebra` package.

This is the one comparison in the repository where the tree came out ahead. The
caveats at the end matter as much as the result.

## Files

| file | what it is |
|---|---|
| `generate.py` | Generator (formerly `zebra.py`). N houses, N attribute categories, each category a permutation of N values. Clues are drawn from the true solution and added until the solution is unique, then removed one at a time while uniqueness still holds, so the final set is minimal. |
| `zsolve.py` | CSP solver used to certify uniqueness. Each (category, value) pair is a variable over the houses, with all-different inside each category. It propagates, then branches on the smallest domain, and counts solutions, stopping at 2. `generate.py` also keeps a brute-force counter as a cross-check oracle; it is only practical up to 4x4. |
| `zebra_3x3_s1.json` … `zebra_5x5_s2.json` | The six puzzles used: 3x3, 4x4 and 5x5, seeds 1 and 2. Each file holds the prompt, the solution and the clue count (3x3: 4 and 5 clues; 4x4: 10 and 11; 5x5: 19 and 18). |
| `run.py` | Runner. It sends one puzzle through one arm and grades the answer against the solution grid. |
| `results.json` | The recorded runs, one JSON object per line. This is JSON Lines despite the extension. |
| `native_5x5_s1_raw.txt` | The raw native stream from the failed 5x5 seed 1 run. The runner writes this stream to `native_raw.txt`; the committed copy was renamed after the run. |

## Running

**Generate and certify a puzzle.** From the repository root:

```bash
python -m experiments.zebra.generate 5 1   # prints the puzzle, the solution, the clue
                                           # count and unique=True, and writes
                                           # experiments/zebra/zebra_5x5_s1.json
```

The puzzle JSON is written next to `generate.py`, where the runner looks for
it, whatever the current directory. (The generator used to write it into the
current directory instead.)

Generation is deterministic in (N, seed). Regenerating a committed puzzle
rewrites its file with identical content. A 5x5 puzzle takes several minutes of
CPU to generate; the smaller ones take well under a second. Generation needs
no endpoint. The offline suite (`tests/test_zebra.py`) checks the generator,
the solver and the grader, and that regeneration reproduces the committed
puzzles; the two 5x5 regenerations are marked `slow` and run only with
`pytest -m slow`.

**Run a puzzle through one arm.** This needs a configured endpoint (`.env` in
the repository root). Run it from the repository root:

```bash
python -m experiments.zebra.run native 5 1 low
python -m experiments.zebra.run ours   5 1 low
```

The usage is `python -m experiments.zebra.run native|ours N SEED
[low|medium|xhigh]`, and the effort level defaults to `low`. The runner loads
the committed puzzle JSON if it exists, and otherwise generates the puzzle and
caches it in `experiments/zebra/`, so both arms get a byte-identical prompt.

- **native.** Builds the ChatML prompt (system prompt, then the user turn) and
  opens the assistant turn without a thinking block, so Qwen thinks natively.
  It streams `/v1/completions` at temperature 0, stopping at `<|im_end|>`, with
  `max_tokens` set to the server's `max_model_len` less a prompt-length estimate
  and a small margin. There is no wall-clock limit. The stream is written to
  `experiments/zebra/native_raw.txt` as it arrives. The answer is the text after
  `</think>`; if `</think>` never appears, the answer is empty and the run
  fails.
- **ours.** Runs the engine with the same system prompt and whatever settings
  `adaptive_thinking/config.py` resolves from `.env` and the environment. It
  prints a line each time a block closes, so several minutes of silence is
  normal.

Each run appends one line to `experiments/zebra/zebra_results.jsonl`, which is
not tracked. The committed `results.json` holds the runs listed below.

**Grading.** The grader reads lines of the form
`House N: name=..., drink=..., ...`, case-insensitively. A later mention of a
house overrides an earlier one. A run passes only if every cell matches. An
answer written as a markdown table does not parse and scores 0 even when it is
correct, so check the printed answer tail before reading a FAIL as a reasoning
failure.

## Results

Every row of `results.json`:

| arm | puzzle | effort | result | cells | time (s) | notes |
|---|---|---|---|---|---|---|
| native | 3x3, seed 1 | none | PASS | 9/9 | 45.7 | 1,902 thinking characters |
| native | 3x3, seed 2 | none | PASS | 9/9 | 29.4 | 1,131 thinking characters |
| native | 4x4, seed 1 | none | PASS | 16/16 | 85.6 | 3,468 thinking characters |
| native | 4x4, seed 2 | none | PASS | 16/16 | 66.8 | 2,440 thinking characters |
| native | 5x5, seed 2 | none | PASS | 25/25 | 506.3 | 17,974 thinking characters |
| native | 5x5, seed 1 | low | **FAIL** | 0/25 | 1,562.4 | 89,395 characters; `</think>` never closed; budget 261,122 tokens |
| tree | 5x5, seed 1 | none | **PASS** | 25/25 | 1,619.6 | 1 block, 32,648 tokens explored |
| tree | 5x5, seed 1 | low | **PASS** | 25/25 | 3,703.0 | 3 blocks, 58,725 tokens explored |

"none" means the record has no effort field: those runs predate the effort
argument and sent no effort sentence, which is equivalent to `medium`. Both tree
runs recorded 0 restarts, 0 retries and no degraded turn. They used the knob
values that `.env.example` ships (`GATE_THRESHOLD=0.80`, `BEAM_WIDTH=4`,
`THINK_CHUNK=500`, `OPEN_THINK=true`); the result records themselves do not
store the configuration. Every run was on
`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`.

**The native failure.** `finish=stop` with `</think>` never closed: the model
ended its own turn mid-word while still thinking (the stream's last characters
are `Dr`). It did not run out of room. It used about 25k tokens (estimated from
the character count) of a 261,122-token budget, leaving roughly 90% unused.
`native_5x5_s1_raw.txt` shows genuine case-splitting: contradiction after
contradiction, never converging.

## The `reasoning_effort` dial

`reasoning_effort` is nothing more than a sentence the server's chat template
puts in front of the system prompt. The sentences were recovered by posting chat
messages to `/tokenize` and reading what the template rendered, and they are
stored in `run.py` (`EFFORT`). `medium` adds nothing. `xhigh`, the template's
default on the server used, asks the model to "think carefully through the task,
validate key assumptions, consider plausible alternatives". `low` asks it to
"keep your thinking brief and focused". The runner uses raw completions, where
no template runs, so it prepends the same sentence to the system prompt of both
arms.

Measured effect:

- **It did not help a model that was stuck.** Native produced *more* text at
  `low` than without the sentence. That uninstructed native run is not recorded
  in `results.json`.
- **It changed the shape of the tree's search.** At `low`, beams finished far
  more readily (14 and 30 finished thoughts per block, against 1), but the run
  cost more overall: 3,703 s against 1,620 s.

## What this does not establish

- **n = 1.** It rests on one puzzle, two tree runs and one native run on that
  puzzle.
- **The puzzle was selected on failure.** Native solved the other five puzzles.
  The tree was run only on the one that native failed, not on the other five.
- **The effort settings do not match.** The only recorded native run on this
  puzzle used `low`, while the faster tree pass sent no effort sentence.
- **Runs vary a lot.** Identical input at temperature 0 has taken 28 and 43
  checkpoints on different runs, from the server's own run-to-run variation.
- **The tree is slow.** Both passes (1,620 s and 3,703 s) took longer than the
  failed native run (1,562 s), and far longer than native took on the puzzles it
  solved.
- **The tree did not think deeper.** The beams share no partial results, so the
  tree is closer to parallel sampling with quality selection than to search.
  Many thoughts complete, and only one has to be right. Native has a single
  thread, and once that thread entered a backtracking loop the whole run went
  with it.
- **The search was not efficient.** In the three-block run, one mid-answer block
  spent 51% of the run re-deriving a question the opening thought had already
  settled, and the churn never fired.
- **One model.** Every run was on `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`.

See [docs/FINDINGS.md](../../docs/FINDINGS.md) for the rest of the measurements.
