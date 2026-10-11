"""Measurement scripts for the adaptive_thinking engine.

Each one runs from the repository root as a module:

    python -m experiments.tree_probe       is the thinking block a real tree?
    python -m experiments.context_probe    do notes stay in place; does eviction work?
    python -m experiments.gate_audit       where the gate fires, and why
    python -m experiments.bench_hard       trap questions, native thinking against the tree
    python -m experiments.degraded_check   offline: a 524 after the opening thought
    python -m experiments.zebra.run        one certified zebra puzzle, graded exactly
                                           (zebra/README.md has the experiment)

All but degraded_check (and gate_audit without --run) call the live endpoint
configured in the repository's .env. None of them is a unit test, and pytest
never collects this directory; the offline suite is in tests/. archive/ holds
scripts written for an earlier engine, kept unchanged as provenance.

Importing this package has no side effects.
"""
