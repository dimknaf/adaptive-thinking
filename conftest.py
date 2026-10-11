"""Keeps `pytest .` and IDE test discovery away from the scripts.

The offline suite is in tests/ (see pyproject.toml). Everything listed here is
a script, a launcher or the engine package, not a test module. The
measurement scripts in experiments/ talk to a live endpoint, so a test run
must never import them by accident; the tests that exercise them import them
explicitly and replace the endpoint with a fake.

This file deliberately imports nothing from the project. Importing
adaptive_thinking.config would read .env before tests/conftest.py has had a
chance to neutralise it.
"""

collect_ignore = [
    # launchers for adaptive_thinking.cli and adaptive_thinking.server
    "cli.py",
    "server.py",
    # the engine package (imported by the tests, never collected as tests)
    "adaptive_thinking",
    # measurement scripts, and the archived ones in experiments/archive
    "experiments",
]

collect_ignore_glob = [
    "experiments/**",
]
