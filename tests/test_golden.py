"""Golden characterisation tests: what the engine does, byte for byte.

Each scenario in tests/golden/scenarios.py drives the real llm.VLLM and
engine.Engine (and cli.py and server.py) against a scripted fake vLLM server
over httpx.MockTransport. It records every request sent, every event
emitted, and the conversation left behind. The recording must equal the
stored golden, tests/golden/<scenario>.json, exactly. tests/golden/harness.py
describes what is recorded and how the fake server answers.

The goldens pin CURRENT behaviour, quirks included; FEATURES in scenarios.py
lists what each scenario is there to exercise. A change meant to alter no
behaviour (a refactor, a package move, a helper extracted) must leave every
one of them passing untouched. A change meant to alter behaviour regenerates
them in the same commit, and the commit message says what changed and why:

    UPDATE_GOLDENS=1 python -B -m pytest -p no:cacheprovider tests/test_golden.py

(PowerShell: $env:UPDATE_GOLDENS='1'; ... ; Remove-Item Env:UPDATE_GOLDENS)

Regenerating rewrites every golden from the current code, then checks that
each file reads back to exactly what was recorded. Only UPDATE_GOLDENS=1
writes anything; without it the tests only read. Review the diff of the
regenerated files before committing it.
"""

import os

import pytest

from golden import harness, scenarios

UPDATE = os.environ.get("UPDATE_GOLDENS") == "1"
NAMES = list(scenarios.SCENARIOS)


@pytest.mark.parametrize("name", NAMES)
def test_recording_matches_its_golden(name, tmp_path):
    record, about = scenarios.SCENARIOS[name]
    fresh = record(tmp_path)
    if UPDATE:
        harness.save(name, about, fresh)
    if not harness.golden_path(name).exists():
        pytest.fail(f"no golden for {name!r}; record one with UPDATE_GOLDENS=1 (see the "
                    "module docstring)", pytrace=False)
    diff = harness.first_difference(harness.load(name), fresh)
    if diff is not None:
        pytest.fail(
            f"{name}: the engine's behaviour differs from tests/golden/{name}.json.\n"
            f"First difference: {diff}\n"
            "If the change is intended, regenerate with UPDATE_GOLDENS=1 and say why in "
            "the commit message.",
            pytrace=False,
        )


@pytest.mark.parametrize("name", NAMES)
def test_golden_still_exercises_its_features(name):
    if not harness.golden_path(name).exists():
        pytest.skip("golden not recorded yet")
    stored = harness.load(name)
    missing = [what for what, check in scenarios.FEATURES.get(name, {}).items()
               if not check(stored)]
    assert missing == [], f"{name} no longer exercises: {missing}"


def test_every_environment_knob_is_pinned():
    harness.check_knobs(harness.KNOBS)
    assert harness.env_knob_names() | harness.DERIVED_KNOBS == set(harness.KNOBS)


def test_every_stored_golden_has_a_scenario():
    stored = {p.stem for p in harness.GOLDEN_DIR.glob("*.json")}
    assert stored <= set(NAMES), f"golden files with no scenario: {sorted(stored - set(NAMES))}"


def test_recording_twice_in_one_process_gives_the_same_result(tmp_path):
    # Nothing leaks from one run into the next: no cached labels, no server
    # singleton, no knob left behind.
    first = scenarios.b_plain_turn(tmp_path)
    second = scenarios.b_plain_turn(tmp_path)
    assert harness.first_difference(first, second) is None


def test_the_fake_server_answers_the_same_bytes_the_same_way():
    # The fake is a function of the request: two servers built from the same
    # script answer an identical request identically.
    import json

    import httpx

    prompt = (harness.USER + "Q" + harness.IM_END + "\n" + harness.ASSISTANT
              + "<think>\n\n</think>\n\n")
    body = json.dumps({"model": "test-model", "prompt": prompt, "max_tokens": 8,
                       "temperature": 0.0, "logprobs": 5, "stop": [], "stream": True}).encode()
    model = harness.Model(harness.TurnScript("Q", "One| two|.| Three|."))
    answers = []
    for _ in range(2):
        server = harness.FakeServer(model)
        req = httpx.Request("POST", "http://vllm.test/v1/completions", content=body)
        answers.append(server(req).content)
    assert answers[0] == answers[1]
    assert b'"text": "One"' in answers[0] and b'"finish_reason": "stop"' in answers[0]
