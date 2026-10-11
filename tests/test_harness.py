"""The harness itself: hermetic config, the network guard, and both fakes.

If one of these fails, every other test in the suite is suspect, so they are
kept small and direct.
"""

import ast
import asyncio
import importlib
import os
import re
import sys
from pathlib import Path

import httpx
import pytest

from adaptive_thinking import chatml, config, llm, prompts
from adaptive_thinking.scoring import _expected
from adaptive_thinking.state import Conversation
from fake_endpoint import (
    BrokenStream,
    Router,
    Stream,
    answer_probs_json,
    asked_axis,
    context_of,
    error_response,
    json_response,
    ln,
    lvl,
    make_mock_client,
    metrics_text,
    quiet,
    run_turn,
    sse,
    sse_response,
    step,
    stream_chunk,
    thought_of,
    tokenize_json,
    types,
    windows,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_SRC = (ROOT / "adaptive_thinking" / "config.py").read_text(encoding="utf-8")

# Knobs the autouse network_guard fixture overrides for every test.
PATCHED_BY_HARNESS = {"LLM_RETRY_BACKOFF": 0.0}


def _source_defaults():
    """{NAME: (ENV_VAR, default)} for every knob config.py reads from the
    environment, taken from its source: `NAME = _f/_i/_b("ENV", default)` and
    `NAME = os.getenv("ENV", "") or <default>`."""
    convert = {"_f": float, "_i": int, "_b": bool}
    out = {}
    for node in ast.parse(CONFIG_SRC).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name, v = node.targets[0].id, node.value
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                and v.func.id in convert):
            env = ast.literal_eval(v.args[0])
            out[name] = (env, convert[v.func.id](ast.literal_eval(v.args[1])))
        elif (isinstance(v, ast.BoolOp) and isinstance(v.op, ast.Or)
              and len(v.values) == 2 and isinstance(v.values[0], ast.Call)
              and isinstance(v.values[0].func, ast.Attribute)
              and v.values[0].func.attr == "getenv"
              and isinstance(v.values[1], ast.Constant)):
            out[name] = (ast.literal_eval(v.values[0].args[0]), v.values[1].value)
    return out


# ---------------------------------------------------------------------------
# (a) config is hermetic
# ---------------------------------------------------------------------------


def test_every_knob_holds_its_code_default():
    defaults = _source_defaults()
    # The parser really found the knobs, rather than trivially passing.
    assert len(defaults) >= 30, sorted(defaults)
    wrong = {
        name: {"config": getattr(config, name), "source default": default}
        for name, (env, default) in defaults.items()
        if not env.startswith("OPENAI_")
        and name not in PATCHED_BY_HARNESS
        and getattr(config, name) != default
    }
    assert wrong == {}
    for name, value in PATCHED_BY_HARNESS.items():
        assert getattr(config, name) == value


def test_user_agent_is_the_code_default():
    env, default = _source_defaults()["USER_AGENT"]
    assert env == "LLM_USER_AGENT"
    assert config.USER_AGENT == default
    assert "LLM_USER_AGENT" not in os.environ


def test_endpoint_settings_point_at_a_host_that_does_not_exist():
    assert os.environ["OPENAI_API_KEY"] == "test-key"
    assert config.API_KEY == "test-key"
    assert config.MODEL == "test-model"
    assert config.BASE_URL == "http://vllm.test/v1"
    assert config.ROOT_URL == "http://vllm.test"


def test_no_env_var_that_config_reads_is_left_set():
    read = set(re.findall(r'(?:\b_[fib]|os\.getenv)\(\s*"([A-Z][A-Z0-9_]*)"', CONFIG_SRC))
    assert {"BEAM_WIDTH", "GATE_THRESHOLD", "INLINE_SEED", "LLM_USER_AGENT"} <= read
    left = sorted(n for n in read if n in os.environ and not n.startswith("OPENAI_"))
    assert left == []


def test_dotenv_is_neutralised_before_config_binds_it():
    import dotenv

    # config did `from dotenv import load_dotenv`; what it bound is the stub
    # defined in tests/conftest.py, not python-dotenv's function.
    code = config.load_dotenv.__code__
    assert Path(code.co_filename).resolve() == ROOT / "tests" / "conftest.py"
    assert dotenv.load_dotenv is config.load_dotenv
    assert config.load_dotenv(ROOT / ".env") is False


def test_the_optional_label_import_is_blocked():
    # llm.labels() used to try this package before /tokenize and no longer
    # imports it; blocked for the session, a copy installed on the machine
    # can never change the labels or download a tokenizer.
    assert sys.modules["jevfire"] is None
    with pytest.raises(ImportError):
        importlib.import_module("jevfire")


def test_cfg_sets_knobs_and_rejects_typos(cfg):
    got = cfg(BEAM_WIDTH=2, OPEN_THINK=False)
    assert got is config
    assert (config.BEAM_WIDTH, config.OPEN_THINK) == (2, False)
    with pytest.raises(AttributeError, match="BEAM_WIDHT"):
        cfg(BEAM_WIDHT=3)


def test_no_measurement_script_was_imported(request):
    # The measurement scripts live in experiments/ (test_tree.py,
    # test_context.py and test_degraded.py were renamed on the way) and talk
    # to a live endpoint. Tests may import them on purpose, with a fake in
    # place of the endpoint, but none may be collected as a test module, and
    # nothing may import them under their old root-level names.
    for name in ("test_degraded", "test_tree", "test_context"):
        assert name not in sys.modules
    experiments = ROOT / "experiments"
    collected = {Path(str(item.fspath)).resolve() for item in request.session.items}
    assert [p for p in collected if experiments in p.parents] == []
    assert all(p.parent == ROOT / "tests" for p in collected), sorted(
        str(p) for p in collected if p.parent != ROOT / "tests")


def test_the_zebra_modules_resolve_inside_the_experiments_package():
    from experiments.zebra import generate, zsolve

    # experiments/zebra/generate.py (formerly zebra/zebra.py) and its solver,
    # found through the package rather than a zebra/ entry on sys.path.
    zebra_dir = ROOT / "experiments" / "zebra"
    assert Path(generate.__file__).resolve() == zebra_dir / "generate.py"
    assert Path(zsolve.__file__).resolve() == zebra_dir / "zsolve.py"
    assert callable(generate.generate) and callable(zsolve.solve)
    assert generate.zsolve is zsolve
    assert all(Path(p).resolve() != ROOT / "zebra" for p in sys.path if p)


# ---------------------------------------------------------------------------
# (b) real network access is impossible
# ---------------------------------------------------------------------------


def test_a_real_request_raises_the_guard_error(network_guard):
    with pytest.raises(AssertionError, match="real network call in a unit test"):
        httpx.get("http://vllm.test/v1/models")

    async def go():
        client = llm.VLLM()  # a real client with the real transport
        try:
            with pytest.raises(AssertionError, match="real network call in a unit test"):
                await client.tokenize("hello")
        finally:
            await client.aclose()

    asyncio.run(go())
    assert network_guard.attempts == [
        "GET http://vllm.test/v1/models",
        "POST http://vllm.test/tokenize",
    ]
    network_guard.attempts.clear()  # handled here, so teardown stays green


def test_a_swallowed_network_error_still_fails_the_test(network_guard):
    # prefix_cache_stats() catches every exception and returns None, so the
    # guard's AssertionError never reaches the test.
    async def go():
        client = llm.VLLM()
        try:
            return await client.prefix_cache_stats()
        finally:
            await client.aclose()

    assert asyncio.run(go()) is None
    assert network_guard.attempts == ["GET http://vllm.test/metrics"]
    # The recorded attempt is what fails the test at teardown.
    with pytest.raises(pytest.fail.Exception, match="real network call"):
        network_guard.check()
    network_guard.attempts.clear()


def test_mock_transport_and_testclient_pass_the_guard():
    t = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    with httpx.Client(transport=t) as client:
        assert client.get("http://vllm.test/anything").json() == {"ok": True}

    async def go():
        async with httpx.AsyncClient(transport=t) as client:
            return (await client.get("http://vllm.test/anything")).json()

    assert asyncio.run(go()) == {"ok": True}

    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = fastapi.FastAPI()

    @app.get("/ping")
    def ping():
        return {"pong": True}

    assert TestClient(app).get("/ping").json() == {"pong": True}


# ---------------------------------------------------------------------------
# (c) the real llm.VLLM over a MockTransport
# ---------------------------------------------------------------------------


def test_vllm_streams_steps_over_a_mock_transport():
    router = Router().add("POST", "/v1/completions", sse_response(
        stream_chunk(["Hel"], [-0.1], [{"Hel": -0.1, "Hi": -2.5}]),
        stream_chunk(["lo"], [-0.2], [{"w": -3.0, "lo": -0.2, "p": None}],
                     finish="length"),
    ))

    async def go():
        v = make_mock_client(router)
        try:
            return v, [s async for s in v.stream("PROMPT", max_tokens=2)]
        finally:
            await v.aclose()

    v, steps = asyncio.run(go())
    assert [s.text for s in steps] == ["Hel", "lo"]
    assert [s.logprob for s in steps] == [-0.1, -0.2]
    assert steps[0].top == [("Hel", -0.1), ("Hi", -2.5)]
    assert steps[1].top == [("lo", -0.2), ("w", -3.0)]  # sorted, None dropped
    assert [s.finish for s in steps] == [None, "length"]
    assert v.last_finish == "length"
    assert v.usage_snapshot() == {"requests": 1, "prompt_tokens": 0,
                                  "completion_tokens": 2, "retries": 0}

    # Only the transport was swapped: base URL, headers and timeout are the
    # ones llm.VLLM builds from config.
    assert str(v._c.base_url) == "http://vllm.test/v1/"
    assert v._c.timeout == httpx.Timeout(config.TIMEOUT, connect=30.0)
    (req,) = router.requests
    assert req.url == "http://vllm.test/v1/completions"
    assert req.headers["authorization"] == "Bearer test-key"
    assert req.headers["user-agent"] == config.USER_AGENT
    assert req.json == {
        "model": "test-model", "prompt": "PROMPT", "max_tokens": 2,
        "temperature": 0.0, "logprobs": config.BRANCH_K, "stop": chatml.STOP,
        "stream": True,
    }
    router.assert_consumed()


def test_wire_helpers_round_trip_through_the_real_client():
    router = (
        Router()
        .add("POST", "/v1/completions",
             error_response(524),  # transient: retried, with zero backoff
             json_response(answer_probs_json([[0.9, 0.1], [0.0, 1.0]], ids=[1000, 1001])),
             sse_response(
                 stream_chunk([" a"], [-0.5], [{" a": -0.5, " b": -1.0}], index=0),
                 stream_chunk([" c"], [-0.25], [{" c": -0.25}], index=1, finish="stop"),
                 stream_chunk([" d"], [-0.5], [{" d": -0.5, " e": -0.9}], index=0,
                              finish="length"),
             ))
        .add("POST", "/tokenize", json_response(tokenize_json([11, 12])))
        .add("GET", "/metrics", httpx.Response(200, text=metrics_text(200, 50)))
    )

    async def go():
        v = make_mock_client(router)
        try:
            probs = await v.answer_probs(["a", "b"], ids=[1000, 1001])
            events = [e async for e in v.top_next(["p0", "p1"], k=2, max_tokens=4)]
            return v, probs, events, await v.tokenize("xy"), await v.prefix_cache_stats()
        finally:
            await v.aclose()

    v, probs, events, toks, hit = asyncio.run(go())
    assert probs == [pytest.approx([0.9, 0.1]), pytest.approx([0.0, 1.0])]
    assert v.usage["retries"] == 1
    assert events == [
        ("tok", 0, " a"), ("tok", 1, " c"), ("tok", 0, " d"),
        ("end", [(" a d", -1.0, [(" d", -0.5), (" e", -0.9)], False),
                 (" c", -0.25, [(" c", -0.25)], True)]),
    ]
    assert toks == [11, 12]
    assert hit == 0.25
    # /tokenize and /metrics live at the server root, not under /v1.
    assert [r.url for r in router.requests[-2:]] == [
        "http://vllm.test/tokenize", "http://vllm.test/metrics"]
    assert router.requests[1].json["logprob_token_ids"] == [1000, 1001]
    router.assert_consumed()


def test_a_stream_that_dies_after_a_token_is_not_replayed():
    router = Router().add("POST", "/v1/completions", httpx.Response(
        200, stream=BrokenStream(sse(stream_chunk(["Hi"], [-0.1]), done=False))))

    async def go():
        v = make_mock_client(router)
        got = []
        try:
            with pytest.raises(httpx.ReadTimeout):
                async for s in v.stream("P", max_tokens=5):
                    got.append(s.text)
            return v, got
        finally:
            await v.aclose()

    v, got = asyncio.run(go())
    assert got == ["Hi"]
    assert v.usage["retries"] == 0 and v.usage["requests"] == 1


# ---------------------------------------------------------------------------
# (d) the Engine on the engine-level fake
# ---------------------------------------------------------------------------

QUESTION = "What is 2 + 2?"


def _answer():
    """Five tokens with real entropy (two alternatives each)."""
    words = ["The", " answer", " is", " 4", "."]
    return [step(w, top=[(w, ln(0.6)), (w + "x", ln(0.4))]) for w in words]


def test_engine_completes_a_plain_turn_on_the_fake(cfg, fake_llm):
    # No opening thought, and an entropy floor nothing can reach: the gate
    # never fires, so the turn is one stream from start to finish.
    cfg(OPEN_THINK=False, ENTROPY_FLOOR=1e9, WINDOW_TOKENS=8)
    fake = fake_llm(streams=[Stream(_answer(), finish="stop")])

    events, conv = run_turn(fake, QUESTION)

    kinds = types(events)
    assert kinds == ["start"] + ["token"] * 5 + ["done"]
    assert [e["text"] for e in events if e["type"] == "token"] == [
        "The", " answer", " is", " 4", "."]
    assert events[1]["h"] == round(step("x", top=[("x", ln(.6)), ("y", ln(.4))]).entropy, 3)

    done = events[-1]
    assert done["text"] == "The answer is 4."
    stats = done["stats"]
    assert stats["blocks"] == 0
    assert stats["kept_tokens"] == 5 and stats["rewound_tokens"] == 0
    assert stats["requests"] == 1 and stats["completion_tokens"] == 5
    assert stats["degraded"] is None and stats["spills"] == 0
    assert stats["opening_thought"] is False
    assert stats["notes_in_context"] == 0

    # The one request is exactly the prompt the conversation renders.
    (call,) = fake.of("stream")
    assert call["prompt"] == Conversation().render(
        pending_user=QUESTION, think=False)
    assert call["max_tokens"] == 8
    assert fake.names() == ["stream"]
    assert [t.role for t in conv.turns] == ["user", "assistant"]
    assert conv.turns[1].text == "The answer is 4."
    assert conv.clock == 5
    fake.assert_consumed()


def test_a_long_answer_is_served_window_by_window(cfg, fake_llm):
    cfg(OPEN_THINK=False, ENTROPY_FLOOR=1e9, WINDOW_TOKENS=3)
    fake = fake_llm(streams=windows(quiet("One", " two", " three", " four", "."), 3))

    events, _ = run_turn(fake, "Count.")

    assert events[-1]["text"] == "One two three four."
    first, second = fake.of("stream")
    # The second window continues the first: same prefix, answer so far appended.
    assert second["prompt"] == first["prompt"] + "One two three"
    fake.assert_consumed()


def test_the_fake_ends_a_turn_its_script_does_not(cfg, fake_llm):
    cfg(OPEN_THINK=False, ENTROPY_FLOOR=1e9, WINDOW_TOKENS=2)
    fake = fake_llm(streams=[Stream(quiet("Still", " going"), finish="length")])
    with pytest.raises(AssertionError, match="script ran out"):
        run_turn(fake, "Go on.")
    assert fake.names() == ["stream", "stream"]


def test_a_scripted_stream_longer_than_the_window_is_a_script_error(cfg, fake_llm):
    cfg(OPEN_THINK=False, ENTROPY_FLOOR=1e9, WINDOW_TOKENS=2)
    fake = fake_llm(streams=[quiet("a", "b", "c")])
    with pytest.raises(AssertionError, match="max_tokens=2"):
        run_turn(fake, "Go.")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def test_prompt_readers_recover_axis_context_and_thought():
    ctx = "<|im_start|>system\nS<|im_end|>\nCTX"
    for name in prompts.GATE_AXES:
        p = prompts.gate_prompt(ctx, name)
        assert asked_axis(p) == ("gate", name)
        assert context_of(p) == ctx
        assert thought_of(p) is None
    for name in prompts.AXES:
        p = prompts.axis_prompt(ctx, "  I think.\n\nThen: done. ", name)
        assert asked_axis(p) == ("judge", name)
        assert thought_of(p) == "I think.\n\nThen: done."
    assert asked_axis(prompts.probe_prompt(ctx, prompts.GATE_PROBE)) == (None, None)


def test_step_and_level_helpers():
    assert lvl(0) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert lvl(2, n=3) == [0.0, 0.0, 1.0]
    for i in range(5):
        assert _expected(lvl(i), 5) == i / 4

    s = step("b", top=[("a", ln(0.3)), ("b", ln(0.7))])
    assert isinstance(s, llm.Step)
    assert [t for t, _ in s.top] == ["b", "a"]
    assert s.logprob == pytest.approx(ln(0.7))
    assert s.alternatives == ["b", "a"]
    assert quiet("x")[0].entropy == 0.0
    assert quiet("x")[0].alternatives == ["x"]

    w = windows(quiet("a", "b", "c"), 2, finish="stop")
    assert [(len(x.steps), x.finish) for x in w] == [(2, "length"), (1, "stop")]
