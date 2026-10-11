"""llm.VLLM at the wire level: the real client over httpx.MockTransport.

Everything here goes through the real request and response parsing in llm.py;
only the transport is swapped (fake_endpoint.make_mock_client). The expected
values were derived from llm.py and are pinned as current behaviour.
"""

import asyncio
import json
import math
import sys
import types

import httpx
import pytest

from adaptive_thinking import chatml, config, llm
from fake_endpoint import (
    BrokenStream,
    Router,
    answer_probs_json,
    completion_json,
    error_response,
    http_error,
    json_response,
    make_mock_client,
    metrics_text,
    sse,
    sse_response,
    stream_chunk,
    tokenize_json,
)

ln = math.log
COMPLETIONS = "/v1/completions"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def call(handler, fn):
    """Build a mocked llm.VLLM inside the event loop, run `fn(client)`, close
    the client. Returns (client, result)."""

    async def go():
        v = make_mock_client(handler)
        try:
            return v, await fn(v)
        finally:
            await v.aclose()

    return asyncio.run(go())


async def collect(agen):
    return [x async for x in agen]


@pytest.fixture
def backoff_log(monkeypatch):
    """Replace the sleep llm.py waits with between retries, and record each
    delay. Only llm's own reference to asyncio is swapped, so nothing else in
    the event loop is affected."""
    delays = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        delays.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(llm, "asyncio", types.SimpleNamespace(sleep=fake_sleep))
    return delays


def step(text, logprob, top=(), finish=None):
    return llm.Step(text=text, logprob=logprob, top=list(top), finish=finish)


# ---------------------------------------------------------------------------
# VLLM._steps: one streamed choice -> Steps
# ---------------------------------------------------------------------------


def test_steps_sorts_each_top_list_and_drops_none():
    choice = {
        "text": "Hello",
        "logprobs": {
            "tokens": ["Hel", "lo"],
            "token_logprobs": [-0.1, -0.2],
            "top_logprobs": [{"Hi": -2.5, "Hel": -0.1},
                             {"p": None, "lo": -0.2, "w": -3.0}],
        },
        "finish_reason": "length",
    }
    assert list(llm.VLLM._steps(choice)) == [
        step("Hel", -0.1, [("Hel", -0.1), ("Hi", -2.5)], None),
        step("lo", -0.2, [("lo", -0.2), ("w", -3.0)], "length"),
    ]


def test_steps_puts_finish_only_on_the_last_token():
    choice = {
        "text": "abc",
        "logprobs": {"tokens": ["a", "b", "c"], "token_logprobs": [-1.0, -2.0, -3.0],
                     "top_logprobs": [{"a": -1.0}, {"b": -2.0}, {"c": -3.0}]},
        "finish_reason": "stop",
    }
    assert [s.finish for s in llm.VLLM._steps(choice)] == [None, None, "stop"]


def test_steps_without_logprobs_is_one_step_of_the_whole_text():
    assert list(llm.VLLM._steps({"text": "Hi there", "finish_reason": "stop"})) == [
        step("Hi there", 0.0, [], "stop")]
    assert list(llm.VLLM._steps({"text": "x", "logprobs": None})) == [
        step("x", 0.0, [], None)]


def test_steps_of_an_empty_choice_is_nothing():
    assert list(llm.VLLM._steps({"text": "", "finish_reason": "stop"})) == []
    assert list(llm.VLLM._steps({"finish_reason": "length"})) == []


def test_steps_fills_missing_logprobs_and_top_lists():
    choice = {
        "text": "ab",
        "logprobs": {"tokens": ["a", "b"], "token_logprobs": [-0.5],
                     "top_logprobs": [None, {"b": -1.0}]},
    }
    assert list(llm.VLLM._steps(choice)) == [
        step("a", -0.5, [], None),
        step("b", 0.0, [("b", -1.0)], None),
    ]
    no_lps = {"text": "ab", "logprobs": {"tokens": ["a", "b"], "token_logprobs": None}}
    assert [(s.text, s.logprob, s.top) for s in llm.VLLM._steps(no_lps)] == [
        ("a", 0.0, []), ("b", 0.0, [])]


# ---------------------------------------------------------------------------
# Step: entropy, margin, alternatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top, expected", [
    ([], 0.0),
    # The unseen mass counts as one bucket: 0.5 seen + 0.5 unseen = ln 2.
    ([("a", ln(0.5))], ln(2)),
    ([("a", ln(0.5)), ("b", ln(0.5))], ln(2)),
    ([(t, ln(0.25)) for t in "abcd"], ln(4)),
    ([("a", 0.0)], 0.0),
])
def test_step_entropy_is_truncated_with_the_residual_as_one_bucket(top, expected):
    assert step("x", 0.0, top).entropy == pytest.approx(expected)


def test_step_entropy_is_a_lower_bound_on_the_true_entropy():
    # Five tokens seen at 0.1 each, the other 0.5 spread over many tokens in
    # truth; the truncated value treats it as a single bucket.
    s = step("a", ln(0.1), [(t, ln(0.1)) for t in "abcde"])
    truncated = -5 * 0.1 * ln(0.1) - 0.5 * ln(0.5)
    spread = -5 * 0.1 * ln(0.1) - 50 * 0.01 * ln(0.01)
    assert s.entropy == pytest.approx(truncated)
    assert s.entropy < spread


def test_step_probs_and_margin():
    s = step("a", ln(0.7), [("a", ln(0.7)), ("b", ln(0.2))])
    assert s.probs == pytest.approx([0.7, 0.2])
    assert s.margin == pytest.approx(0.5)
    assert step("a", 0.0, [("a", 0.0)]).margin == 1.0
    assert step("a", 0.0, []).margin == 1.0


def test_step_alternatives_are_deduplicated_and_skip_empty_tokens():
    s = step(" a", -0.1, [(" a", -0.1), (" a", -0.2), ("", -0.3), ("b", -0.4)])
    assert s.alternatives == [" a", "b"]
    assert step("x", 0.0, []).alternatives == []


# ---------------------------------------------------------------------------
# transient(): what is worth retrying
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 524])
def test_gateway_and_overload_statuses_are_transient(status):
    assert llm.transient(http_error(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_transient(status):
    assert llm.transient(http_error(status)) is False


def test_retry_status_set_is_exactly_the_documented_one():
    assert llm.RETRY_STATUS == {429, 500, 502, 503, 504, 524}


@pytest.mark.parametrize("exc", [
    httpx.ConnectError("refused"),
    httpx.ConnectTimeout("slow connect"),
    httpx.ReadTimeout("slow read"),
    httpx.WriteTimeout("slow write"),
    httpx.PoolTimeout("no connection"),
    httpx.RemoteProtocolError("peer closed"),
])
def test_connection_failures_are_transient(exc):
    assert llm.transient(exc) is True


@pytest.mark.parametrize("exc", [
    # Current behaviour: a read error is not in the retry list.
    httpx.ReadError("reset"),
    httpx.WriteError("broken pipe"),
    ValueError("bad json"),
    RuntimeError("anything else"),
])
def test_other_errors_are_not_transient(exc):
    assert llm.transient(exc) is False


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("knob", ["BASE_URL", "MODEL"])
def test_the_client_refuses_to_start_without_an_endpoint(cfg, knob):
    cfg(**{knob: ""})
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL and OPENAI_MODEL"):
        llm.VLLM()


def test_a_new_client_has_zeroed_usage_and_no_finish():
    router = Router()

    async def snapshot(v):
        return v.usage_snapshot()

    v, snap = call(router, snapshot)
    assert snap == {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "retries": 0}
    assert v.last_finish is None
    # A snapshot is a copy, not the live dict.
    snap["requests"] = 99
    assert v.usage["requests"] == 0
    assert router.requests == []


def test_identity_headers_are_read_from_config_at_call_time(cfg):
    assert llm.identity_headers() == {"Authorization": "Bearer test-key",
                                      "User-Agent": config.USER_AGENT}
    cfg(API_KEY="other-key", USER_AGENT="probe/2")
    assert list(llm.identity_headers().items()) == [("Authorization", "Bearer other-key"),
                                                    ("User-Agent", "probe/2")]


def test_the_client_sends_the_identity_headers_and_a_json_content_type(cfg):
    cfg(API_KEY="other-key", USER_AGENT="probe/2")
    router = Router()

    async def headers(v):
        return v._c.headers

    _, h = call(router, headers)
    assert (h["authorization"], h["user-agent"], h["content-type"]) == (
        "Bearer other-key", "probe/2", "application/json")


# ---------------------------------------------------------------------------
# _post (complete, answer_probs): retries
# ---------------------------------------------------------------------------


def test_post_retries_a_503_then_succeeds(cfg, backoff_log):
    cfg(LLM_RETRIES=2, LLM_RETRY_BACKOFF=1.5)
    router = Router().add("POST", COMPLETIONS,
                          error_response(503),
                          json_response(completion_json(["x", "y"])))

    v, out = call(router, lambda v: v.complete(["p0", "p1"], max_tokens=4))

    assert out == ["x", "y"]
    assert len(router.requests) == 2
    assert router.requests[0].json == router.requests[1].json  # replayed in full
    assert v.usage["retries"] == 1
    # complete() counts one request per call, however many attempts it took.
    assert v.usage["requests"] == 1
    assert backoff_log == [1.5]
    router.assert_consumed()


def test_post_backoff_grows_linearly_with_the_attempt(cfg, backoff_log):
    cfg(LLM_RETRIES=3, LLM_RETRY_BACKOFF=2.0)
    router = Router().add("POST", COMPLETIONS,
                          error_response(502), error_response(524),
                          error_response(429),
                          json_response(completion_json(["ok"])))

    v, out = call(router, lambda v: v.complete(["p"], max_tokens=1))

    assert out == ["ok"]
    assert backoff_log == [2.0, 4.0, 6.0]
    assert v.usage["retries"] == 3


def test_post_gives_up_after_llm_retries(cfg, backoff_log):
    cfg(LLM_RETRIES=2)
    router = Router().add("POST", COMPLETIONS, *[error_response(503) for _ in range(4)])

    async def go(v):
        with pytest.raises(httpx.HTTPStatusError) as err:
            await v.complete(["p"], max_tokens=1)
        return err.value

    v, err = call(router, go)
    assert err.response.status_code == 503
    assert len(router.requests) == 3  # the first attempt plus two retries
    assert v.usage["retries"] == 2
    assert v.usage["requests"] == 0  # counted only on success
    assert len(backoff_log) == 2


def test_post_with_zero_retries_fails_on_the_first_503(cfg, backoff_log):
    cfg(LLM_RETRIES=0)
    router = Router().add("POST", COMPLETIONS, error_response(503))

    async def go(v):
        with pytest.raises(httpx.HTTPStatusError):
            await v.complete(["p"], max_tokens=1)

    v, _ = call(router, go)
    assert len(router.requests) == 1
    assert v.usage["retries"] == 0
    assert backoff_log == []


def test_post_does_not_retry_a_400(cfg, backoff_log):
    cfg(LLM_RETRIES=5)
    router = Router().add("POST", COMPLETIONS, error_response(400),
                          json_response(completion_json(["never"])))

    async def go(v):
        with pytest.raises(httpx.HTTPStatusError) as err:
            await v.complete(["p"], max_tokens=1)
        return err.value

    v, err = call(router, go)
    assert err.response.status_code == 400
    assert len(router.requests) == 1
    assert v.usage["retries"] == 0
    assert backoff_log == []


def test_post_retries_a_connect_error_but_not_a_read_error(cfg, backoff_log):
    cfg(LLM_RETRIES=2)
    router = Router().add("POST", COMPLETIONS,
                          httpx.ConnectError("refused"),
                          json_response(completion_json(["ok"])))
    v, out = call(router, lambda v: v.complete(["p"], max_tokens=1))
    assert out == ["ok"]
    assert v.usage["retries"] == 1

    router = Router().add("POST", COMPLETIONS, httpx.ReadError("reset"),
                          json_response(completion_json(["never"])))

    async def go(v):
        with pytest.raises(httpx.ReadError):
            await v.complete(["p"], max_tokens=1)

    v, _ = call(router, go)
    assert len(router.requests) == 1
    assert v.usage["retries"] == 0


# ---------------------------------------------------------------------------
# complete
# ---------------------------------------------------------------------------


def test_complete_sends_a_list_and_orders_results_by_index():
    body = {
        "choices": [{"index": 2, "text": "c"}, {"index": 0, "text": "a"},
                    {"index": 1, "text": None}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 3},
    }
    router = Router().add("POST", COMPLETIONS, json_response(body))

    v, out = call(router, lambda v: v.complete(("p0", "p1", "p2"), max_tokens=7,
                                                temperature=0.5))

    assert out == ["a", "", "c"]
    (req,) = router.requests
    assert req.json == {"model": "test-model", "prompt": ["p0", "p1", "p2"],
                        "max_tokens": 7, "temperature": 0.5, "stop": chatml.STOP}
    assert v.usage_snapshot() == {"requests": 1, "prompt_tokens": 30,
                                  "completion_tokens": 3, "retries": 0}


def test_complete_passes_a_stop_override():
    router = Router().add("POST", COMPLETIONS, json_response(completion_json(["x"])))
    call(router, lambda v: v.complete(["p"], max_tokens=1, stop=[chatml.IM_END]))
    assert router.requests[0].json["stop"] == [chatml.IM_END]


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


def test_stream_payload_takes_the_overrides():
    router = Router().add("POST", COMPLETIONS, sse_response(
        stream_chunk(["a"], [-0.1], finish="stop")))

    call(router, lambda v: collect(v.stream("P", max_tokens=9, top_k=3,
                                            temperature=0.7, stop=[chatml.IM_END])))

    assert router.requests[0].json == {
        "model": "test-model", "prompt": "P", "max_tokens": 9, "temperature": 0.7,
        "logprobs": 3, "stop": [chatml.IM_END], "stream": True,
    }


def test_stream_reads_the_finish_from_a_chunk_that_carries_no_token():
    router = Router().add("POST", COMPLETIONS, sse_response(
        stream_chunk(["Hi"], [-0.1]),
        # vLLM's last chunk: no text, just the finish reason.
        {"choices": [{"index": 0, "text": "", "logprobs": None,
                      "finish_reason": "stop"}]},
    ))

    v, steps = call(router, lambda v: collect(v.stream("P", max_tokens=5)))

    assert [(s.text, s.finish) for s in steps] == [("Hi", None)]
    assert v.last_finish == "stop"
    assert v.usage["completion_tokens"] == 1


def test_stream_ignores_non_data_lines_and_stops_at_done():
    body = (b": keep-alive\n\n"
            + sse(stream_chunk(["a"], [-0.1]), done=False)
            + b"event: ping\n\n"
            + sse(stream_chunk(["b"], [-0.2], finish="length"))
            + sse(stream_chunk(["never"], [-0.3]), done=False))
    router = Router().add("POST", COMPLETIONS, httpx.Response(
        200, content=body, headers={"content-type": "text/event-stream"}))

    v, steps = call(router, lambda v: collect(v.stream("P", max_tokens=5)))

    assert [s.text for s in steps] == ["a", "b"]
    assert v.last_finish == "length"


def test_stream_resets_last_finish_for_each_request():
    router = Router().add(
        "POST", COMPLETIONS,
        sse_response(stream_chunk(["a"], [-0.1], finish="length")),
        sse_response(stream_chunk(["b"], [-0.1])),  # no finish reason at all
    )

    async def go(v):
        await collect(v.stream("P", max_tokens=1))
        first = v.last_finish
        await collect(v.stream("P", max_tokens=1))
        return first, v.last_finish

    _, (first, second) = call(router, go)
    assert (first, second) == ("length", None)


def test_stream_retries_a_503_before_the_first_token(cfg, backoff_log):
    cfg(LLM_RETRIES=2, LLM_RETRY_BACKOFF=1.0)
    router = Router().add("POST", COMPLETIONS, error_response(503), sse_response(
        stream_chunk(["Hi"], [-0.1]), stream_chunk(["!"], [-0.2], finish="stop")))

    v, steps = call(router, lambda v: collect(v.stream("P", max_tokens=5)))

    assert [s.text for s in steps] == ["Hi", "!"]
    assert len(router.requests) == 2
    # Streams count every attempt, unlike _post.
    assert v.usage_snapshot() == {"requests": 2, "prompt_tokens": 0,
                                  "completion_tokens": 2, "retries": 1}
    assert backoff_log == [1.0]


def test_stream_retries_a_timeout_before_the_first_token(cfg):
    cfg(LLM_RETRIES=1)
    router = Router().add(
        "POST", COMPLETIONS,
        httpx.Response(200, stream=BrokenStream(exc=httpx.ReadTimeout("slow"))),
        sse_response(stream_chunk(["ok"], [-0.1], finish="stop")),
    )

    v, steps = call(router, lambda v: collect(v.stream("P", max_tokens=5)))

    assert [s.text for s in steps] == ["ok"]
    assert v.usage["retries"] == 1


def test_stream_does_not_retry_a_400(cfg):
    cfg(LLM_RETRIES=3)
    router = Router().add("POST", COMPLETIONS, error_response(400),
                          sse_response(stream_chunk(["never"], [-0.1])))

    async def go(v):
        with pytest.raises(httpx.HTTPStatusError) as err:
            await collect(v.stream("P", max_tokens=5))
        return err.value

    v, err = call(router, go)
    assert err.response.status_code == 400
    assert len(router.requests) == 1
    assert v.usage["retries"] == 0


def test_stream_that_dies_after_a_token_is_not_replayed_even_with_retries_left(cfg):
    cfg(LLM_RETRIES=5)
    broken = BrokenStream(sse(stream_chunk(["Hel"], [-0.1]), done=False),
                          exc=httpx.RemoteProtocolError("peer closed"))
    router = Router().add("POST", COMPLETIONS, httpx.Response(200, stream=broken),
                          sse_response(stream_chunk(["never"], [-0.1])))
    got = []

    async def go(v):
        with pytest.raises(httpx.RemoteProtocolError):
            async for s in v.stream("P", max_tokens=5):
                got.append(s.text)

    v, _ = call(router, go)
    # The error is transient and retries remain, but a token was already
    # handed out, so a replay would write it twice.
    assert llm.transient(httpx.RemoteProtocolError("x"))
    assert got == ["Hel"]
    assert len(router.requests) == 1
    assert v.usage["retries"] == 0


# ---------------------------------------------------------------------------
# top_next
# ---------------------------------------------------------------------------


def test_top_next_defaults_to_the_chunk_size_and_branch_k(cfg):
    cfg(THINK_CHUNK=37, THINK_BRANCH_K=3)
    router = Router().add("POST", COMPLETIONS, sse_response(
        stream_chunk(["a"], [-0.1], finish="stop")))

    _, events = call(router, lambda v: collect(v.top_next(("p",))))

    assert router.requests[0].json == {
        "model": "test-model", "prompt": ["p"], "max_tokens": 37, "temperature": 0,
        "logprobs": 3, "stop": chatml.STOP, "stream": True,
    }
    assert events[-1] == ("end", [("a", -0.1, [("a", -0.1)], True)])


def test_top_next_keeps_the_last_distribution_truncated_to_k():
    router = Router().add("POST", COMPLETIONS, sse_response(
        stream_chunk(["x"], [-0.5], [{"x": -0.5, "y": -1.0, "z": -2.0}]),
        # A None logprob is not summed; its distribution replaces the last.
        stream_chunk(["r"], [None], [{"s": -0.9, "r": -0.3, "t": -4.0}]),
        # An empty top dict does not overwrite the last real one.
        stream_chunk(["q"], [-0.25], [{}]),
    ))

    v, events = call(router, lambda v: collect(v.top_next(["p"], k=2, max_tokens=8)))

    assert events == [
        ("tok", 0, "x"), ("tok", 0, "r"), ("tok", 0, "q"),
        ("end", [("xrq", -0.75, [("r", -0.3), ("s", -0.9)], False)]),
    ]
    assert v.usage["completion_tokens"] == 2  # one per non-None logprob
    assert v.usage["requests"] == 1


def test_top_next_marks_only_a_stop_finish_as_stopped():
    router = Router().add("POST", COMPLETIONS, sse_response(
        stream_chunk(["a"], [-1.0], index=0, finish="length"),
        stream_chunk(["b"], [-1.0], index=1, finish="stop"),
        stream_chunk([], [], index=2, finish="length"),
    ))

    _, events = call(router, lambda v: collect(
        v.top_next(["p0", "p1", "p2"], k=5, max_tokens=4)))

    # A prompt that produced nothing yields no "tok" event and ends empty.
    assert events == [
        ("tok", 0, "a"), ("tok", 1, "b"),
        ("end", [("a", -1.0, [("a", -1.0)], False),
                 ("b", -1.0, [("b", -1.0)], True),
                 ("", 0.0, [], False)]),
    ]


def test_top_next_reads_prompt_tokens_from_a_usage_chunk():
    router = Router().add("POST", COMPLETIONS, sse_response(
        stream_chunk(["a"], [-1.0], finish="stop"),
        {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 0}},
    ))

    v, _ = call(router, lambda v: collect(v.top_next(["p"], k=1, max_tokens=4)))

    assert v.usage["prompt_tokens"] == 120


def test_top_next_retries_before_any_token_and_restarts_the_chunk(cfg):
    cfg(LLM_RETRIES=1)
    router = Router().add(
        "POST", COMPLETIONS,
        error_response(524),
        sse_response(stream_chunk(["a"], [-1.0], finish="stop")),
    )

    v, events = call(router, lambda v: collect(v.top_next(["p"], k=1, max_tokens=4)))

    assert events == [("tok", 0, "a"), ("end", [("a", -1.0, [("a", -1.0)], True)])]
    assert v.usage["retries"] == 1
    assert v.usage["requests"] == 2


def test_top_next_does_not_replay_a_chunk_that_already_streamed(cfg):
    cfg(LLM_RETRIES=3)
    broken = BrokenStream(sse(stream_chunk(["a"], [-1.0]), done=False))
    router = Router().add("POST", COMPLETIONS, httpx.Response(200, stream=broken),
                          sse_response(stream_chunk(["never"], [-1.0])))
    got = []

    async def go(v):
        with pytest.raises(httpx.ReadTimeout):
            async for ev in v.top_next(["p"], k=1, max_tokens=4):
                got.append(ev)

    v, _ = call(router, go)
    assert got == [("tok", 0, "a")]
    assert len(router.requests) == 1
    assert v.usage["retries"] == 0


# ---------------------------------------------------------------------------
# answer_probs / p_yes / answer_ids
# ---------------------------------------------------------------------------


def test_answer_probs_renormalises_over_the_ids_asked_for():
    ids = [1000, 1001]
    tl = {"token_id:1000": ln(0.6), "token_id:1001": ln(0.2)}
    body = {"choices": [{"index": 0, "logprobs": {"top_logprobs": [tl]}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1}}
    router = Router().add("POST", COMPLETIONS, json_response(body))

    v, rows = call(router, lambda v: v.answer_probs(["q"], ids=ids))

    assert rows == [pytest.approx([0.75, 0.25])]
    (req,) = router.requests
    assert req.json == {
        "model": "test-model", "prompt": ["q"], "max_tokens": 1, "temperature": 0,
        "logprobs": 0, "logprob_token_ids": [1000, 1001],
        "return_tokens_as_token_ids": True,
    }
    assert v.usage_snapshot() == {"requests": 1, "prompt_tokens": 12,
                                  "completion_tokens": 1, "retries": 0}


def test_answer_probs_counts_a_missing_id_as_zero_and_orders_by_index():
    ids = [7, 8, 9]
    body = {"choices": [
        {"index": 1, "logprobs": {"top_logprobs": [{"token_id:9": 0.0}]}},
        {"index": 0, "logprobs": {"top_logprobs": [
            {"token_id:7": ln(0.3), "token_id:8": ln(0.1), "token_id:555": ln(0.5)}]}},
    ]}
    router = Router().add("POST", COMPLETIONS, json_response(body))

    _, rows = call(router, lambda v: v.answer_probs(["a", "b"], ids=ids))

    # An id outside `ids` is ignored; mass elsewhere is not part of the answer.
    assert rows[0] == pytest.approx([0.75, 0.25, 0.0])
    assert rows[1] == pytest.approx([0.0, 0.0, 1.0])


def test_answer_probs_refuses_a_response_with_none_of_the_ids():
    body = {"choices": [{"index": 0, "logprobs": {"top_logprobs": [{"token_id:1": -0.1}]}}]}
    router = Router().add("POST", COMPLETIONS, json_response(body))

    async def go(v):
        with pytest.raises(RuntimeError, match="logprob_token_ids returned nothing"):
            await v.answer_probs(["q"], ids=[1000, 1001])

    call(router, go)


@pytest.mark.parametrize("choice", [
    {"index": 0, "logprobs": None},
    {"index": 0, "logprobs": {"top_logprobs": None}},
    {"index": 0, "logprobs": {"top_logprobs": [None]}},
])
def test_answer_probs_treats_a_missing_top_list_as_nothing_returned(choice):
    router = Router().add("POST", COMPLETIONS, json_response({"choices": [choice]}))

    async def go(v):
        with pytest.raises(RuntimeError, match="silently ignoring"):
            await v.answer_probs(["q"], ids=[1])

    call(router, go)


def test_answer_probs_matches_the_shared_wire_helper():
    ids = [1000, 1001, 1002]
    rows = [[0.5, 0.25, 0.25], [0.0, 0.0, 1.0]]
    router = Router().add("POST", COMPLETIONS, json_response(answer_probs_json(rows, ids)))
    _, got = call(router, lambda v: v.answer_probs(["a", "b"], ids=ids))
    assert got == [pytest.approx(r) for r in rows]


def _tokenizer(table):
    """A /tokenize answer that looks each prompt up in `table`."""

    def answer(request):
        prompt = json.loads(request.content)["prompt"]
        return json_response(tokenize_json(table[prompt]))

    return answer


def test_answer_ids_takes_the_first_single_token_spelling_and_caches_it():
    table = {"Yes": [9001], "No": [44, 45], " No": [9002]}
    router = Router().add("POST", "/tokenize", *[_tokenizer(table)] * 3)

    async def go(v):
        first = await v.answer_ids()
        again = await v.answer_ids()
        return first, again

    _, (first, again) = call(router, go)

    assert first == again == (9001, 9002)
    assert [r.json["prompt"] for r in router.requests] == ["Yes", "No", " No"]
    router.assert_consumed()


def test_answer_ids_fails_loudly_when_no_spelling_is_one_token():
    table = {"Yes": [1], "No": [2, 3], " No": [4, 5]}
    router = Router().add("POST", "/tokenize", *[_tokenizer(table)] * 3)

    async def go(v):
        with pytest.raises(RuntimeError, match="'No' is not a single token"):
            await v.answer_ids()

    call(router, go)


def test_p_yes_is_the_first_column_over_the_resolved_yes_no_ids():
    table = {"Yes": [9001], "No": [9002]}
    router = (Router()
              .add("POST", "/tokenize", *[_tokenizer(table)] * 2)
              .add("POST", COMPLETIONS, json_response(
                  answer_probs_json([[0.8, 0.2], [0.1, 0.3]], [9001, 9002]))))

    _, ps = call(router, lambda v: v.p_yes(["a", "b"]))

    assert ps == pytest.approx([0.8, 0.25])
    assert router.requests[-1].json["logprob_token_ids"] == [9001, 9002]


# ---------------------------------------------------------------------------
# tokenize, labels, prefix_cache_stats: the server ROOT, not /v1
# ---------------------------------------------------------------------------


def test_tokenize_posts_to_the_server_root():
    router = Router().add("POST", "/tokenize", json_response(tokenize_json([5, 6, 7])))

    _, toks = call(router, lambda v: v.tokenize("abc"))

    assert toks == [5, 6, 7]
    (req,) = router.requests
    assert req.url == "http://vllm.test/tokenize"
    assert req.json == {"model": "test-model", "prompt": "abc",
                        "add_special_tokens": False}


def test_tokenize_is_not_retried():
    router = Router().add("POST", "/tokenize", error_response(503))

    async def go(v):
        with pytest.raises(httpx.HTTPStatusError):
            await v.tokenize("abc")

    v, _ = call(router, go)
    assert len(router.requests) == 1


def _letters(multi=()):
    """A /tokenize handler for the label fallback: every capital letter is one
    token (id 100 + position) except those in `multi`, which are two."""
    seen = []

    def handler(request):
        assert request.url.path == "/tokenize", request.url
        ch = json.loads(request.content)["prompt"]
        seen.append(ch)
        ids = [1, 2] if ch in multi else [100 + ord(ch) - ord("A")]
        return json_response(tokenize_json(ids))

    handler.seen = seen
    return handler


def test_labels_fall_back_to_single_token_capital_letters():
    handler = _letters(multi={"B"})

    async def go(v):
        three = await v.labels(3)
        cached = await v.labels(5)
        return three, cached

    _, (three, cached) = call(handler, go)

    # Every letter is verified through /tokenize, and a two-token letter is
    # skipped.
    assert three == [("A", 100), ("C", 102), ("D", 103)]
    assert cached == [("A", 100), ("C", 102), ("D", 103), ("E", 104), ("F", 105)]
    assert handler.seen == [chr(65 + i) for i in range(26)]  # read once, then cached


def test_labels_stop_reading_once_26_are_found():
    handler = _letters()
    _, got = call(handler, lambda v: v.labels(2))
    assert got == [("A", 100), ("B", 101)]
    assert len(handler.seen) == 26


def test_labels_fail_when_too_few_letters_are_single_tokens():
    handler = _letters(multi={"Q"})

    async def go(v):
        with pytest.raises(RuntimeError, match="only 25 single-token labels available, need 26"):
            await v.labels(26)

    call(handler, go)


def _optional_label_packages(monkeypatch, load_fails=False):
    """Stand-ins for the label package and for transformers that llm.labels()
    used to try before /tokenize. The suite blocks the real import
    (tests/conftest.py); these replace the block for one test, so the test
    can show that labels() no longer touches either. Returns the list every
    call is recorded in."""
    calls = []

    class Label:
        def __init__(self, text, token_id):
            self.text, self.token_id = text, token_id

    def make_labels(tokenizer, n):
        calls.append(("make_labels", tokenizer, n))
        return [Label(f"L{i}", 500 + i) for i in range(n)]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            calls.append(("from_pretrained", name))
            if load_fails:
                raise OSError("tokenizer not available offline")
            return "tokenizer"

    core = types.ModuleType("jevfire.core")
    core.make_labels = make_labels
    package = types.ModuleType("jevfire")
    package.core = core
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "jevfire", package)
    monkeypatch.setitem(sys.modules, "jevfire.core", core)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


def test_labels_come_from_tokenize_even_when_a_label_package_is_importable(monkeypatch):
    calls = _optional_label_packages(monkeypatch)
    handler = _letters()

    async def go(v):
        three, all26 = await v.labels(3), await v.labels(26)
        with pytest.raises(RuntimeError, match="only 26 single-token labels available, need 30"):
            await v.labels(30)
        return three, all26

    _, (three, all26) = call(handler, go)

    # The letters are verified through the server's /tokenize and cached;
    # asking for more than the cache holds reads them again, and there are
    # only 26. The importable package and tokenizer are never touched.
    assert three == [("A", 100), ("B", 101), ("C", 102)]
    assert all26 == [(chr(65 + i), 100 + i) for i in range(26)]
    assert handler.seen == [chr(65 + i) for i in range(26)] * 2
    assert calls == []


def test_labels_never_load_a_local_tokenizer(monkeypatch):
    calls = _optional_label_packages(monkeypatch, load_fails=True)
    handler = _letters()

    _, got = call(handler, lambda v: v.labels(2))

    assert got == [("A", 100), ("B", 101)]
    assert calls == []
    assert len(handler.seen) == 26
    # Nothing in the client imports a label package or a local tokenizer.
    import ast
    from pathlib import Path

    tree = ast.parse(Path(llm.__file__).read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)}
    assert imported & {"jevfire", "transformers"} == set()


def test_prefix_cache_stats_reads_the_hit_rate_from_the_server_root():
    router = Router().add("GET", "/metrics", httpx.Response(200, text=metrics_text(300, 100)))

    _, hit = call(router, lambda v: v.prefix_cache_stats())

    assert hit == 0.3333
    assert router.requests[0].url == "http://vllm.test/metrics"


def test_prefix_cache_stats_ignores_other_metrics_and_comments():
    text = (
        "# HELP vllm:prefix_cache_queries_total Prefix cache queries.\n"
        "vllm:num_requests_waiting{model_name=\"m\"} 3.0\n"
        + metrics_text(200, 50)
        + "vllm:gpu_prefix_cache_hits_total{model_name=\"m\"} 999.0\n"
    )
    router = Router().add("GET", "/metrics", httpx.Response(200, text=text))
    _, hit = call(router, lambda v: v.prefix_cache_stats())
    assert hit == 0.25


@pytest.mark.parametrize("body", [
    "",
    'vllm:prefix_cache_queries_total{model_name="m"} 200.0\n',  # no hits counter
    'vllm:prefix_cache_hits_total{model_name="m"} 50.0\n',      # no queries counter
    metrics_text(0, 0),                                         # nothing queried yet
    # Current behaviour: the pattern requires a label set in braces.
    "vllm:prefix_cache_queries_total 200.0\nvllm:prefix_cache_hits_total 50.0\n",
])
def test_prefix_cache_stats_is_none_without_both_counters(body):
    router = Router().add("GET", "/metrics", httpx.Response(200, text=body))
    _, hit = call(router, lambda v: v.prefix_cache_stats())
    assert hit is None


@pytest.mark.parametrize("answer", [error_response(404), httpx.ConnectError("down")])
def test_prefix_cache_stats_is_none_when_the_endpoint_fails(answer):
    router = Router().add("GET", "/metrics", answer)
    _, hit = call(router, lambda v: v.prefix_cache_stats())
    assert hit is None
