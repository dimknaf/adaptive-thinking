"""An offline stand-in for the vLLM endpoint, at two levels, plus small helpers.

ENGINE LEVEL: FakeVLLM
    Implements exactly the members the engine uses (every `llm.` call in
    engine.py, gate.py, tree.py and scoring.py) and nothing else, so a call
    the engine starts making later fails loudly with AttributeError instead
    of being answered by accident:

        usage_snapshot()      stream(prompt, max_tokens, ...)   last_finish
        top_next(prompts, k, max_tokens)    labels(n)    answer_probs(prompts, ids)
        p_yes(prompts)        tokenize(text)     prefix_cache_stats()    aclose()

    Everything it returns is scripted, and every call is recorded in order.
    The engine has no caps by design (a block ends when its beams close it, an
    answer when the model ends its turn), so the FAKE is what ends things:
    when a script runs out it raises AssertionError. The engine never swallows
    that, because it is not a transient endpoint error.

WIRE LEVEL: for testing llm.VLLM itself over httpx.MockTransport
    sse(), stream_chunk(), sse_response(), json_response(), completion_json(),
    answer_probs_json(), tokenize_json(), metrics_text(), http_error(),
    error_response(), BrokenStream, Router, mock_transport(), make_mock_client().

SMALL HELPERS
    step(), quiet(), Stream, windows(), openings(), chunk(), closed(), lvl(),
    uniform(), run_turn(), types(), of_type(), user_body(), context_of(),
    asked_axis(), thought_of().

This module must be imported after tests/conftest.py has neutralised .env,
which pytest guarantees.
"""

import asyncio
import contextlib
import json
import math
from dataclasses import dataclass, field
from unittest import mock

import httpx

from adaptive_thinking import chatml, config, llm, prompts
from adaptive_thinking.engine import Engine
from adaptive_thinking.state import Conversation

ln = math.log

# Token ids the fake hands out. labels(n) returns (letter, LABEL_ID + i); the
# Yes/No pair is only used when answer_probs is called without ids.
LABEL_ID = 1000
YES_ID, NO_ID = 9001, 9002


# ---------------------------------------------------------------------------
# Steps, streams and chunks
# ---------------------------------------------------------------------------


def _vllm_top(pairs, k=None):
    """A top-logprobs list as the real client builds it: best first, None
    values dropped, truncated to k. vLLM reports top_logprobs as a JSON object,
    so the same token can never appear twice; a script that says otherwise is
    describing something the endpoint cannot send."""
    if isinstance(pairs, dict):
        pairs = list(pairs.items())
    pairs = [(t, float(v)) for t, v in (pairs or []) if v is not None]
    toks = [t for t, _ in pairs]
    if len(toks) != len(set(toks)):
        raise AssertionError(
            f"top-logprobs list repeats a token {toks!r}; vLLM sends a dict, "
            "so every token appears at most once"
        )
    out = sorted(pairs, key=lambda kv: kv[1], reverse=True)
    return out if k is None else out[:k]


def step(text, top=None, logprob=None, finish=None):
    """A real llm.Step. `top` is [(token, logprob), ...] in any order; it is
    sorted best first as the client does. Without `top` the token is the only
    alternative (entropy 0). `logprob` defaults to the token's own entry in
    `top`, else the best entry."""
    if top is None:
        top = [(text, 0.0 if logprob is None else float(logprob))]
    top = sorted(((t, float(v)) for t, v in top), key=lambda kv: kv[1], reverse=True)
    if logprob is None:
        logprob = next((v for t, v in top if t == text), top[0][1] if top else 0.0)
    return llm.Step(text=text, logprob=float(logprob), top=top, finish=finish)


def quiet(*texts):
    """Steps the gate can never pick: one alternative each, entropy 0."""
    return [step(t) for t in texts]


@dataclass
class Stream:
    """One scripted stream() call.

    `steps` are yielded in order; `finish` becomes last_finish ("stop" = the
    model ended its turn, "length" = the window ran out). If `error` is set it
    is raised AFTER the steps, which is how a connection that dies mid-answer
    looks to the engine. A scripted stream longer than the max_tokens the
    engine asks for is a script error; split long answers with windows().
    """

    steps: list
    finish: str = "stop"
    error: BaseException = None


def windows(steps, size, finish="stop"):
    """Split one answer into the per-window Stream calls the engine will make
    when nothing interrupts it: every window but the last ends on "length"."""
    steps = list(steps)
    cut = [steps[i:i + size] for i in range(0, len(steps), size)] or [[]]
    return [Stream(c, "length") for c in cut[:-1]] + [Stream(cut[-1], finish)]


def openings(*tokens, gap=0.5):
    """A fan-out read: the given first tokens with descending logprobs."""
    return [(t, -gap * (i + 1)) for i, t in enumerate(tokens)]


def chunk(text, lp=-1.0, top=(), stopped=False):
    """One beam's result from a chunk request: (text, logprob, top, stopped).
    `top` is the distribution AT the chunk's last token."""
    return (text, float(lp), list(top), bool(stopped))


def closed(text, lp=-1.0):
    """A chunk in which the thought closed itself (finish_reason "stop")."""
    return (text, float(lp), [], True)


def lvl(i, n=5):
    """A levelled answer that is certainly level i (0 = worst): one-hot."""
    return [1.0 if j == i else 0.0 for j in range(n)]


def uniform(prompt):
    """The default score: every level equally likely (expected value 0.5)."""
    return [1.0] * 5


# ---------------------------------------------------------------------------
# Reading the engine's prompts, so a score() can answer per axis/per thought
# ---------------------------------------------------------------------------

_USER = chatml.IM_START + "user\n"
_REASONING = "Here is a piece of reasoning:\n\n"
_ONE_LETTER = "\n\nAnswer with one letter."


def user_body(prompt):
    """The body of the last user turn in a ChatML prompt ("" if none)."""
    i = prompt.rfind(_USER)
    if i == -1:
        return ""
    j = prompt.find(chatml.IM_END, i)
    return prompt[i + len(_USER): j if j != -1 else len(prompt)]


def context_of(prompt):
    """What a closed question was asked ABOUT: the prompt before the question
    turn (for the gate, the full prompt up to that answer position)."""
    i = prompt.rfind(chatml.IM_END + "\n" + _USER)
    return prompt if i == -1 else prompt[:i]


def _axis_suffix(a):
    return "\n\n" + a["question"] + "\n\n" + prompts._levels(a) + _ONE_LETTER


def asked_axis(prompt):
    """("gate", axis) for a gate prompt, ("judge", axis) for a thought-judging
    prompt, (None, None) for anything else (e.g. the Yes/No probe)."""
    body = user_body(prompt)
    for name, a in prompts.GATE_AXES.items():
        if body == a["question"] + "\n\n" + prompts._levels(a) + _ONE_LETTER:
            return "gate", name
    if body.startswith(_REASONING):
        for name, a in prompts.AXES.items():
            if body.endswith(_axis_suffix(a)):
                return "judge", name
    return None, None


def thought_of(prompt):
    """The thought a judge prompt quotes (already stripped), else None."""
    family, name = asked_axis(prompt)
    if family != "judge":
        return None
    body = user_body(prompt)
    return body[len(_REASONING): len(body) - len(_axis_suffix(prompts.AXES[name]))]


# ---------------------------------------------------------------------------
# Engine-level fake
# ---------------------------------------------------------------------------


@dataclass
class Call:
    """One recorded call. `name` is the method, except that top_next is
    recorded as "fanout" (max_tokens=1) or "chunk"; `kw` holds its arguments."""

    name: str
    kw: dict = field(default_factory=dict)


class FakeVLLM:
    """A scripted endpoint for engine.Engine. Based on the Stub in the script
    experiments/degraded_check.py, with every answer supplied by the test.

    streams   queue, one entry per stream() call. An entry is a Stream, a list
              of Steps (finish "stop"), a (steps, finish) pair, an exception
              instance (raised before any token, as a failed request), or a
              callable (prompt, max_tokens) -> any of those.
    fanouts   queue, one entry per fan-out read (top_next with max_tokens=1).
              An entry is a top list [(token, logprob), ...] or a dict, an
              exception, or a callable (prompts, k) -> top list.
    fanout    callable (prompts, k) -> top list, used once `fanouts` is empty.
    chunks    how chunk requests (top_next without max_tokens=1) are answered:
              a callable (prompts) -> [(text, lp, top, stopped)] with exactly
              one result per prompt, or a queue of such lists / exceptions /
              callables.
    score     callable (prompt) -> level probabilities, used by answer_probs.
              Renormalised over the ids asked for, as the client does. The
              default is uniform. See asked_axis() and thought_of().
    yes       callable (prompt) -> P(Yes), used by p_yes. Default 0.5.
    tokens    callable (text) -> token list, used by tokenize.
    cache_hit what prefix_cache_stats returns (value or zero-arg callable).

    Running out of `streams`, `fanouts` (with no `fanout`) or a chunk queue
    raises AssertionError naming the call, so a scenario that does not end the
    way the test expects fails instead of spinning.
    """

    def __init__(self, streams=(), fanouts=(), fanout=None, chunks=None,
                 score=None, yes=None, tokens=None, cache_hit=None):
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "retries": 0}
        self.last_finish = None
        self.streams = list(streams)
        self.fanouts = list(fanouts)
        self.fanout = fanout
        if callable(chunks):
            self._chunk_fn, self.chunk_queue = chunks, []
        else:
            self._chunk_fn, self.chunk_queue = None, list(chunks or [])
        self.score = score or uniform
        self.yes = yes or (lambda prompt: 0.5)
        self.tokens = tokens or (lambda text: list(range(len(text) // 4)))
        self.cache_hit = cache_hit
        self.calls = []
        self.closed = False

    # -- bookkeeping ----------------------------------------------------------

    def _record(self, name, **kw):
        self.calls.append(Call(name, kw))

    def names(self):
        """Call names in order, e.g. ["fanout", "chunk", "labels", ...]."""
        return [c.name for c in self.calls]

    def of(self, name):
        """The recorded arguments of every call to `name`, in order.
        "top_next" matches both fan-out and chunk requests."""
        want = ("fanout", "chunk") if name == "top_next" else (name,)
        return [c.kw for c in self.calls if c.name in want]

    def assert_consumed(self):
        left = {"streams": len(self.streams), "fanouts": len(self.fanouts),
                "chunks": len(self.chunk_queue)}
        left = {k: v for k, v in left.items() if v}
        assert not left, f"FakeVLLM: scripted entries never used: {left}"

    def _take(self, queue, what, *args):
        if not queue:
            n = sum(1 for c in self.calls if c.name == what)
            raise AssertionError(
                f"FakeVLLM: {what} call #{n} but its script ran out. The engine "
                "has no caps, so the script has to end the turn."
            )
        entry = queue.pop(0)
        if callable(entry) and not isinstance(entry, BaseException):
            entry = entry(*args)
        return entry

    def usage_snapshot(self):
        return dict(self.usage)

    # -- generation -----------------------------------------------------------

    async def stream(self, prompt, max_tokens, top_k=None, temperature=0.0, stop=None):
        self._record("stream", prompt=prompt, max_tokens=max_tokens, top_k=top_k,
                     temperature=temperature, stop=stop)
        self.usage["requests"] += 1
        self.last_finish = None
        entry = self._take(self.streams, "stream", prompt, max_tokens)
        if isinstance(entry, BaseException):
            raise entry
        s = _as_stream(entry)
        if len(s.steps) > max_tokens:
            raise AssertionError(
                f"FakeVLLM: scripted stream has {len(s.steps)} steps but the "
                f"engine asked for max_tokens={max_tokens}; split it with windows()"
            )
        n = len(s.steps)
        for i, st in enumerate(s.steps):
            last = i == n - 1 and s.error is None
            if last:
                # The real client sets last_finish from the chunk that carries
                # the final token, before yielding that token.
                self.last_finish = s.finish
            self.usage["completion_tokens"] += 1
            # A copy: the engine rewrites step.text when it cuts a marker.
            yield llm.Step(st.text, st.logprob, list(st.top),
                           s.finish if last else None)
        if s.error is not None:
            raise s.error
        self.last_finish = s.finish

    async def top_next(self, prompts, k=None, max_tokens=None):
        prompts = list(prompts)
        k = config.THINK_BRANCH_K if k is None else k
        if max_tokens == 1:
            # The fan-out read: one prompt, one token, the distribution over
            # how the thought could open.
            self._record("fanout", prompts=prompts, k=k, max_tokens=max_tokens)
            self.usage["requests"] += 1
            if len(prompts) != 1:
                raise AssertionError(
                    f"FakeVLLM: a fan-out read with {len(prompts)} prompts; the "
                    "engine has only ever sent one")
            if self.fanouts:
                entry = self._take(self.fanouts, "fanout", prompts, k)
            elif self.fanout is not None:
                entry = self.fanout(prompts, k)
            else:
                entry = self._take(self.fanouts, "fanout", prompts, k)
            if isinstance(entry, BaseException):
                raise entry
            top = _vllm_top(entry, k)
            text, lp = top[0] if top else ("", 0.0)
            if text:
                self.usage["completion_tokens"] += 1
                yield ("tok", 0, text)
            yield ("end", [(text, lp, top, False)])
            return

        self._record("chunk", prompts=prompts, k=k, max_tokens=max_tokens)
        self.usage["requests"] += 1
        if self._chunk_fn is not None:
            res = self._chunk_fn(prompts)
        else:
            res = self._take(self.chunk_queue, "chunk", prompts)
        if isinstance(res, BaseException):
            raise res
        res = list(res)
        if len(res) != len(prompts):
            raise AssertionError(
                f"FakeVLLM: chunk script returned {len(res)} results for "
                f"{len(prompts)} prompts; top_next always answers every prompt")
        out = []
        for i, r in enumerate(res):
            text, lp, top, stopped = r
            text = text or ""
            if text:
                self.usage["completion_tokens"] += max(1, len(text) // 4)
                yield ("tok", i, text)
            out.append((text, float(lp), _vllm_top(top, k), bool(stopped)))
        yield ("end", out)

    # -- closed questions -----------------------------------------------------

    async def labels(self, n):
        self._record("labels", n=n)
        if n > 26:
            raise RuntimeError(f"only 26 single-token labels available, need {n}")
        return [(chr(65 + i), LABEL_ID + i) for i in range(n)]

    async def answer_probs(self, prompts, ids=None):
        prompts = list(prompts)
        self._record("answer_probs", prompts=prompts, ids=ids)
        self.usage["requests"] += 1
        ids = [YES_ID, NO_ID] if ids is None else list(ids)
        out = []
        for p in prompts:
            row = [float(v) for v in self.score(p)][: len(ids)]
            row += [0.0] * (len(ids) - len(row))
            tot = sum(row)
            if tot <= 0:
                raise AssertionError(
                    f"FakeVLLM: score() gave no mass to any of {len(ids)} labels")
            out.append([v / tot for v in row])
        return out

    async def p_yes(self, prompts):
        prompts = list(prompts)
        self._record("p_yes", prompts=prompts)
        self.usage["requests"] += 1
        return [float(self.yes(p)) for p in prompts]

    # -- bookkeeping endpoints ------------------------------------------------

    async def tokenize(self, text):
        self._record("tokenize", text=text)
        return list(self.tokens(text))

    async def prefix_cache_stats(self):
        self._record("prefix_cache_stats")
        return self.cache_hit() if callable(self.cache_hit) else self.cache_hit

    async def aclose(self):
        self._record("aclose")
        self.closed = True


def _as_stream(entry):
    if isinstance(entry, Stream):
        return entry
    if (isinstance(entry, tuple) and len(entry) == 2
            and isinstance(entry[0], (list, tuple))
            and (entry[1] is None or isinstance(entry[1], str))):
        return Stream(list(entry[0]), entry[1])
    if isinstance(entry, (list, tuple)) and all(isinstance(x, llm.Step) for x in entry):
        return Stream(list(entry))
    raise TypeError(f"FakeVLLM: not a stream script entry: {entry!r}")


# ---------------------------------------------------------------------------
# Driving the engine
# ---------------------------------------------------------------------------


def run_turn(fake, user_text, conv=None):
    """One Engine.respond() turn to completion. Returns (events, conv)."""
    conv = Conversation() if conv is None else conv
    engine = Engine(fake)

    async def go():
        return [ev async for ev in engine.respond(conv, user_text)]

    return asyncio.run(go()), conv


def types(events):
    return [e["type"] for e in events]


def of_type(events, kind):
    return [e for e in events if e["type"] == kind]


# ---------------------------------------------------------------------------
# Wire level: what the endpoint sends back
# ---------------------------------------------------------------------------


def stream_chunk(tokens, logprobs=None, tops=None, finish=None, index=0):
    """One server-sent chunk of POST /v1/completions with stream=true and
    logprobs=k: tokens, their logprobs, and a top-logprobs dict per token."""
    if isinstance(tokens, str):
        tokens = [tokens]
    tokens = list(tokens)
    logprobs = [0.0] * len(tokens) if logprobs is None else list(logprobs)
    if tops is None:
        tops = [{t: lp} for t, lp in zip(tokens, logprobs)]
    return {
        "id": "cmpl-test",
        "object": "text_completion",
        "model": config.MODEL,
        "choices": [{
            "index": index,
            "text": "".join(tokens),
            "logprobs": {"tokens": tokens, "token_logprobs": logprobs,
                         "top_logprobs": list(tops)},
            "finish_reason": finish,
        }],
    }


def sse(*chunks, done=True):
    """Server-sent events body: one `data:` line per chunk (dict -> JSON,
    str -> as is), then `data: [DONE]`."""
    out = [
        "data: " + (c if isinstance(c, str) else json.dumps(c)) + "\n\n"
        for c in chunks
    ]
    if done:
        out.append("data: [DONE]\n\n")
    return "".join(out).encode()


def sse_response(*chunks, done=True):
    return httpx.Response(200, content=sse(*chunks, done=done),
                          headers={"content-type": "text/event-stream"})


def json_response(obj, status=200):
    return httpx.Response(status, json=obj)


def completion_json(texts, usage=None):
    """Non-streaming /v1/completions for a list of prompts."""
    return {
        "choices": [{"index": i, "text": t, "finish_reason": "length"}
                    for i, t in enumerate(texts)],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
    }


def answer_probs_json(rows, ids, usage=None):
    """/v1/completions with logprob_token_ids: one choice per prompt, the
    requested ids keyed "token_id:<id>" in top_logprobs[0]. A probability of 0
    leaves the id out, as a missing id does on the wire."""
    choices = []
    for i, row in enumerate(rows):
        tl = {f"token_id:{tid}": math.log(p) for tid, p in zip(ids, row) if p > 0}
        choices.append({
            "index": i,
            "text": "",
            "logprobs": {"tokens": [""], "token_logprobs": [0.0], "top_logprobs": [tl]},
            "finish_reason": "length",
        })
    return {"choices": choices,
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": len(rows)}}


def tokenize_json(tokens, max_model_len=32768):
    return {"count": len(tokens), "max_model_len": max_model_len, "tokens": list(tokens)}


def metrics_text(queries, hits, model=None):
    """The two vLLM prefix-cache counters in Prometheus text format."""
    m = model or config.MODEL
    return (
        "# TYPE vllm:prefix_cache_queries_total counter\n"
        f'vllm:prefix_cache_queries_total{{model_name="{m}"}} {float(queries)}\n'
        "# TYPE vllm:prefix_cache_hits_total counter\n"
        f'vllm:prefix_cache_hits_total{{model_name="{m}"}} {float(hits)}\n'
    )


def http_error(status, message=None, url=None):
    """An httpx.HTTPStatusError carrying a real request and response, the way
    raise_for_status() builds one. Raise it from a FakeVLLM script."""
    url = url or f"{config.BASE_URL}/completions"
    req = httpx.Request("POST", url)
    resp = httpx.Response(status, request=req)
    return httpx.HTTPStatusError(
        message or f"Server error '{status}' for url '{url}'",
        request=req, response=resp,
    )


def error_response(status):
    return httpx.Response(status, text=f"error {status}")


class BrokenStream(httpx.AsyncByteStream):
    """A response body that sends some bytes, then dies with `exc`."""

    def __init__(self, *parts, exc=None):
        self.parts = [p if isinstance(p, bytes) else p.encode() for p in parts]
        self.exc = exc or httpx.ReadTimeout("read timed out")

    async def __aiter__(self):
        for p in self.parts:
            yield p
        raise self.exc


@dataclass
class Seen:
    method: str
    url: str
    path: str
    headers: dict
    json: object


class Router:
    """A MockTransport handler that answers by (method, path), in order.

        router = Router().add("POST", "/v1/completions", sse_response(...))

    Each queued answer is used once: an httpx.Response, an exception instance
    (raised as a transport failure) or a callable (request) -> Response. Every
    request is recorded in `requests`, JSON body parsed. A request with nothing
    queued for it raises AssertionError.
    """

    def __init__(self):
        self._routes = {}
        self.requests = []

    def add(self, method, path, *answers):
        self._routes.setdefault((method.upper(), path), []).extend(answers)
        return self

    def __call__(self, request):
        body = request.content
        self.requests.append(Seen(
            method=request.method,
            url=str(request.url),
            path=request.url.path,
            headers=dict(request.headers),
            json=json.loads(body) if body else None,
        ))
        queue = self._routes.get((request.method, request.url.path))
        if not queue:
            raise AssertionError(f"Router: nothing queued for {request.method} {request.url}")
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer(request) if callable(answer) else answer

    def assert_consumed(self):
        left = {f"{m} {p}": len(q) for (m, p), q in self._routes.items() if q}
        assert not left, f"Router: queued answers never requested: {left}"


@contextlib.contextmanager
def mock_transport(handler):
    """While active, every httpx.AsyncClient is built on MockTransport(handler).

    Everything else the caller passes (base_url, headers, timeout) is kept, so
    llm.VLLM() constructed inside the block is the real client with only the
    transport swapped. trust_env is off so a proxy variable in the environment
    cannot route a request around the mock. The client keeps its transport
    after the block exits.
    """
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    class _MockedAsyncClient(real):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", transport)
            kwargs.setdefault("trust_env", False)
            super().__init__(*args, **kwargs)

    with mock.patch.object(httpx, "AsyncClient", _MockedAsyncClient):
        yield transport


def make_mock_client(handler):
    """A real llm.VLLM whose requests all go to `handler`."""
    with mock_transport(handler):
        return llm.VLLM()
