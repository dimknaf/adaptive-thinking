"""vLLM client. Three request shapes, nothing else.

    stream()        one prompt, tokens back one at a time WITH the top-K
                    distribution at each position. Tier 1 of the gate is free
                    because this distribution is already on the wire.

    top_next()      a LIST of prompts in ONE streamed request. This is how the
                    beams of a thinking block are extended: every live beam
                    advances by THINK_CHUNK tokens in one round trip, one batch
                    of forward passes, and hands back the top-k alternatives
                    at the end of its chunk. complete() is the same batched
                    shape without streaming or logprobs; only the smoke test
                    below uses it.

    answer_probs()  a LIST of prompts, max_tokens=1, and `logprob_token_ids` -
                    a vLLM extension that returns the logprob of specific token
                    ids even when they are not in the top 20. Reads P(Yes), or
                    a letter label, without generating anything.

The shapes were verified against a live vLLM server before being written down;
`logprob_token_ids` comes back keyed as "token_id:<id>" in top_logprobs[0].
"""

import asyncio
import json
import math
import re
import string
import time
from dataclasses import dataclass, field

import httpx

from . import chatml, config, prompts


# Failures that are the endpoint having a bad moment, not us asking for
# something wrong. If the endpoint sits behind a proxy or CDN, gateway
# timeouts such as 524 are transient and retried: on a busy server with
# requests queued for capacity, one killed a turn that had already spent 374
# seconds producing its opening thought.
RETRY_STATUS = {429, 500, 502, 503, 504, 524}


def transient(exc):
    """Is this worth trying again, or is it our fault?"""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRY_STATUS
    return isinstance(
        exc,
        (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
         httpx.WriteTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError),
    )


def _give_up(exc, attempt, started=False):
    """Is a failed attempt final, or worth another try?

    Final once a stream has handed anything to its caller (`started`), once
    LLM_RETRIES retries have been spent, or when the failure is not transient.
    The one retry rule for every request shape; VLLM._backoff does the rest.
    """
    return started or attempt >= config.LLM_RETRIES or not transient(exc)


def identity_headers():
    """The Authorization and User-Agent headers every request carries.

    Read from config at call time. VLLM sends these plus a JSON Content-Type;
    a caller with its own HTTP client (the zebra runner's native arm) sends
    them as they are, so the endpoint sees one identity from both.
    """
    return {
        "Authorization": f"Bearer {config.API_KEY}",
        "User-Agent": config.USER_AGENT,
    }


def _payload(prompt, **fields):
    """A request body: the model, the prompt, then `fields` in the order given.

    Every value goes on the wire exactly as the caller passed it, so each call
    site keeps its own literals: `temperature` is 0 in some requests and 0.0
    in others, and the two serialise differently.
    """
    return {"model": config.MODEL, "prompt": prompt, **fields}


def _sorted_top(d, k=None):
    """A top-logprobs dict as [(token, logprob)], best first.

    None values are dropped, and the list is cut to `k` when one is given.
    """
    top = sorted(
        ((t, v) for t, v in (d or {}).items() if v is not None),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return top if k is None else top[:k]


async def _iter_sse(resp):
    """The parsed JSON of each `data:` line of a streamed response.

    An error status is raised before anything is parsed, once the body has
    been read so that the response is complete. Lines that are not `data:`
    are skipped, and the stream ends at `[DONE]`.
    """
    if resp.status_code >= 400:
        await resp.aread()
        resp.raise_for_status()
    async for line in resp.aiter_lines():
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            return
        yield json.loads(body)


@dataclass
class Step:
    """One generated token, plus the distribution it was drawn from."""

    text: str
    logprob: float
    top: list = field(default_factory=list)  # [(token_text, logprob)] desc
    finish: str = None

    @property
    def probs(self):
        return [math.exp(lp) for _, lp in self.top]

    @property
    def entropy(self):
        """Truncated entropy in nats.

        We only receive the top K, so this is a LOWER BOUND on the true
        entropy: the unseen mass is treated as one bucket rather than spread
        over the tail. That is fine for a gate - it is monotone in the thing we
        care about - but it is not the model's real entropy.
        """
        ps = [p for p in self.probs if p > 0]
        resid = max(0.0, 1.0 - sum(ps))
        h = -sum(p * math.log(p) for p in ps)
        if resid > 1e-9:
            h -= resid * math.log(resid)
        return h

    @property
    def margin(self):
        ps = self.probs
        return ps[0] - ps[1] if len(ps) > 1 else 1.0

    @property
    def alternatives(self):
        """Top-K token strings, best first, deduplicated.

        Index 0 is what greedy decoding would have emitted, so a winning
        branch index of 0 means the block changed nothing.
        """
        out, seen = [], set()
        for tok, _ in self.top:
            if tok and tok not in seen:
                seen.add(tok)
                out.append(tok)
        return out


class VLLM:
    def __init__(self):
        if not config.BASE_URL or not config.MODEL:
            raise RuntimeError("OPENAI_BASE_URL and OPENAI_MODEL must be set - see .env.example")
        self._c = httpx.AsyncClient(
            base_url=config.BASE_URL,
            timeout=httpx.Timeout(config.TIMEOUT, connect=30.0),
            headers={**identity_headers(), "Content-Type": "application/json"},
        )
        # completion_tokens is counted LOCALLY for streams, one per Step, and
        # taken from the server for batched calls. The server's own count is
        # useless for streams here: it arrives in a final usage chunk that an
        # aborted stream never reaches, and this engine aborts streams on
        # purpose. Relying on it made the baseline report zero decode tokens
        # and turned the cost multiplier into a division by zero.
        # prompt_tokens is server-reported and therefore covers BATCHED calls
        # only. Do not quote it as a total.
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "retries": 0}
        # Why a mutable attribute rather than a return value: stream() is an
        # async generator the caller may abandon mid-way, so there is no return
        # value to read. "stop" here means the model ended its turn; "length"
        # means we simply ran out of window.
        self.last_finish = None
        self._yes_id = None
        self._no_id = None
        self._labels = None

    async def aclose(self):
        await self._c.aclose()

    # -- tokenizer ----------------------------------------------------------

    async def tokenize(self, text):
        r = await self._c.post(
            f"{config.ROOT_URL}/tokenize",
            json=_payload(text, add_special_tokens=False),
        )
        r.raise_for_status()
        return r.json()["tokens"]

    async def answer_ids(self):
        """Token ids for the two probe answers, resolved once.

        Whitespace decides tokenisation, so both spellings are tried and the
        single-token one wins. If neither is single-token the probe is
        meaningless and we say so loudly rather than scoring noise.
        """
        if self._yes_id is not None:
            return self._yes_id, self._no_id
        out = []
        for word in (prompts.YES, prompts.NO):
            for spelling in (word, " " + word):
                ids = await self.tokenize(spelling)
                if len(ids) == 1:
                    out.append(ids[0])
                    break
            else:
                raise RuntimeError(f"{word!r} is not a single token for {config.MODEL}")
        self._yes_id, self._no_id = out
        return self._yes_id, self._no_id

    async def _post(self, path, payload):
        """POST with a retry on a transient failure.

        Safe to replay in full, unlike the streaming calls: nothing has been
        handed to the caller until the response parses.
        """
        attempt = 0
        while True:
            try:
                r = await self._c.post(path, json=payload)
                r.raise_for_status()
                return r
            except Exception as exc:
                if _give_up(exc, attempt):
                    raise
                attempt = await self._backoff(attempt)

    async def _backoff(self, attempt):
        """Count one retry and wait before it is made. Returns its number.

        Linear: retry n waits LLM_RETRY_BACKOFF x n seconds. Only `retries` is
        counted here. `requests` is counted by each caller, because the shapes
        differ on purpose: a stream counts every attempt it opens, while
        _post's callers count one request, once it has succeeded.
        """
        attempt += 1
        self.usage["retries"] += 1
        await asyncio.sleep(config.LLM_RETRY_BACKOFF * attempt)
        return attempt

    # -- generation ---------------------------------------------------------

    async def stream(self, prompt, max_tokens, top_k=None, temperature=0.0, stop=None):
        """Yield a Step per token. Break out of the loop to abort the request.

        `stop` overrides the default list, which is needed to let the model use
        its OWN thinking: chatml.STOP contains "<think>" precisely to stop that
        happening inside an answer, so a native-thinking baseline has to pass
        [IM_END] instead.
        """
        top_k = config.BRANCH_K if top_k is None else top_k
        payload = _payload(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=top_k,
            stop=chatml.STOP if stop is None else stop,
            stream=True,
        )
        self.last_finish = None
        attempt = 0
        while True:
            started = False
            try:
                self.usage["requests"] += 1
                async for step in self._stream_once(payload):
                    started = True
                    self.usage["completion_tokens"] += 1
                    yield step
                return
            except Exception as exc:
                # Never replay a stream that has already handed out a token:
                # the caller has appended it to the answer, and a second copy
                # would be written underneath the first.
                if _give_up(exc, attempt, started):
                    raise
                attempt = await self._backoff(attempt)

    async def _stream_once(self, payload):
        async with self._c.stream("POST", "/completions", json=payload) as resp:
            async for chunk in _iter_sse(resp):
                for choice in chunk.get("choices") or []:
                    if choice.get("finish_reason"):
                        self.last_finish = choice["finish_reason"]
                    for step in self._steps(choice):
                        yield step

    @staticmethod
    def _steps(choice):
        lp = choice.get("logprobs") or {}
        tokens = lp.get("tokens") or ([choice["text"]] if choice.get("text") else [])
        tlps = lp.get("token_logprobs") or [0.0] * len(tokens)
        tops = lp.get("top_logprobs") or [{}] * len(tokens)
        finish = choice.get("finish_reason")
        for i, tok in enumerate(tokens):
            yield Step(
                text=tok,
                logprob=tlps[i] if i < len(tlps) else 0.0,
                top=_sorted_top(tops[i]),
                finish=finish if i == len(tokens) - 1 else None,
            )

    async def top_next(self, prompts, k=None, max_tokens=None):
        """Advance every beam by a chunk, and hand back where it can branch.

        Returns, per prompt: (chunk_text, chunk_logprob, [(token, logprob)] at
        the END of the chunk, stopped_early).

        A request cost ~2.7s on the measured endpoint whatever it generated - a
        one-token request cost the same as a fifty-token one. So generating one
        token per round trip wasted that cost fifty times over. The chunk is how
        often the tree checkpoints, not a limit on how much it may think: the
        caller keeps calling until a beam closes its own think block.

        Streams, so the caller sees tokens as they are produced. That costs
        nothing - it is the same single request either way - and it is the only
        way the conversations can be watched growing rather than appearing in
        100-token lumps.

        Yields ("tok", beam_index, text) as tokens arrive, then ("end", results)
        once, where results is that 4-tuple per prompt.
        """
        k = config.THINK_BRANCH_K if k is None else k
        n = config.THINK_CHUNK if max_tokens is None else max_tokens
        payload = _payload(
            list(prompts),
            max_tokens=n,
            temperature=0,
            logprobs=k,
            # A beam that finishes thinking stops here rather than running on
            # to the end of the chunk.
            stop=chatml.STOP,
            stream=True,
        )
        m = len(prompts)
        attempt = 0
        while True:
            # Reset per attempt: a retry re-generates the whole chunk, so
            # carrying the dead attempt's partial text would corrupt it.
            text = [""] * m
            score = [0.0] * m
            tops = [[] for _ in range(m)]
            stopped = [False] * m
            started = False
            try:
                self.usage["requests"] += 1
                async with self._c.stream("POST", "/completions", json=payload) as resp:
                    async for ch_json in _iter_sse(resp):
                        if ch_json.get("usage"):
                            self._add_usage(ch_json["usage"])
                        for ch in ch_json.get("choices") or []:
                            i = ch.get("index", 0)
                            piece = ch.get("text") or ""
                            lp = ch.get("logprobs") or {}
                            for v in lp.get("token_logprobs") or []:
                                if v is not None:
                                    score[i] += v
                                    self.usage["completion_tokens"] += 1
                            # Only the LAST position matters for branching, so
                            # this keeps being overwritten until the chunk ends.
                            for d in lp.get("top_logprobs") or []:
                                if d:
                                    tops[i] = _sorted_top(d, k)
                            if ch.get("finish_reason") == "stop":
                                stopped[i] = True
                            if piece:
                                text[i] += piece
                                started = True
                                yield ("tok", i, piece)
                break
            except Exception as exc:
                # Tokens already streamed to the UI cannot be un-yielded, so a
                # chunk that died part-way is not replayed.
                if _give_up(exc, attempt, started):
                    raise
                attempt = await self._backoff(attempt)

        yield ("end", [(text[i], score[i], tops[i], stopped[i]) for i in range(m)])

    async def labels(self, n):
        """n verified single-token labels for a closed-set choice.

        A label only counts if it encodes to exactly one token id. Without that
        check a "label" can silently be two tokens and every probability read
        off it is meaningless. (The labelled single-token question these serve
        is inspired by jevfire.)

        The letters A-Z are verified one by one through the server's
        /tokenize, so the labels come from the served model's own tokenizer
        and nothing is loaded locally. The single-token letters are read once
        and cached. More labels than there are single-token letters is a
        RuntimeError.
        """
        if self._labels is not None and len(self._labels) >= n:
            return self._labels[:n]
        out = []
        for ch in string.ascii_uppercase:
            ids = await self.tokenize(ch)
            if len(ids) == 1:
                out.append((ch, ids[0]))
            if len(out) >= max(n, 26):
                break
        if len(out) < n:
            raise RuntimeError(f"only {len(out)} single-token labels available, need {n}")
        self._labels = out
        return self._labels[:n]

    # -- batch --------------------------------------------------------------

    async def complete(self, prompts, max_tokens, temperature=0.0, stop=None):
        """A list of prompts in one request -> a list of continuations."""
        r = await self._post(
            "/completions",
            _payload(
                list(prompts),
                max_tokens=max_tokens,
                temperature=temperature,
                stop=chatml.STOP if stop is None else stop,
            ),
        )
        js = r.json()
        self.usage["requests"] += 1
        self._add_usage(js.get("usage") or {})
        out = [""] * len(prompts)
        for ch in js["choices"]:
            out[ch["index"]] = ch.get("text") or ""
        return out

    async def answer_probs(self, prompts, ids=None):
        """P over `ids` at the next position, for each prompt, in one request.

        Renormalised over the given ids only: we are asking a closed question,
        so the mass the model puts elsewhere is not part of the answer.
        """
        if ids is None:
            ids = list(await self.answer_ids())
        r = await self._post(
            "/completions",
            _payload(
                list(prompts),
                max_tokens=1,
                temperature=0,
                logprobs=0,
                logprob_token_ids=list(ids),
                return_tokens_as_token_ids=True,
            ),
        )
        js = r.json()
        self.usage["requests"] += 1
        self._add_usage(js.get("usage") or {})

        out = [None] * len(prompts)
        for ch in js["choices"]:
            tl = ((ch.get("logprobs") or {}).get("top_logprobs") or [{}])[0] or {}
            raw = [tl.get(f"token_id:{i}") for i in ids]
            if all(v is None for v in raw):
                raise RuntimeError(
                    "logprob_token_ids returned nothing - this endpoint is "
                    "silently ignoring the parameter, so every score would be "
                    "meaningless. Check the vLLM version (0.26+)."
                )
            vals = [math.exp(v) if v is not None else 0.0 for v in raw]
            tot = sum(vals) or 1.0
            out[ch["index"]] = [v / tot for v in vals]
        return out

    async def p_yes(self, prompts):
        """Shorthand: probability of Yes for each prompt."""
        return [p[0] for p in await self.answer_probs(prompts)]

    # -- bookkeeping --------------------------------------------------------

    def _add_usage(self, u):
        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
        self.usage["completion_tokens"] += u.get("completion_tokens", 0) or 0

    def usage_snapshot(self):
        return dict(self.usage)

    async def prefix_cache_stats(self):
        """Cumulative prefix-cache hit rate from vLLM's own /metrics.

        Used to price an eviction honestly. Deleting notes rewrites the prefix,
        so the next request re-prefills from the first evicted position; this is
        the only way to show that cost rather than assert it is small. Returns
        None if the endpoint does not expose the counters.
        """
        try:
            r = await self._c.get(f"{config.ROOT_URL}/metrics", timeout=20.0)
            r.raise_for_status()
            body = r.text
        except Exception:
            return None
        got = {}
        for name in ("prefix_cache_queries_total", "prefix_cache_hits_total"):
            m = re.search(rf"^vllm:{name}\{{[^}}]*}}\s+(\S+)", body, re.M)
            if m:
                got[name] = float(m.group(1))
        if len(got) != 2 or not got["prefix_cache_queries_total"]:
            return None
        return round(got["prefix_cache_hits_total"] / got["prefix_cache_queries_total"], 4)


class UsageMeter:
    """Wall time and client usage since the meter started: a turn, a block.

    Takes the clock and a usage snapshot when made, in that order. Nothing is
    rounded here; each caller rounds what it reports, to its own precision.
    """

    def __init__(self, client):
        self._client = client
        self.t0 = time.time()
        self.u0 = client.usage_snapshot()

    def secs(self):
        """Seconds since the meter started."""
        return time.time() - self.t0

    def used(self):
        """How far each usage counter has moved, from ONE fresh snapshot."""
        u1 = self._client.usage_snapshot()
        return {k: u1[k] - v for k, v in self.u0.items()}


async def _smoke():
    """python -m adaptive_thinking.llm - confirms endpoint, UA, logprobs shape
    and the extension."""
    llm = VLLM()
    try:
        print(f"model    : {config.MODEL}")
        print(f"endpoint : {config.BASE_URL}")
        yes, no = await llm.answer_ids()
        print(f"answer   : Yes={yes} No={no}")

        prompt = (
            chatml.turn("system", prompts.DEFAULT_SYSTEM)
            + chatml.turn("user", "In two sentences, why did the Hindenburg use hydrogen?")
            + chatml.assistant_header()
        )
        print("\nstreaming 40 tokens (entropy per token):")
        n = 0
        async for step in llm.stream(prompt, max_tokens=40):
            bar = "#" * min(20, int(step.entropy * 6))
            print(f"  {step.entropy:5.2f} {bar:<20} {step.text!r}")
            n += 1
        print(f"\n{n} tokens")

        ctx = prompt + "Because hydrogen is the lightest gas"
        p = await llm.p_yes([prompts.probe_prompt(ctx, prompts.GATE_PROBE)])
        print(f"gate probe P(yes) = {p[0]:.3f}")

        outs = await llm.complete([ctx + t for t in (",", " and", " -")], max_tokens=12)
        for t, o in zip((",", " and", " -"), outs):
            print(f"  branch {t!r:8} -> {o!r}")
        print("usage:", llm.usage_snapshot())
    finally:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(_smoke())
