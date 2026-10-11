"""Offline: does a 524 after the opening thought keep the thinking?

No network. A stub stands in for the endpoint: the opening thought succeeds,
then the answer stream raises a gateway timeout. The turn must finish with the
thinking intact, the answer marked incomplete, and no exception escaping.

    python -m experiments.degraded_check

Prints one OK/FAIL line per check and exits with status 1 if any failed.
tests/test_degraded_turn.py covers the same scenario with assertions.

(Formerly test_degraded.py at the repository root, where it ran its scenario
as soon as it was imported.)
"""

import asyncio
import sys

import httpx

from adaptive_thinking import Conversation, Engine, chatml, head_note

from ._common import utf8_stdout

utf8_stdout()


class Resp:
    def __init__(self, code):
        self.status_code = code


class Stub:
    """Just enough endpoint to run one opening thought, then fall over."""

    def __init__(self, fail_after_block=True):
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "retries": 0}
        self.last_finish = None
        self.fail = fail_after_block
        self.block_done = False

    def usage_snapshot(self):
        return dict(self.usage)

    async def top_next(self, prompts, k=None, max_tokens=None):
        self.usage["requests"] += 1
        if max_tokens == 1:                       # the fan-out read
            top = [("Let", -0.1), ("The", -1.0), ("We", -2.0), ("This", -3.0)]
            yield ("end", [("Let", -0.1, top, False)])
            return
        out = []
        for i, _p in enumerate(prompts):
            body = f"thought {i}: the cases are A and B, so the answer follows."
            yield ("tok", i, body)
            out.append((body + "</think>", -5.0, [], True))
        self.block_done = True
        yield ("end", out)

    async def labels(self, n):
        return [(chr(65 + i), 1000 + i) for i in range(n)]

    async def answer_probs(self, prompts, ids=None):
        self.usage["requests"] += 1
        n = len(ids or [1])
        return [[1.0 / n] * n for _ in prompts]

    async def p_yes(self, prompts):
        return [0.5] * len(prompts)

    async def stream(self, prompt, max_tokens, **kw):
        self.usage["requests"] += 1
        if self.fail and self.block_done:
            raise httpx.HTTPStatusError("gateway timeout", request=None,
                                        response=Resp(524))
        yield None  # pragma: no cover - never reached in this test

    async def tokenize(self, text):
        return list(range(len(text) // 4))

    async def prefix_cache_stats(self):
        return None

    async def aclose(self):
        pass


async def main(stub=None):
    """Run the scenario and print the checks. True when every check passed.

    `stub` replaces the default Stub, so the checks can be seen to fail.
    """
    conv = Conversation()
    engine = Engine(Stub() if stub is None else stub)
    seen, done = [], None
    async for ev in engine.respond(conv, "Is a married person looking at an unmarried person?"):
        seen.append(ev["type"])
        if ev["type"] == "done":
            done = ev

    print("events:", " ".join(dict.fromkeys(seen)))
    print()
    ok = True

    def check(label, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  {'OK  ' if cond else 'FAIL'} {label}")

    check("the turn finished instead of raising", done is not None)
    check("a degraded event was emitted", "degraded" in seen)
    check("the stats say it is incomplete",
          bool(done and done["stats"].get("degraded")))
    t = next((x for x in conv.turns if x.role == "assistant"), None)
    check("the turn was still committed to the conversation", t is not None)
    h = head_note(t.notes) if t else None
    check("the opening thought survived in the header", h is not None and bool(h.text))
    check("it renders into the prompt",
          bool(h) and chatml.think_head(h.text) in conv.render())
    print()
    print("  kept thinking:", repr((h.text if h else "")[:70]))
    print("  reported     :", done["stats"].get("degraded") if done else None)
    print()
    print("RESULT:", "all good" if ok else "SOMETHING FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
