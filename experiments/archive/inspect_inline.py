"""Show exactly what the model reads when a thinking block is left inline.

The first inline run copied the block's closing nudge into the visible answer.
That is either the model imitating a pattern it can see, or a bug in how the
note is spliced. Guessing between those is pointless - this prints the raw
model-facing context around every block, the monologue the model wrote for
itself, and the tokens it produced immediately afterwards.

    python inspect_inline.py ["question"]
"""

import sys
# Windows consoles default to cp1252 and the model emits U+2212 and friends.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import asyncio
import os

import config
from llm import VLLM
from thinking import Conversation, Engine

config.KEEP_BLOCKS_INLINE = True
config.MAX_BLOCKS = int(os.getenv("MAX_BLOCKS", "2"))
config.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "160"))


async def main():
    q = " ".join(sys.argv[1:]) or "How many minutes are there between 09:35 and 13:12? Show your working."
    llm = VLLM()
    engine = Engine(llm)
    conv = Conversation()

    # Wrap splice so every context the engine builds can be seen, not inferred.
    import thinking

    real_splice = thinking.splice
    last = {"ctx": ""}

    def spy(vis, notes):
        out = real_splice(vis, notes)
        last["ctx"] = out
        return out

    thinking.splice = spy

    after, watching = [], 0
    try:
        async for ev in engine.respond(conv, q):
            t = ev["type"]
            if t == "monologue":
                print(f"\n--- MONOLOGUE the model wrote for block {ev['block']} ---")
                print(repr(ev["text"]))
            elif t == "block_close":
                print(f"\n--- CONTEXT the model reads after block {ev['id']} closes ---")
                print(repr(last["ctx"][-420:]))
                print(f"    commit = {ev['commit']!r}")
                after, watching = [], 24
            elif t == "token" and watching:
                after.append(ev["text"])
                watching -= 1
                if watching == 0:
                    print(f"    next 24 tokens -> {''.join(after)!r}")
            elif t == "done":
                print("\n--- FINAL VISIBLE ANSWER ---")
                print(ev["text"])
                leaked = [m for m in ("<think>", "</think>", "The next word is") if m in ev["text"]]
                print(f"\nleaked markers in the visible answer: {leaked or 'none'}")
                print(f"resume delta: {ev['stats']['resume_logprob_delta']}")
    finally:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
