"""Measure the entropy distribution before choosing a threshold.

A gate threshold picked by intuition is worthless: with top-5 logprobs the
truncated entropy is bounded by ln(6) = 1.79 nats, so the "obviously
reasonable" 2.0 would never fire once. This script generates over a spread of
prompts, collects every token's entropy and margin, prints the percentiles, and
shows WHICH tokens sit above each candidate threshold - point 2 of the
verification plan, answered with data instead of a guess.

    python calibrate.py [tokens_per_prompt]
"""

import sys
# Windows consoles default to cp1252 and the model emits U+2212 and friends.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import asyncio
import json
import os

import config
from llm import VLLM

PROMPTS = [
    # factual, low ambiguity - the gate should be mostly quiet here
    "In two sentences, why did the Hindenburg use hydrogen rather than helium?",
    "What is the capital of Australia? One sentence.",
    # arithmetic - a place where a wrong commitment is unrecoverable
    "A shop sells pens at 1.35 each and notebooks at 4.20. What do 7 pens and "
    "3 notebooks cost? Show the arithmetic.",
    "If a train leaves at 14:47 and the journey takes 3 hours 38 minutes, when "
    "does it arrive? Show your working.",
    # judgement calls - genuinely open continuations
    "Is it better to index a database column that is 95% NULL? Explain the "
    "trade-off in three sentences.",
    "Give one reason a company might deliberately keep a loss-making product "
    "line, and one reason it should not.",
    # list boundaries and names
    "Name three programming languages designed at Bell Labs and say what each "
    "was for.",
    "Summarise, in three bullets, what changed in HTTP/2 compared to HTTP/1.1.",
]

CANDIDATES = (0.8, 1.0, 1.2, 1.4, 1.6)


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))]


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    llm = VLLM()
    rows = []
    try:
        for i, q in enumerate(PROMPTS, 1):
            prompt = (
                config.turn("system", config.DEFAULT_SYSTEM)
                + config.turn("user", q)
                + config.assistant_header()
            )
            k = 0
            async for step in llm.stream(prompt, max_tokens=n):
                rows.append(
                    {
                        "prompt": i,
                        "pos": k,
                        "tok": step.text,
                        "h": round(step.entropy, 4),
                        "margin": round(step.margin, 4),
                        "alts": step.alternatives[:5],
                    }
                )
                k += 1
            print(f"  [{i}/{len(PROMPTS)}] {k:4d} tokens  {q[:52]}", flush=True)
    finally:
        await llm.aclose()

    hs = [r["h"] for r in rows]
    ms = [r["margin"] for r in rows]
    print(f"\n{len(rows)} tokens over {len(PROMPTS)} prompts")
    print(f"  max possible truncated entropy with top-{config.BRANCH_K}: "
          f"{__import__('math').log(config.BRANCH_K + 1):.2f} nats")
    print("\n  percentile     entropy    margin")
    for p in (50, 75, 90, 95, 97, 99):
        print(f"      p{p:<3d}        {pct(hs, p):6.2f}    {pct(ms, p):6.2f}")

    print("\n  threshold   fires   rate   per 120 tokens")
    for t in CANDIDATES:
        hit = [r for r in rows if r["h"] >= t]
        rate = len(hit) / len(rows) if rows else 0
        print(f"      {t:4.1f}     {len(hit):5d}  {rate:5.1%}   {rate * 120:6.1f}")

    # Point 2 of the plan: do the firings land anywhere sensible?
    top = sorted(rows, key=lambda r: -r["h"])[:25]
    print("\n  the 25 highest-entropy positions (does the gate land sensibly?)")
    for r in top:
        alts = " | ".join(repr(a) for a in r["alts"][:4])
        print(f"    {r['h']:5.2f}  p{r['prompt']}@{r['pos']:<4d} {r['tok']!r:14} <- {alts}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  wrote {out}")

    # A block is expensive; aim for a handful per response, not a hundred.
    target = next((t for t in CANDIDATES
                   if sum(1 for r in rows if r["h"] >= t) / max(1, len(rows)) <= 0.08), CANDIDATES[-1])
    print(f"\n  suggested ENTROPY_THRESHOLD={target}  "
          f"(<=8% of positions; MIN_GAP_TOKENS thins it further)")


if __name__ == "__main__":
    asyncio.run(main())
