"""Does the semantic probe discriminate, or does entropy just pick synonyms?

calibrate.py produced an uncomfortable result: the highest-entropy positions are
overwhelmingly stylistic - 'chose' vs 'opted' vs 'relied', 'inexpensive' vs
'cheap'. Branching there yields five paraphrases of one answer. Meanwhile the
arithmetic prompts, where a wrong commitment is unrecoverable, never reach the
top of the entropy ranking at all.

That is exactly what tier 2 exists to fix. This script tests whether it does.
It reuses calibration.jsonl, rebuilds the context at each position, asks the
gate probe, and checks one thing:

    is P(yes) higher at positions that matter than at positions that don't?

If P(yes) is flat, the probe is not discriminating and the two-tier gate is
decoration. Report that, do not tune around it.

    python probe_check.py
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
import statistics as st

import config
from calibrate import PROMPTS
from llm import VLLM

HERE = os.path.dirname(os.path.abspath(__file__))
# Prompts 3 and 4 are the arithmetic ones; a probe that works should rate them
# higher than the essay prompts regardless of entropy.
ARITHMETIC = {3, 4}
BATCH = 16


def load():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "calibration.jsonl"), encoding="utf-8")]
    running = {}
    for r in rows:
        p = r["prompt"]
        r["ctx"] = running.get(p, "")
        running[p] = r["ctx"] + r["tok"]
    return rows


def full_prefix(row):
    return (
        config.turn("system", config.DEFAULT_SYSTEM)
        + config.turn("user", PROMPTS[row["prompt"] - 1])
        + config.assistant_header()
        + row["ctx"]
    )


async def main():
    rows = load()
    hi = [r for r in rows if r["h"] >= 1.4]
    # A control set matched in size, drawn from the calm middle of the range.
    mid = [r for r in rows if 0.2 <= r["h"] < 1.4]
    lo = mid[:: max(1, len(mid) // len(hi))][: len(hi)]
    for r in hi:
        r["band"] = "high-H"
    for r in lo:
        r["band"] = "mid-H"
    sample = hi + lo
    print(f"{len(hi)} high-entropy and {len(lo)} mid-entropy positions\n")

    llm = VLLM()
    try:
        for i in range(0, len(sample), BATCH):
            chunk = sample[i : i + BATCH]
            ps = await llm.p_yes(
                [config.probe_prompt(full_prefix(r), config.GATE_PROBE) for r in chunk]
            )
            for r, p in zip(chunk, ps):
                r["p_yes"] = p
            print(f"  probed {min(i + BATCH, len(sample))}/{len(sample)}", flush=True)
    finally:
        await llm.aclose()

    def summarise(label, rs):
        if not rs:
            return
        v = [r["p_yes"] for r in rs]
        print(f"  {label:22s} n={len(v):4d}  median {st.median(v):.3f}   "
              f"mean {st.mean(v):.3f}   min {min(v):.3f}  max {max(v):.3f}")

    print("\n=== by entropy band (is tier 2 independent of tier 1?) ===")
    summarise("high entropy", hi)
    summarise("mid entropy", lo)

    print("\n=== by prompt kind (does the probe know what matters?) ===")
    summarise("arithmetic prompts", [r for r in sample if r["prompt"] in ARITHMETIC])
    summarise("essay prompts", [r for r in sample if r["prompt"] not in ARITHMETIC])

    print("\n=== per prompt ===")
    for i, q in enumerate(PROMPTS, 1):
        rs = [r for r in sample if r["prompt"] == i]
        if rs:
            v = [r["p_yes"] for r in rs]
            print(f"  p{i} {'(arith)' if i in ARITHMETIC else '       '} "
                  f"median {st.median(v):.3f}  n={len(v):3d}  {q[:44]}")

    ranked = sorted(sample, key=lambda r: -r["p_yes"])
    print("\n=== the 15 positions the probe most wants to think about ===")
    for r in ranked[:15]:
        print(f"  P(yes)={r['p_yes']:.3f} H={r['h']:4.2f} {r['band']:6s} "
              f"p{r['prompt']} {r['tok']!r:14} <- {' | '.join(repr(a) for a in r['alts'][:3])}")
    print("\n=== the 10 it least wants to ===")
    for r in ranked[-10:]:
        print(f"  P(yes)={r['p_yes']:.3f} H={r['h']:4.2f} {r['band']:6s} "
              f"p{r['prompt']} {r['tok']!r:14} <- {' | '.join(repr(a) for a in r['alts'][:3])}")

    out = os.path.join(HERE, "probe_check.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps({k: r[k] for k in ("prompt", "pos", "tok", "h", "band", "p_yes")},
                                ensure_ascii=False) + "\n")
    print(f"\n  wrote {out}")

    allp = [r["p_yes"] for r in sample]
    spread = max(allp) - min(allp)
    print(f"\n  P(yes) spread across all sampled positions: {spread:.3f}")
    if spread < 0.15:
        print("  VERDICT: the probe is near-constant. It is not a gate, it is a "
              "bias term, and PROBE_THRESHOLD would just be an on/off switch.")
    else:
        hi_med = st.median([r["p_yes"] for r in hi])
        lo_med = st.median([r["p_yes"] for r in lo])
        print(f"  high-vs-mid entropy median gap: {hi_med - lo_med:+.3f}")


if __name__ == "__main__":
    asyncio.run(main())
