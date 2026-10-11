"""Does any of this help? Three modes, the same prompts, a checkable answer.

    plain     straight greedy generation. The baseline.
    adaptive  thinking blocks, invisible to the model. Measured across 14 tasks
              this NEVER changed the next token, and it cannot: with the block
              hidden the context after it is identical to the context before it,
              so at temperature 0 the model re-emits what it was going to emit.
              Keep it as the control that proves that, not as a treatment.
    inline    the same, but the block stays in the context the model reads, so
              it has to resume an answer interrupted mid-sentence. This is the
              one that tests whether the model copes with the seam.

Prompts are arithmetic on purpose: the answer is either right or it is not, so
"did thinking help" is graded rather than admired. Cost is reported as a
multiplier over plain, because this design is not free.

TWO THINGS TO KNOW BEFORE QUOTING THE NUMBERS.

Everything runs at temperature 0, so repeats are DETERMINISTIC - `runs > 1`
re-measures the same answers and tells you nothing. Breadth comes from adding
tasks, not from repeating them, which is why the list below is long rather than
the loop deep.

`prompt_tokens` is server-reported and covers BATCHED requests only. A streamed
request reports its usage in a final chunk that this engine deliberately never
reaches, because it aborts streams to open blocks. Decode tokens are counted
locally per token and are exact; prompt tokens for `plain` therefore read zero
and carry no multiplier.

    python bench.py [runs_per_prompt]
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
import time

import config
from llm import VLLM
from thinking import Conversation, Engine, cut_at_marker

HERE = os.path.dirname(os.path.abspath(__file__))

TASKS = [
    {
        "q": "A train leaves at 14:47 and the journey takes 3 hours 38 minutes. "
             "When does it arrive? Show your working.",
        "ok": ["18:25", "6:25"],
    },
    {
        "q": "A shop sells pens at 1.35 each and notebooks at 4.20 each. What do "
             "7 pens and 3 notebooks cost in total? Show the arithmetic.",
        "ok": ["22.05"],
    },
    {
        "q": "How many minutes are there between 09:35 and 13:12? Show your working.",
        "ok": ["217"],
    },
    {
        "q": "A car travels 246 km in 3 hours. What is its average speed in km/h?",
        "ok": ["82"],
    },
    {
        "q": "A rope is 7.5 m long. You cut off two pieces of 1.85 m each. "
             "How much rope remains? Show your working.",
        "ok": ["3.8"],
    },
    {
        "q": "What is 17% of 240? Show your working.",
        "ok": ["40.8"],
    },
    {
        "q": "A meeting starts at 11:50 and runs for 2 hours 45 minutes. "
             "When does it end? Show your working.",
        "ok": ["14:35", "2:35"],
    },
    {
        "q": "A bag holds 3.6 kg of flour. You use 850 g. How much is left, "
             "in kilograms? Show your working.",
        "ok": ["2.75", "2,75"],
    },
    {
        "q": "A rectangle is 14 cm by 9 cm. What is the difference between its "
             "area and its perimeter? Show your working.",
        "ok": ["80"],
    },
    {
        "q": "You buy 4 items at 3.75 and pay with a 20 note. What change do "
             "you get? Show your working.",
        "ok": ["5.00", "5 "],
    },
    {
        "q": "A tank loses 1.4 litres per hour and holds 23 litres. After how "
             "many whole hours does less than 5 litres remain? Show your working.",
        "ok": ["13"],
    },
    {
        "q": "Split 91 sweets between two children so one gets 13 more than the "
             "other. How many does each get? Show your working.",
        "ok": ["52", "39"],
    },
    {
        "q": "A film is 137 minutes long and starts at 19:40. When does it end? "
             "Show your working.",
        "ok": ["21:57", "9:57"],
    },
    {
        "q": "What is 3/8 of 216? Show your working.",
        "ok": ["81"],
    },
]


def graded(text, ok):
    tail = text[-260:]  # the conclusion, not a number that appeared in passing
    return any(o in tail for o in ok)


def repetition(text, n=34):
    """Fraction of n-char shingles that occur more than once.

    The failure mode to watch for when a thinking block is left in the context:
    the model loses the thread and restarts the sentence, or the paragraph, it
    was half-way through. That shows up as duplicated spans long before it
    shows up as a wrong answer.
    """
    if len(text) < n * 2:
        return 0.0
    seen, dup = set(), 0
    for i in range(len(text) - n):
        sh = text[i : i + n]
        if sh in seen:
            dup += 1
        seen.add(sh)
    return round(dup / (len(text) - n), 4)


async def plain(llm, q):
    """The baseline: one streamed request, nothing else."""
    t0 = time.time()
    u0 = llm.usage_snapshot()
    prompt = (
        config.turn("system", config.DEFAULT_SYSTEM)
        + config.turn("user", q)
        + config.assistant_header()
    )
    out, lps, spills = "", [], 0
    # Cut the same way the engine does, or the baseline gets marked down for a
    # leak the adaptive path is filtering and the comparison is not like-for-like.
    async for step in llm.stream(prompt, max_tokens=config.MAX_TOKENS):
        clean, marker = cut_at_marker(step.text)
        out += clean
        if marker is not None:
            spills += 1
            break
        lps.append(step.logprob)
    u1 = llm.usage_snapshot()
    return out.strip(), {
        "secs": round(time.time() - t0, 2),
        "kept_tokens": len(lps),
        "rewound_tokens": 0,
        "explored_tokens": 0,
        "blocks": 0,
        "blocks_that_changed_next_token": 0,
        "think_steps": 0,
        "requests": u1["requests"] - u0["requests"],
        "prompt_tokens": u1["prompt_tokens"] - u0["prompt_tokens"],
        "completion_tokens": u1["completion_tokens"] - u0["completion_tokens"],
        "logprob": round(sum(lps) / len(lps), 4) if lps else None,
        "resume_logprob_delta": None,
        "spills": spills,
        "switches_a_margin_would_reject": 0,
    }


async def adaptive(llm, q, inline):
    config.KEEP_BLOCKS_INLINE = inline
    engine = Engine(llm)
    conv = Conversation()
    text, stats = "", {}
    async for ev in engine.respond(conv, q):
        if ev["type"] == "done":
            text, stats = ev["text"], ev["stats"]
    return text, stats


MODES = ("plain", "adaptive", "inline")


async def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    llm = VLLM()
    rows = []
    out_path = os.path.join(HERE, "bench.jsonl")
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            for r in range(runs):
                for t in TASKS:
                    for mode in MODES:
                        if mode == "plain":
                            text, stats = await plain(llm, t["q"])
                        else:
                            text, stats = await adaptive(llm, t["q"], mode == "inline")
                        row = {
                            "run": r,
                            "mode": mode,
                            "q": t["q"][:48],
                            "correct": graded(text, t["ok"]),
                            "repetition": repetition(text),
                            # If this is ever non-zero the note leaked out of the
                            # model's context into the reader's answer.
                            "think_leak": "<think>" in text,
                            "text": text,
                            **stats,
                        }
                        rows.append(row)
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                        mark = "OK " if row["correct"] else "BAD"
                        print(
                            f"  {mark} {mode:9s} {stats['secs']:6.1f}s "
                            f"blocks={stats['blocks']:2d} "
                            f"changed={stats['blocks_that_changed_next_token']:2d}  {t['q'][:44]}",
                            flush=True,
                        )
    finally:
        await llm.aclose()

    def agg(mode, key):
        v = [r[key] for r in rows if r["mode"] == mode and r.get(key) is not None]
        return sum(v) / len(v) if v else 0.0

    base_secs = agg("plain", "secs") or 1.0
    base_comp = agg("plain", "completion_tokens") or 1.0

    print(f"\n=== {len(rows)} runs, {len(TASKS)} tasks x {runs} ===")
    print(f"  {'mode':10s} {'correct':>9s} {'secs':>7s} {'x':>5s} {'decode tok':>11s} "
          f"{'x':>5s} {'batch prompt':>13s} {'blocks':>7s} {'changed':>8s}")
    for mode in MODES:
        rs = [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        n_ok = sum(1 for r in rs if r["correct"])
        s, c, p = agg(mode, "secs"), agg(mode, "completion_tokens"), agg(mode, "prompt_tokens")
        print(f"  {mode:10s} {n_ok:3d}/{len(rs):<3d} {s:8.1f} {s / base_secs:5.1f} {c:11.0f} "
              f"{c / base_comp:5.1f} {p:13.0f} "
              f"{agg(mode, 'blocks'):7.1f} {agg(mode, 'blocks_that_changed_next_token'):8.1f}")
    print("  decode tokens are counted locally and are exact; 'batch prompt' is\n"
          "  server-reported for batched requests only, so plain reads 0 - see the\n"
          "  note at the top of this file.")

    print("\n  can it resume after an interruption?")
    for mode in MODES:
        rs = [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        d = [r["resume_logprob_delta"] for r in rs if r.get("resume_logprob_delta") is not None]
        lp = agg(mode, "logprob")
        rep = agg(mode, "repetition")
        # Any marker reaching the reader is a bug; `spills` counts what the
        # engine cut out before it could.
        leaks = sum(1 for r in rs
                    if any(m in r.get("text", "") for m in
                           ("<|im_start|>", "<|im_end|>", "<think>", "</think>")))
        delta = f"{sum(d) / len(d):+.3f}" if d else "   n/a"
        print(f"  {mode:10s} mean logprob {lp:7.3f}   post-block delta {delta}   "
              f"repetition {rep:.4f}   markers leaked {leaks}/{len(rs)}   "
              f"cut {agg(mode, 'spills'):.1f}/run")
    print("\n  a delta near 0 means the model does not notice the seam; a clearly\n"
          "  negative one means the interruption costs it the thread. Rising\n"
          "  repetition means it lost the thread and restarted the sentence.")

    print("\n  per task (correct?)")
    print(f"  {'task':50s} " + " ".join(f"{m:>9s}" for m in MODES))
    for t in TASKS:
        cells = []
        for m in MODES:
            rs = [r for r in rows if r["mode"] == m and r["q"] == t["q"][:48]]
            cells.append(f"{sum(1 for r in rs if r['correct'])}/{len(rs)}")
        print(f"  {t['q'][:50]:50s} " + " ".join(f"{c:>9s}" for c in cells))
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
