"""Does the gate stop where it should, or for trivial reasons?

The old Yes/No probe was measured in isolation once (median P(yes): arithmetic
0.622, essay prose 0.202) but that says nothing about where a gate fires inside
a real generation. This reads what the gate actually did and shows, for every
position it considered, what the model was about to write and how the gate
scored it on its axes.

Two modes:

    python -m experiments.gate_audit        audit events.jsonl written by the
                                            demo server
    python -m experiments.gate_audit --run  generate over a prompt set first
                                            (calls the live endpoint)

The judgement is deliberately left to a human reading the table. A gate that
fires before a number, a name, a carry or a conclusion is doing its job. One
that fires before "the", a markdown bullet or a line break is stopping for
nothing, and no summary statistic will tell you which is which.
"""

import asyncio
import re
import statistics
import sys

from adaptive_thinking import (VLLM, Conversation, Engine, collapse_ws, config, events_log,
                               paths, prompts)

from ._common import collect_turn, rows_from_window, utf8_stdout

utf8_stdout()

# The file the demo server appends to, at the repository root.
EVENTS = paths.EVENTS

PROMPTS = [
    # arithmetic with a carry - the gate should care
    "A train leaves at 14:47 and the journey takes 3 hours 38 minutes. When does "
    "it arrive? Show your working.",
    "A meeting starts at 11:50 and runs for 2 hours 45 minutes. When does it end?",
    # a trap where the famous version has a different answer
    "Three boxes, one has a prize. You pick A. Your friend, who does not know "
    "where the prize is, opens B at random and it is empty. Should you switch to C?",
    # named facts
    "Name three programming languages designed at Bell Labs and say what each was for.",
    # pure prose, no commitments - the gate should mostly stay quiet
    "In two sentences, describe the mood of a seaside town in winter.",
]

# A firing is "cheap" if the token it stopped on carries no commitment: pure
# punctuation, whitespace, markdown furniture, or a bare function word. This is
# a heuristic for sorting the table, not a verdict.
FUNCTION_WORDS = {
    "the", "a", "an", "of", "to", "is", "are", "and", "or", "in", "on", "at",
    "it", "this", "that", "we", "you", "i", "for", "be", "as", "so", "then",
}


def is_cheap(tok):
    t = (tok or "").strip()
    if not t:
        return True
    if not re.search(r"[A-Za-z0-9]", t):
        return True
    return t.lower() in FUNCTION_WORDS


def is_substantive(tok):
    """Numbers, names, operators - the places a wrong commitment is costly."""
    t = (tok or "").strip()
    return bool(re.search(r"\d", t)) or (t[:1].isupper() and len(t) > 1)


async def run_prompts():
    llm = VLLM()
    engine = Engine(llm)
    rows = []
    try:
        for q in PROMPTS:
            ctx = {"q": q, "answer": ""}

            def on_event(ev, q=q, ctx=ctx):
                if ev["type"] == "window":
                    rows.extend(rows_from_window(ev, q, tail=ctx["answer"][-60:]))
                elif ev["type"] == "token":
                    ctx["answer"] += ev["text"]
                elif ev["type"] == "done":
                    print(f"  ran: {q[:58]}", flush=True)

            await collect_turn(engine, Conversation(), q, on_event)
    finally:
        await llm.aclose()
    return rows


def from_events():
    rows, q = [], ""
    for e in events_log.read(EVENTS):
        if e["type"] == "turn":
            q = e.get("q", "")
        elif e["type"] == "window":
            rows.extend(rows_from_window(e, q))
    return rows


def report(rows):
    rows = [r for r in rows if r.get("crit") is not None]
    if not rows:
        print("No gate decisions found. Run the server and ask something, or use --run.\n"
              "(Lines written before the gate moved to axes carry p_yes but no crit,\n"
              " and are skipped - the two numbers are not comparable as one column.)")
        return
    fired = [r for r in rows if r["fired"]]
    passed = [r for r in rows if not r["fired"]]
    print(f"\n{len(rows)} positions considered, {len(fired)} opened a block "
          f"({len(fired) / len(rows):.0%})")
    print(f"  criticality on fired   median {statistics.median([r['crit'] for r in fired]):.3f}"
          if fired else "  nothing fired")
    if passed:
        print(f"  criticality on passed  median {statistics.median([r['crit'] for r in passed]):.3f}")

    # What the threshold is actually buying. This is the table GATE_THRESHOLD
    # is set from. The rows stop at 0.75, so the measured 0.80 that ships in
    # .env.example has no row of its own and gets no "<- now" marker.
    crits = sorted(r["crit"] for r in rows)
    print(f"\n  where the threshold bites (now {config.GATE_THRESHOLD}):")
    for t in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        n = sum(1 for c in crits if c >= t)
        bar = "#" * round(40 * n / len(crits))
        here = "  <- now" if abs(t - config.GATE_THRESHOLD) < 1e-9 else ""
        print(f"    >= {t:.2f}  {n:4d}/{len(crits)} positions {n / len(crits):4.0%} {bar}{here}")

    # Per axis, so a single axis quietly driving every decision is visible.
    have_axes = [r for r in rows if r.get("axes")]
    if have_axes:
        print("\n  per axis, median over every position considered:")
        for a in prompts.GATE_AXES:
            vals = [r["axes"][a] for r in have_axes if a in r["axes"]]
            if vals:
                print(f"    {a:12s} {statistics.median(vals):.3f}   "
                      f"spread {min(vals):.2f}-{max(vals):.2f}  "
                      f"weight {prompts.GATE_AXES[a]['weight']}")

    # The old binary probe, if it was logged alongside. The point is whether it
    # agrees: if it ranks the same positions the same way, the axes bought
    # nothing but a nicer number and GATE_LOG_YESNO can go.
    both = [r for r in rows if r.get("p_yes") is not None]
    if both:
        hi = [r for r in both if r["crit"] >= config.GATE_THRESHOLD]
        lo = [r for r in both if r["crit"] < config.GATE_THRESHOLD]
        print(f"\n  the old yes/no probe, on the same {len(both)} positions:")
        print(f"    median P(yes) where criticality is high  "
              f"{statistics.median([r['p_yes'] for r in hi]):.3f}" if hi else
              "    nothing above the threshold")
        if lo:
            print(f"    median P(yes) where criticality is low   "
                  f"{statistics.median([r['p_yes'] for r in lo]):.3f}")
        disagree = sum(1 for r in both if (r['crit'] >= config.GATE_THRESHOLD)
                       != (r['p_yes'] >= config.PROBE_THRESHOLD))
        print(f"    they disagree on {disagree}"
              f"/{len(both)} positions")

    if fired:
        cheap = [r for r in fired if is_cheap(r["tok"])]
        subst = [r for r in fired if is_substantive(r["tok"])]
        print("\n  of the firings:")
        print(f"    on a number or a name        {len(subst):3d}  "
              f"{len(subst) / len(fired):.0%}   <- stopping for a real commitment")
        print(f"    on punctuation/whitespace/   {len(cheap):3d}  "
              f"{len(cheap) / len(fired):.0%}   <- stopping for nothing")
        print("    a bare function word")
        print(f"    other (content words)        {len(fired) - len(cheap) - len(subst):3d}")

    print("\n  every firing, worst-looking first (judge these yourself):")
    order = sorted(fired, key=lambda r: (not is_cheap(r["tok"]), -r["crit"]))
    for r in order[:24]:
        mark = "cheap" if is_cheap(r["tok"]) else ("number/name" if is_substantive(r["tok"]) else "word")
        tail = collapse_ws(r["tail"])[-46:]
        print(f"    crit={r['crit']:.2f} h={r['h']:4.2f} {mark:11s} {r['tok']!r:14} "
              f"after ...{tail!r}")

    print("\n  highest-scoring positions it did NOT fire on:")
    for r in sorted(passed, key=lambda r: -r["crit"])[:8]:
        print(f"    crit={r['crit']:.2f} h={r['h']:4.2f} {r['tok']!r:14}")

    by_q = {}
    for r in rows:
        by_q.setdefault(r["q"][:52], []).append(r)
    print("\n  per prompt (does it fire where the task is hard?)")
    for q, rs in by_q.items():
        f = sum(1 for r in rs if r["fired"])
        print(f"    {f:2d}/{len(rs):3d} fired  median crit={statistics.median([r['crit'] for r in rs]):.3f}"
              f"  {q}")


async def main():
    if "--run" in sys.argv:
        print(f"generating over {len(PROMPTS)} prompts...", flush=True)
        rows = await run_prompts()
    else:
        rows = from_events()
        print(f"read {EVENTS}")
    report(rows)


if __name__ == "__main__":
    asyncio.run(main())
