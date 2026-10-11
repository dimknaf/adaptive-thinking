"""The tree ("ours") vs Qwen's own thinking ("native"), on five logic traps.

KEPT FOR HISTORY ONLY - ITS NUMBERS ARE NOT A LIKE-FOR-LIKE COMPARISON. The
native arm is capped at config.NATIVE_MAX_TOKENS (1600 by default) while the
tree arm is not capped at all, and a native run that reaches the cap before
closing `</think>` is graded as wrong. Both arms must get the same room to
think or the comparison means nothing; experiments/zebra/run.py is the
comparison that does this, giving the native arm the server's max_model_len
less the prompt.

Two further caveats. The `think` column mixes units - characters / 4 for
native, tree checkpoints for ours - so it cannot show whether the two arms had
comparable room. And graded() is a substring match over the last 420
characters with no word boundaries, so a short accepted phrase such as "no"
also matches inside "know" or "cannot".

    python -m experiments.bench_hard    calls the live endpoint; writes
                                        bench_hard.jsonl at the repository root
"""

import asyncio
import json
import re

from adaptive_thinking import Conversation, Engine, UsageMeter, VLLM, chatml, config, paths, prompts

from ._common import collect_turn, split_native, utf8_stdout

utf8_stdout()

# Where the rows go: the repository root, where this script has always
# written them.
OUT = paths.ROOT / "bench_hard.jsonl"

# Each has a wrong reflex answer, and the five right answers differ in KIND so
# no single guess scores well. `looking` and `underdetermined` are a pair: one
# rewards "cannot be determined", the other punishes it.
TASKS = [
    ("monty-ignorant",
     "Three boxes, one hides a prize. You pick box A. Your friend, who does NOT "
     "know where the prize is, opens box B at random and it happens to be empty. "
     "Is switching to C better than staying with A? Give the probability.",
     ["1/2", "50%", "0.5", "equal", "no better", "not better", "no advantage",
      "makes no difference", "does not matter", "doesn't matter", "neither"]),

    ("wason",
     "Four cards show E, K, 4 and 7. Each has a letter on one side and a number "
     "on the other. Someone claims: 'if a card has a vowel on one side, it has an "
     "even number on the other side.' Which cards must you turn over to test it?",
     ["e and 7", "7 and e", "e, 7", "e and the 7", "the e and the 7",
      "vowel and the odd", "e' and '7"]),

    ("syllogism",
     "All the artists in the room are painters. Some of the painters are tall. "
     "Does it follow that some of the artists are tall? Answer yes or no.",
     ["no", "does not follow", "doesn't follow", "not follow", "cannot conclude"]),

    ("looking",
     "Anna is looking at Bruno. Bruno is looking at Carla. Anna is married, Carla "
     "is not, and we do not know about Bruno. Is a married person looking at an "
     "unmarried person?",
     ["yes"]),

    ("underdetermined",
     "X is taller than Y. Z is shorter than X. Who is the shortest of the three?",
     ["cannot be determined", "can't be determined", "not enough information",
      "insufficient information", "cannot tell", "can't tell", "cannot say",
      "impossible to determine"]),
]


def graded(text, ok):
    tail = re.sub(r"\s+", " ", re.sub(r"[*_`#]", "", text[-420:])).lower()
    return any(o in tail for o in ok)


async def native(llm, q):
    """Qwen thinking for itself: no empty <think>, and <think> not a stop word."""
    meter, out = UsageMeter(llm), ""
    prompt = (chatml.turn("system", prompts.DEFAULT_SYSTEM) + chatml.turn("user", q)
              + chatml.assistant_header(think=True))
    async for step in llm.stream(prompt, config.NATIVE_MAX_TOKENS, stop=[chatml.IM_END]):
        out += step.text
    # No closing tag means it ran out of budget mid-reasoning. Grading the
    # working as if it were the answer would score such a run as correct.
    _, answer, _ = split_native(out)
    return answer.strip(), dict(secs=round(meter.secs(), 1),
                                think=(len(out) - len(answer)) // 4,
                                reqs=meter.used()["requests"])


async def ours(llm, q):
    meter = UsageMeter(llm)
    text, stats = await collect_turn(Engine(llm), Conversation(), q)
    return text, dict(secs=round(meter.secs(), 1),
                      think=stats.get("think_steps", 0),
                      reqs=stats.get("requests", 0))


async def main():
    llm, rows = VLLM(), []
    try:
        for tid, q, ok in TASKS:
            for mode, fn in (("native", native), ("ours", ours)):
                text, st = await fn(llm, q)
                rows.append(dict(id=tid, mode=mode, correct=graded(text, ok),
                                 text=text, **st))
                print(f"  {'OK ' if rows[-1]['correct'] else 'BAD'} {mode:7s} "
                      f"{st['secs']:6.1f}s think={st['think']:4d}  {tid}", flush=True)
    finally:
        await llm.aclose()

    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  {'mode':8s} {'correct':>8s} {'secs':>7s} {'think':>7s} {'reqs':>6s}")
    for m in ("native", "ours"):
        rs = [r for r in rows if r["mode"] == m]
        print(f"  {m:8s} {sum(r['correct'] for r in rs):4d}/{len(rs):<3d} "
              f"{sum(r['secs'] for r in rs) / len(rs):7.1f} "
              f"{sum(r['think'] for r in rs) / len(rs):7.0f} "
              f"{sum(r['reqs'] for r in rs) / len(rs):6.0f}")
    print()
    for tid, _, _ in TASKS:
        c = {r["mode"]: r["correct"] for r in rows if r["id"] == tid}
        print(f"    {'OK' if c.get('native') else '..':>3s} {'OK' if c.get('ours') else '..':>4s}   {tid}")


if __name__ == "__main__":
    asyncio.run(main())
