"""Run the engine in a terminal. No server, no browser.

    python cli.py "your question"
    python cli.py --json "your question"     raw events, one per line

`python cli.py` is the launcher at the repository root; `python -m
adaptive_thinking.cli` runs the same thing.

Everything the frontend will show is visible here first: the gate's scores,
the rewind, the opening thought and each thinking block's conversations
advancing checkpoint by checkpoint, their axis scores, retirements, and which
thought won.
"""

import sys
# Windows consoles default to cp1252 and the model emits U+2212 and friends.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import asyncio
import json

from . import config, prompts
from .engine import Engine
from .llm import VLLM
from .state import Conversation
from .text import collapse_ws

DIM, RED, GRN, YEL, CYA, OFF = "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"


async def main():
    args = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv
    question = " ".join(args) or (
        "A train leaves at 14:47 and the journey takes 3 hours 38 minutes. "
        "When does it arrive? Show your working."
    )

    llm = VLLM()
    engine = Engine(llm)
    conv = Conversation()
    last_text = {}
    try:
        # The knobs the browser shows (config.summary, served as /config), plus
        # the churn pair, which only the terminal reports.
        s = config.summary()
        print(f"{DIM}entropy floor {s['entropy_floor']}  window {s['window_tokens']}  "
              f"gate>={s['gate_threshold']} on {'/'.join(s['gate_axes'])}  "
              f"beam {s['beam_width']}x{s['think_branch_k']} @ {s['think_chunk']} tok  "
              f"opening thought {'on' if s['open_think'] else 'off'}  "
              f"churn {config.CHURN_MARGIN}/{config.CHURN_PATIENCE}{OFF}\n")
        print(f"{CYA}> {question}{OFF}\n")
        async for ev in engine.respond(conv, question):
            if as_json:
                print(json.dumps(ev, ensure_ascii=False), flush=True)
                continue
            t = ev["type"]
            if t == "token":
                sys.stdout.write(ev["text"])
                sys.stdout.flush()
            elif t == "window":
                top = ev["probed"][0]
                mark = f"{GRN}FIRE{OFF}" if ev["opened"] else f"{DIM}pass{OFF}"
                bits = " ".join(
                    f"{p['tok']!r}:{p['crit']:.2f}" for p in ev["probed"][:4]
                    if p.get("crit") is not None
                )
                # The axis breakdown behind the number that decided it.
                axes = " ".join(f"{k}={top[k]}" for k in prompts.GATE_AXES if k in top)
                print(f"\n{DIM}  [gate {mark} {DIM}best {top['tok']!r} "
                      f"crit={top.get('crit')} ({axes}) | {bits}]{OFF}")
            elif t == "rewind":
                print(f"{YEL}  << rewind {ev['tokens']} tokens{OFF}")
            elif t == "block_open":
                # The opening thought has no tail and no gate score - nothing
                # preceded it and nothing triggered it.
                where = (
                    "before the answer starts"
                    if ev.get("kind") == "open"
                    else f"(crit={ev['crit']}) after \"...{(ev['tail'] or '').strip()[-40:]}\""
                )
                print(f"\n{YEL}  +-- THINKING BLOCK {ev['id']} {where}{OFF}")
                print(f"{DIM}  |   {ev['width']} beams advancing together, "
                      f"{ev['width']}x{ev['branch_k']} candidates per token{OFF}")
            elif t == "think_step":
                # The running conversations, then the frontier in one line. A
                # lineage that ends is printed once, so a dropped conversation
                # is visible rather than silently gone.
                for uid in ev.get("ended", []):
                    txt = last_text.get(uid, "")
                    print(f"{RED}  |  x  #{uid} discontinued: "
                          f"{collapse_ws(txt)[-62:]!r}{OFF}")
                for b in ev["beams"]:
                    last_text[b["uid"]] = b["text"]
                frontier = "  ".join(
                    f"{GRN if b['i'] == 0 else DIM}{b['tok']!r}<{b['parent']}>{OFF}"
                    for b in ev["beams"]
                )
                pool = len(ev.get("pool") or [])
                print(f"{DIM}  |  {ev['step']:2d} [{ev['considered']:2d}->{len(ev['beams'])} live"
                      f" + {pool} finished]{OFF} {frontier}")
                # One out, one in - a retirement never costs the block a slot.
                ent = {e["uid"]: e for e in (ev.get("entered") or [])}
                for i, r in enumerate(ev.get("retired") or []):
                    e = list(ent.values())[i] if i < len(ent) else None
                    where = (f"{GRN}-> #{e['uid']} in, a variation of "
                             f"#{e['from_uid']} at {e['tok']!r}{OFF}" if e else
                             f"{DIM}-> slot closed, {ev.get('dry')} checkpoints "
                             f"without a better thought{OFF}")
                    print(f"{RED}  |  - #{r['uid']} retired ({r['overall']:+.2f}, "
                          f"not a challenger){OFF} {where}")
                if ev["step"] % 4 == 3 or ev["step"] == 0:
                    for b in ev["beams"]:
                        col = GRN if b["i"] == 0 else DIM
                        print(f"{col}  |     #{b['uid']:<2d} {collapse_ws(b['text'])[-70:]!r}{OFF}")
            elif t == "monologue":
                print(f"{DIM}  |   winner: {collapse_ws(ev['text'])[:130]}{OFF}")
            elif t == "block_close":
                # The opening thought replaced no token, so there is nothing it
                # "was about to write".
                was = ("" if ev.get("kind") == "open" else
                       f"{DIM}(was about to write {ev['was_going_to_write']!r})")
                print(f"{YEL}  +-- close: {ev['steps']} checkpoints, "
                      f"{ev['finished']} thoughts finished, "
                      f"{ev['requests']} requests, {ev['secs']}s  {was}{OFF}\n")
            elif t == "frontier":
                warn = "" if ev["started"] >= ev["width"] else (
                    f"  {RED}<- only {ev['started']}, the frontier starts narrow{OFF}")
                print(f"{DIM}  |   fanned out into {ev['started']}/{ev['width']} "
                      f"conversations: "
                      f"{', '.join(repr(b['text']) for b in ev['beams'])}{OFF}{warn}")
            elif t == "standings":
                # raw is what the churn compares; the spread of THESE numbers
                # is what CHURN_MARGIN has to be set from.
                bits = "  ".join(
                    f"#{r['uid']}{'*' if r['done'] else ''}={r['raw']}"
                    for r in ev["ranked"] if r["raw"] is not None
                )
                print(f"{DIM}  |   raw: {bits}{OFF}")
            elif t == "degraded":
                print(f"\n{RED}  !! the endpoint gave out after {ev['kept']} tokens: "
                      f"{ev['error']}{OFF}\n{DIM}     keeping the thinking and the answer "
                      f"so far; it is incomplete{OFF}")
            elif t == "restart":
                print(f"\n{RED}  !! the block made it rewrite the answer from the top "
                      f"({ev['rewrote']} tokens) - note dropped, resuming without it{OFF}")
            elif t == "resumed":
                flag = (f"{GRN}THINKING CHANGED IT{OFF}" if ev["changed"]
                        else f"{DIM}same as before{OFF}")
                print(f"\n{DIM}  [resumed: {ev['was']!r} -> {ev['now']!r}  {flag}]{OFF}")
            elif t == "evict":
                c = ev.get("cache_hit_before")
                c = f"  cache was {c:.1%}." if isinstance(c, float) else ""
                print(f"\n{CYA}  ~~ EVICTED {ev['notes']} thinking blocks "
                      f"(oldest {ev['oldest_age']} tokens behind; kept anything newer than "
                      f"{ev['keep_newer_than']}), {ev['remaining']} left: "
                      f"{ev['before_tokens']} -> {ev['after_tokens']} tokens."
                      f"{c}  Next request re-prefills.{OFF}")
            elif t == "done":
                s = ev["stats"]
                print(f"\n\n{DIM}{'-' * 62}{OFF}")
                print(f"  {s['secs']}s   {s['kept_tokens']} kept   "
                      f"{s['rewound_tokens']} rewound   {s['explored_tokens']} explored")
                opening = " (1 opening)" if s.get("opening_thought") else ""
                print(f"  {s['blocks']} blocks{opening} over {s['think_steps']} checkpoints, "
                      f"{s['blocks_that_changed_next_token']} changed the next token")
                marks = " ".join(f"{m['marker']}@{m['at']}" for m in s.get("spill_markers", []))
                print(f"  spills {s['spills']}{(' [' + marks + ']') if marks else ''}"
                      f"   restarts {s.get('restarts', 0)}"
                      f"   retries {s.get('retries', 0)}"
                      f"   notes now in context {s['notes_in_context']}")
                if s.get("degraded"):
                    print(f"  {RED}INCOMPLETE - {s['degraded']}{OFF}")
                print(f"  {s['requests']} requests   {s['prompt_tokens']} prompt   "
                      f"{s['completion_tokens']} completion tokens")
    finally:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
