"""Run one zebra puzzle through native Qwen or through ours. Grade exactly.

    python -m experiments.zebra.run native N SEED [EFFORT]
    python -m experiments.zebra.run ours   N SEED [EFFORT]

from the repository root, e.g. `python -m experiments.zebra.run ours 4 1`.
EFFORT is low (the default), medium or xhigh - see EFFORT below. Both arms
call the live endpoint and are uncapped: the native arm's max_tokens is the
server's max_model_len less the prompt, and the tree has no token limit.

Puzzles are generated deterministically from (n, seed) and cached to json
next to this file, so both arms get a byte-identical prompt. Each run appends
one line to zebra_results.jsonl here, and the native arm writes its raw stream
to native_raw.txt here as it arrives.
"""
import asyncio
import json
import os
import re
import sys
import time

import httpx

from adaptive_thinking import VLLM, Conversation, Engine, chatml, collapse_ws, config, llm, prompts

from .._common import collect_turn, split_native, utf8_stdout
from . import generate

utf8_stdout()

# This directory: the puzzles are read from here, and the raw native stream
# and the results are written here.
HERE = os.path.dirname(os.path.abspath(__file__))


# Exactly what the server's chat template injects for each level, recovered
# from /tokenize. `reasoning_effort` is nothing more than this sentence in
# front of the system prompt - so it can be applied identically to both arms
# without touching the repo. medium injects nothing. xhigh is the default only
# for chat requests, where the template runs; this repo drives raw
# /completions, where no template runs and no sentence is injected unless it
# is added here. So a run without an effort sentence is equivalent to medium,
# not to maximum effort.
EFFORT = {
    "low": "Reasoning effort is set to low. Keep your thinking brief and "
           "focused, moving directly to the conclusion without unnecessary "
           "elaboration.\n\n",
    "medium": "",
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through "
             "the task, validate key assumptions, consider plausible "
             "alternatives, and prioritize correctness, consistency, and "
             "clarity in the final answer.\n\n",
}


def system_for(effort):
    return EFFORT[effort] + prompts.DEFAULT_SYSTEM


def puzzle(n, seed):
    path = generate.puzzle_path(n, seed, HERE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    sol, cats, clues = generate.generate(n, seed)
    d = generate.puzzle_record(n, seed, sol, cats, clues)
    generate.write_puzzle(path, d)
    return d


def grade(text, d):
    """Exact grid match. Returns (correct, parsed_rows, n_right)."""
    n, cats, sol = d["n"], d["cats"], d["solution"]
    rows = {}
    for m in re.finditer(r"House\s*(\d+)\s*:?\s*([^\n]*)", text, re.I):
        h = int(m.group(1))
        if not 1 <= h <= n:
            continue
        got = {}
        for c in cats:
            mm = re.search(rf"{c}\s*=\s*([A-Za-z]+)", m.group(2), re.I)
            if mm:
                got[c] = mm.group(1).strip().lower()
        if got:
            rows[h] = got          # later mentions win
    right = 0
    for h in range(1, n + 1):
        for c in cats:
            if rows.get(h, {}).get(c) == sol[c][h - 1].lower():
                right += 1
    total = n * len(cats)
    return right == total, rows, f"{right}/{total}"


async def native(prompt_text, system):
    """Qwen thinking for itself, with the whole context window to do it in.

    Its own HTTP clients, so that the request is exactly the one the recorded
    runs sent: the payload below, the timeouts, and the endpoint's identity
    headers (Authorization and User-Agent, as the engine's client sends them).
    """
    prompt = (chatml.turn("system", system)
              + chatml.turn("user", prompt_text)
              + chatml.assistant_header(think=True))
    h = llm.identity_headers()
    async with httpx.AsyncClient(base_url=config.BASE_URL, timeout=60,
                                 headers=h) as c:
        info = (await c.get("/models")).json()["data"][0]
    budget = int(info["max_model_len"]) - (len(prompt) // 2) - 64
    payload = {"model": config.MODEL, "prompt": prompt, "temperature": 0,
               "stop": [chatml.IM_END], "stream": True, "max_tokens": budget}
    out, t0, last, fin = "", time.time(), time.time(), None
    # Write the stream to disk as it arrives. Without this, a run that is
    # stopped leaves nothing behind but a character count, which is why the
    # earlier native failures could not be inspected at all.
    with open(os.path.join(HERE, "native_raw.txt"), "w", encoding="utf-8") as raw:
        async with httpx.AsyncClient(base_url=config.BASE_URL, timeout=7200,
                                     headers=h) as c:
            async with c.stream("POST", "/completions", json=payload) as r:
                async for chunk in llm._iter_sse(r):
                    for ch in chunk.get("choices") or []:
                        piece = ch.get("text") or ""
                        out += piece
                        raw.write(piece)
                        if ch.get("finish_reason"):
                            fin = ch["finish_reason"]
                    if time.time() - last > 90:
                        last = time.time()
                        raw.flush()
                        print(f"    ...{time.time()-t0:5.0f}s {len(out)} chars", flush=True)
    think, answer, closed = split_native(out)
    return answer.strip(), {"secs": round(time.time() - t0, 1), "finish": fin,
                            "closed": closed, "think_chars": len(think),
                            "budget": budget}


async def ours(prompt_text, system):
    client, t0 = VLLM(), time.time()

    def show_block(ev):
        if ev["type"] == "block_close":
            print(f"    [{time.time()-t0:6.1f}s] block {ev['id']} "
                  f"{ev['secs']}s checkpoints={ev['steps']} "
                  f"finished={ev['finished']} explored={ev['explored_tokens']}",
                  flush=True)

    try:
        text, st = await collect_turn(Engine(client), Conversation(system=system),
                                      prompt_text, show_block)
    finally:
        await client.aclose()
    return text, {"secs": st.get("secs"), "blocks": st.get("blocks"),
                  "explored": st.get("explored_tokens"),
                  "restarts": st.get("restarts"), "retries": st.get("retries"),
                  "degraded": st.get("degraded"), "closed": True,
                  "think_chars": None}


async def main():
    arm, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    effort = sys.argv[4] if len(sys.argv) > 4 else "low"
    system = system_for(effort)
    d = puzzle(n, seed)
    print(f"=== {arm}  {n}x{n} seed{seed}  ({d['n_clues']} clues)  effort={effort}")
    text, meta = await (native(d["prompt"], system) if arm == "native"
                        else ours(d["prompt"], system))
    meta["effort"] = effort
    ok, rows, score = grade(text, d)
    print(f"    {meta}")
    print(f"    RESULT {'PASS' if ok else 'FAIL'}   cells {score}")
    if not ok:
        print("    got:      " + json.dumps(rows, sort_keys=True)[:400])
        print("    expected: " + json.dumps(
            {h: {c: d['solution'][c][h-1] for c in d['cats']}
             for h in range(1, n + 1)}, sort_keys=True)[:400])
    print("    --- answer tail ---")
    print("    " + collapse_ws(text)[-500:])
    with open(os.path.join(HERE, "zebra_results.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"arm": arm, "n": n, "seed": seed, "ok": ok,
                             "cells": score, **meta}) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
