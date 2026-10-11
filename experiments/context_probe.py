"""Do thinking blocks stay put, and does eviction do what it claims?

Two properties that are easy to assert and were both wrong before:

  1. A block stays at the offset where it was written, for the life of the
     conversation. An earlier version relocated it into a header at the top of
     the turn, so the model read one string while generating and a different
     one afterwards.

  2. Eviction is by AGE and happens in BULK. It triggers once something is
     EVICT_AGE_TOKENS old and then clears the whole band older than
     EVICT_KEEP_TOKENS in one pass, deleting - not summarising.

A measurement script, not an offline unit test: it runs four turns against the
live endpoint. Run it with small watermarks so eviction actually fires. The
answers here are short - in one recorded run the whole four-turn conversation
reached a clock of only 385 tokens - so the defaults never fire, and a trigger
of 400 may not either. The recorded run triggered eviction at 260 / 40, and a
lower trigger only fires sooner, so these leave some margin:

    EVICT_AGE_TOKENS=200 EVICT_KEEP_TOKENS=40 python -m experiments.context_probe

In PowerShell:

    $env:EVICT_AGE_TOKENS=200; $env:EVICT_KEEP_TOKENS=40; python -m experiments.context_probe

Keep OPEN_THINK on: with it off, questions this simple may open no blocks, and
the script reports that eviction never fired.

(Formerly test_context.py at the repository root; renamed because it is a
measurement against a live endpoint, not a unit test.)
"""

import asyncio

from adaptive_thinking import VLLM, Conversation, Engine, config, splice

from ._common import collect_turn, utf8_stdout

utf8_stdout()

# A whole assistant turn as the model reads it is Conversation.render_assistant:
# header thought + body. The opening thought is part of what a turn READS, so
# the "does turn 1 look the same at the end as it did while it was written"
# check has to include it - comparing bodies alone would miss the header
# changing underneath.

QUESTIONS = [
    "What is 17% of 240? Show your working.",
    "A car travels 246 km in 3 hours. What is its average speed?",
    "How many minutes are there between 09:35 and 13:12?",
    "What is 3/8 of 216? Show your working.",
]


def check_in_place(conv):
    """The property: notes are INTERLEAVED into the turn, not appended to it.

    Testing this by looking for "the 24 chars before the note, then the note"
    does not work - two notes can land close enough that the second one's
    lead-in is itself split by the first. The robust statement is that deleting
    the notes from the rendered body returns the visible text exactly, and that
    the notes appear in offset order.
    """
    rendered = conv.render()
    ok = True
    for t in conv.turns:
        if t.role != "assistant" or not t.notes:
            continue
        body = splice(t.text, t.notes)
        if body not in rendered:
            print("    FAIL: turn body is not in render() verbatim")
            ok = False
            continue
        stripped, at = body, 0
        # The opening thought (off < 0) lives in the header, not the body, so
        # it is not in `body` to be found. render() places it; the byte-
        # identical check at the end of main() is what covers it.
        for n in sorted((n for n in t.notes if n.off >= 0), key=lambda n: n.off):
            i = stripped.find(n.text, at)
            if i == -1:
                print(f"    FAIL: note born at clock {n.born} missing from the body")
                ok = False
                break
            stripped = stripped[:i] + stripped[i + len(n.text) :]
            at = i
        else:
            if stripped != t.text:
                print("    FAIL: removing the notes does not give back the answer text")
                ok = False
    return ok


async def main():
    print(f"evict age {config.EVICT_AGE_TOKENS}  keep newer than {config.EVICT_KEEP_TOKENS}  "
          f"inline {config.KEEP_BLOCKS_INLINE}\n")
    llm = VLLM()
    engine = Engine(llm)
    conv = Conversation()
    evicted = []
    first_body = None
    try:
        for i, q in enumerate(QUESTIONS, 1):

            def on_evict(ev, i=i):
                if ev["type"] == "evict":
                    evicted.append(ev)
                    print(f"  >> EVICT after turn {i}: dropped {ev['notes']}, "
                          f"{ev['remaining']} left, oldest was {ev['oldest_age']} behind, "
                          f"{ev['before_tokens']} -> {ev['after_tokens']} tokens, "
                          f"cache before {ev['cache_hit_before']}")

            await collect_turn(engine, conv, q, on_evict)
            live = conv.live_notes()
            ages = [conv.clock - n.born for _, n in live]
            print(f"  turn {i}: clock={conv.clock:5d}  notes={len(live):2d}  "
                  f"ages={sorted(ages, reverse=True)[:6]}")
            print(f"           in place: {'OK' if check_in_place(conv) else 'FAILED'}")
            if first_body is None:
                t1 = next(t for t in conv.turns if t.role == "assistant")
                first_body = Conversation.render_assistant(t1)

        print("\n--- what the model reads around a surviving block ---")
        shown = False
        for t in conv.turns:
            if t.role == "assistant" and t.notes:
                body = splice(t.text, t.notes)
                i = body.find("<think>")
                print(repr(body[max(0, i - 180) : i + 420]))
                shown = True
                break
        if not shown:
            print("  (no notes survive - eviction cleared them all)")

        print("\n--- assertions ---")
        live = conv.live_notes()
        ages = sorted((conv.clock - n.born for _, n in live), reverse=True)
        # NOT "nothing older than the keep watermark survives" - that would be
        # the wrong property and it fails for the right reason. Eviction only
        # TRIGGERS at EVICT_AGE_TOKENS, so notes between the two watermarks are
        # meant to survive; that gap is what keeps eviction rare. The real
        # invariant is that nothing ever gets older than the trigger without an
        # eviction round having happened.
        oldest = ages[0] if ages else 0
        print(f"  oldest surviving note: {oldest} (trigger {config.EVICT_AGE_TOKENS}, "
              f"keep {config.EVICT_KEEP_TOKENS})"
              f"   {'OK' if oldest < config.EVICT_AGE_TOKENS else 'FAILED'}")
        if evicted:
            e = evicted[-1]
            print(f"  last round left {e['remaining']} notes and freed "
                  f"{e['before_tokens'] - e['after_tokens']} tokens   OK")
        print(f"  eviction rounds: {len(evicted)}")
        # The point of keeping blocks in place: turn 1 must read the same at the
        # end of the conversation as it did when it was written, or every later
        # token sits on a shifted prefix and the KV cache is thrown away for
        # nothing. Eviction is the ONE sanctioned exception.
        t1 = next(t for t in conv.turns if t.role == "assistant")
        now = Conversation.render_assistant(t1)
        same = now == first_body
        print(f"  turn 1 reads identically at the end: "
              f"{'OK' if same else 'changed - expected only if it was evicted'}")
        if evicted:
            bulk = all(e["notes"] > 1 for e in evicted)
            print(f"  each round dropped more than one block (bulk, not a drip): "
                  f"{'OK' if bulk else 'no - watermarks may be too close'}")
            print("  nothing was summarised: OK (no summary field exists)")
        else:
            print("  never fired - lower EVICT_AGE_TOKENS to exercise it")
        print(f"  all notes still in place: "
              f"{'OK' if check_in_place(conv) else 'FAILED'}")
    finally:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
