"""What the measurement scripts share.

Each script drives the adaptive_thinking package as a client would. These are
the pieces more than one of them needs: a console that can print any text,
running one turn to the end, splitting a native-thinking completion, and
flattening the gate's `window` event into rows.
"""

import sys

from adaptive_thinking import prompts


def utf8_stdout():
    """Print UTF-8 whatever the console's encoding.

    Characters the console cannot show are replaced instead of raising
    UnicodeEncodeError half-way through a report. Where stdout cannot be
    reconfigured (it has been replaced by something that is not a text
    stream), nothing changes.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


async def collect_turn(engine, conv, q, on_event=None):
    """Run one turn to the end and return (text, stats) from its `done` event.

    `on_event`, when given, is called with every event in order, the `done`
    event included. A turn that ends without a `done` event returns
    ("", {}), as each script's own loop did before this was shared.
    """
    text, stats = "", {}
    async for ev in engine.respond(conv, q):
        if on_event is not None:
            on_event(ev)
        if ev["type"] == "done":
            text, stats = ev["text"], ev["stats"]
    return text, stats


def split_native(out):
    """A native-thinking completion as (thinking, answer, closed).

    The answer is everything after the first `</think>`, not stripped. With
    no closing tag the model never finished thinking, so all of `out` is
    thinking and the answer is empty: grading the working as if it were the
    answer would score such a run as correct.
    """
    if "</think>" in out:
        think, answer = out.split("</think>", 1)
        return think, answer, True
    return out, "", False


def rows_from_window(ev, q, tail=""):
    """One row per position the gate considered in a `window` event.

    Only the first probed position can have fired: the gate lists them by
    criticality, best first, and opens its block on the best. A row keeps the
    per-axis scores under `axes`; `tail` is the answer written before the
    window, which only a live run knows.
    """
    return [{"q": q, "tok": p["tok"], "h": p["h"],
             "crit": p.get("crit"), "p_yes": p.get("p_yes"),
             "axes": {k: p[k] for k in prompts.GATE_AXES if k in p},
             "fired": bool(ev.get("opened")) and i == 0, "tail": tail}
            for i, p in enumerate(ev.get("probed", []))]
