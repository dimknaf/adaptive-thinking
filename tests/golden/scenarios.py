"""The scripted situations the goldens record. See harness.py for the format.

Each scenario is a function (tmp_dir) -> recording, listed in SCENARIOS with a
one-line description. FEATURES (at the bottom) says what each recording is
there to exercise; tests/test_golden.py checks the stored goldens still do,
so a regenerated golden cannot quietly stop covering something.

Path specs, briefly: "|" separates answer tokens, "‖" marks where a thinking
note sits (the path is what the model writes once that note is in its
context), and a token prefixed "^" is where the gate fires, "~" where it
scores just under the threshold.
"""

import json

from .harness import (
    CLOSED,
    GENERIC_OPENINGS,
    BlockScript,
    Chunk,
    Fail,
    Model,
    TurnScript,
    alts,
    classify,
    knobs,
    parse_prompt,
    record_cli,
    record_engine,
    record_server,
    split_question,
)

# ---------------------------------------------------------------------------
# Thinking blocks
# ---------------------------------------------------------------------------

# Five checkpoints. The fan-out's first read is truncated junk and the retry
# is fine. One thought closes with `</think>` at once. A clearly weaker one is
# retired while the block is still improving, and a variation of the best
# branch takes its slot (and closes off script). Two checkpoints without a
# better thought later, another laggard is retired and its slot CLOSED. The
# leader writes on alone and closes last.
RICH = BlockScript(
    first_read=[("Let", -0.6), ("!", -1.1)],
    openings=GENERIC_OPENINGS,
    lineages={
        "Let": [
            Chunk(" me work it out step by step: 14:47 plus 3 hours is 17:47.", -3.1,
                  alts=alts(".", " now", ",", " first", " then")),
            Chunk(" Then I add the 38 minutes: 47 plus 38 is 85.", -2.9,
                  alts=alts(".", ",", " minutes", " here", " so")),
            Chunk(" 85 minutes is 1 hour and 25 minutes.", -2.2,
                  alts=alts(".", ",", " over", " extra", " then")),
            Chunk(" So 17:47 plus 38 minutes is 18:25.", -2.4,
                  alts=alts(".", ",", " exactly", " then", " so")),
            Chunk(" The train arrives at 18:25.</think>", -1.6, alts=CLOSED, finish="stop"),
        ],
        "First": [
            Chunk(" convert everything to minutes: 887 plus 218 is 1105.</think>", -3.8,
                  alts=CLOSED, finish="stop"),
        ],
        "We": [
            Chunk(" need the sum of 14:47 and 3:38.", -2.6,
                  alts=alts(".", ",", " first", " now", " here")),
            Chunk(" The hours give 17 and the minutes give 85.", -2.8,
                  alts=alts(".", ",", " total", " then", " so")),
            Chunk(" Hmm, 85 minutes, so maybe 17:85 is fine.", -4.2,
                  alts=alts(".", "?", ",", " too", " then")),
            Chunk(" I am not sure about the carry here.", -3.7,
                  alts=alts(".", ",", "?", " yet", " then")),
        ],
        "So": [
            Chunk(" the the answer is the answer", -6.2,
                  alts=alts(" answer", " is", " the", ".", " so")),
            Chunk(" is is the the answer", -6.8,
                  alts=alts(" answer", " is", " the", ".", " so")),
        ],
    },
    grades={
        "Let": [4, 4, 4, 4, 4],
        "First": [3],
        "We": [(4, 3, 4, 4), (4, 3, 4, 4), (2, 2, 2, 2), (2, 2, 2, 2)],
        "So": [1, 1],
    },
)

# Two checkpoints. Both fan-out reads are narrow, so the block starts two
# conversations wide and backfills: one lineage contributes two survivors and
# the second is split off under a new id. An alternative last token that is
# `</think>` finishes a twin on the spot. The stop rule ends the block while
# one conversation is still writing, out of contention.
NARROW = BlockScript(
    first_read=[("Let", -0.9), ("!", -1.3), ("\n", -1.6)],
    openings=[("Let", -0.75), ("We", -1.4), ("!", -2.2), (" ", -2.6), ("</think>", -3.0),
              ("...", -3.6)],
    lineages={
        "Let": [
            Chunk(" me add 3 hours to 14:47 first, which gives 17:47.", -3.2,
                  alts=alts(".", " now", ",", " then", "</think>")),
            Chunk(" Then 38 minutes on top of 17:47 gives 18:25.</think>", -2.8,
                  alts=CLOSED, finish="stop"),
        ],
        "We": [
            Chunk(" can also count in minutes: 47 plus 38 is 85.", -3.6,
                  alts=alts(".", ",", " so", " and", " then")),
            Chunk(" 85 minutes is 1 hour 25, so the hours become 14 plus 3 plus 1, which is"
                  " 18, and the time is 18:25", -5.5,
                  alts=alts("25", "25.", " 25", "26", "24")),
        ],
    },
    grades={"Let": [4, 4], "We": [3, 2]},
)

# One checkpoint, and every way a thought can finish. One copies the closing
# line of a note ("Right - back") and is cut there, which finishes both of its
# variations; one ends with no alternatives at all; one closes with
# `</think>`. The last quotes the first question, which is in its context, so
# its natural ending pays the copy penalty and a variation survives instead.
ENDINGS = BlockScript(
    openings=[("Let", -1.0), ("The", -1.5), ("We", -1.9), ("So", -2.3), ("!", -2.7), ("\n", -3.1)],
    lineages={
        "Let": [Chunk(" me see: 14:47 plus 25 minutes is 15:12. Right - back to the answer", -3.0,
                      alts=alts(" answer", " question"))],
        "The": [Chunk(" delay moves the departure to 15:12.", -2.0, no_top=True)],
        "We": [Chunk(" add 25 minutes, since the journey takes 3 hours 38 minutes.", -3.5,
                     alts=alts(".", ",", " anyway"))],
        "So": [Chunk(" it arrives 25 minutes later.</think>", -1.8, alts=CLOSED, finish="stop")],
    },
    grades={"Let": [3], "The": [2], "We": [4], "So": [3]},
)

# Two checkpoints. Two short thoughts finish at once. One conversation runs on
# far past them and scores badly, so it is dropped as rambling; the other
# finishes on the second checkpoint and wins.
RAMBLE = BlockScript(
    openings=GENERIC_OPENINGS,
    lineages={
        "Let": [Chunk(" 180 plus 38 is 218.</think>", -1.2, alts=CLOSED, finish="stop")],
        "First": [Chunk(" 3 times 60 is 180, then add 38.</think>", -1.5, alts=CLOSED,
                        finish="stop")],
        "We": [
            Chunk(" should double-check: 3 hours is 180 minutes.", -2.4,
                  alts=alts(".", ",", " exactly", " so", " then")),
            Chunk(" Adding 38 gives 218 minutes.</think>", -1.9, alts=CLOSED, finish="stop"),
        ],
        "So": [Chunk(" I think I should first reconsider what exactly the question is asking"
                     " about the journey and the minutes and the hours and whether", -9.0,
                     alts=alts(" whether", " if", " how", " what", " why"))],
    },
    grades={"Let": [3], "First": [3], "We": [4, 4], "So": [1]},
)

# The fan-out finds only two real openings, twice.
NARROW_START = BlockScript(openings=[("Let", -0.7), ("!", -1.2), ("We", -1.6), (" ", -2.0),
                                     ("</think>", -2.5)])

# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------

Q1 = "A train leaves at 14:47 and the journey takes 3 hours 38 minutes. When does it arrive?"
T1_HEAD = "The| journey| is| 3| hours| and| 38| minutes|."
T1_BASE = (T1_HEAD + "|^ Adding| 3| hours| to| 14|:|47| gives| 17|:|47|.| Adding| 38| minutes|"
           " gives| 18|:|25|.| So| the| train| arrives| at| 18|:|25|.")
T1_AFTER = (T1_HEAD + "‖ First|,| 3| hours| after| 14|:|47| is| 17|:|47|.| Then| 38| more|"
            " minutes| gives| 18|:|25|.")

Q2 = "And if it is delayed by 25 minutes?"
T2_HEAD = "It| leaves| at| 15|:|12| instead|,| so| it| arrives| 25| minutes| later|."
# Ends its turn in the same chunk as its last word.
T2_BASE = (T2_HEAD + "|^ The| new| arrival| is| 18|:|50|.| That| is| still| before| 19|:|00|"
           ".<|im_end|>")
# After thinking it starts the answer over: what the restart check is for.
T2_RESTART = (T2_HEAD + "‖ It| leaves| at| 15|:|12| instead|,| so| it| arrives| 25| minutes|"
              " later|.| The| new| arrival| is| 18|:|50|.")

Q3 = "How long is the journey in minutes?"
T3_HEAD = ("The| journey| itself| does| not| change|.| It| is| 3| hours| and| 38| minutes|.|~ That|"
           " is| 180| plus| 38|.")
# It opens its own think block mid-answer, in the window where the gate fires.
T3_BASE = T3_HEAD + "|^ So| the|<think>| total| is| 218|."
T3_AFTER = T3_HEAD + "‖ That| makes| 218| minutes| in| total|."

CONVERSATION = Model(
    TurnScript(Q1, T1_BASE, T1_AFTER, blocks={"open": NARROW}),
    TurnScript(Q2, T2_BASE, T2_RESTART, blocks={"open": ENDINGS}),
    TurnScript(Q3, T3_BASE, T3_AFTER, blocks={"38.": RAMBLE}),
    stream_usage=True,
)

PLAIN = Model(TurnScript(Q1, T1_BASE, blocks={"38 minutes.": RICH}))

# The server's error path: an answer request the endpoint rejects outright
# (a 400 is not transient, so the engine raises instead of degrading).
QG_ERR = "What is 23:50 plus 20 minutes?"
SERVER = Model(
    TurnScript(Q1, T1_BASE, blocks={"38 minutes.": RICH}),
    TurnScript(QG_ERR, "It| is| 00|:|10|."),
    failures=[Fail("stream", 400, times=None,
                   when=lambda body: parse_prompt(body["prompt"]).question == QG_ERR)],
)

QC1 = "What time is 3 hours 38 minutes after 14:47?"
QC2 = "And 25 minutes after that?"
DEGRADED = Model(
    TurnScript(QC1, "The| answer| is| 18|:|25|.", blocks={"open": NARROW_START}),
    TurnScript(QC2, "That| is| 18|:|50|."),
    failures=[Fail("stream", 524, when=lambda body: parse_prompt(body["prompt"]).question == QC1,
                   times=None)],
)

QD = "What is 9:50 plus 25 minutes?"
TD_HEAD = "Adding| 25| minutes| to| 9|:|50| gives| 10|:|15|."
TD_BASE = TD_HEAD + "|^ The| hour| rolls| over| once|."
TD_AFTER = TD_HEAD + "‖ So| the| answer| is| 10|:|15|."


def _gate_batch_at(vis):
    """The gate batch that scores the position right after `vis`."""
    def when(body):
        for p in body["prompt"]:
            ctx, q = split_question(p)
            if q is not None and classify(q)[0] == "gate" and parse_prompt(ctx).vis == vis:
                return True
        return False
    return when


RETRIES = Model(
    TurnScript(QD, TD_BASE, TD_AFTER),
    failures=[
        Fail("stream", 503, when=lambda body: parse_prompt(body["prompt"]).vis == ""),
        Fail("probs", 503, when=_gate_batch_at("Adding 25 minutes to 9:50 gives 10:15.")),
        Fail("fanout", 503),
        Fail("chunk", 502),
    ],
)

QE = "What is 11:40 plus 35 minutes?"
TE_HEAD = "Adding| 35| minutes| to| 11|:|40| gives| 12|:|15|."
TE_BASE = TE_HEAD + "|^ The| hour| changes| from| 11| to| 12|."
TE_AFTER = TE_HEAD + "‖ So| it| is| 12|:|15|."
GATE_ONLY = Model(TurnScript(QE, TE_BASE, TE_AFTER))

# ---------------------------------------------------------------------------
# Knobs per scenario
# ---------------------------------------------------------------------------

# Eviction fires after the third turn of CONVERSATION and not before: only
# then is the first turn's opening thought EVICT_AGE_TOKENS old.
KNOBS_CONVERSATION = knobs(OPEN_THINK=True, KEEP_BLOCKS_INLINE=True, GATE_LOG_YESNO=True,
                           EVICT_AGE_TOKENS=80, EVICT_KEEP_TOKENS=40)
KNOBS_PLAIN = knobs(OPEN_THINK=False, KEEP_BLOCKS_INLINE=False, GATE_LOG_YESNO=False)
KNOBS_DEGRADED = knobs(OPEN_THINK=True)
KNOBS_RETRIES = knobs(OPEN_THINK=False, GATE_LOG_YESNO=False)
# One turn long enough to age its own notes out.
KNOBS_ONE_TURN_EVICT = knobs(OPEN_THINK=True, KEEP_BLOCKS_INLINE=True, GATE_LOG_YESNO=False,
                             EVICT_AGE_TOKENS=20, EVICT_KEEP_TOKENS=10)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def a_conversation(tmp_dir):
    return record_engine(CONVERSATION, [Q1, Q2, Q3], KNOBS_CONVERSATION)


def b_plain_turn(tmp_dir):
    return record_engine(PLAIN, [Q1], KNOBS_PLAIN)


def c_degraded(tmp_dir):
    return record_engine(DEGRADED, [QC1, QC2], KNOBS_DEGRADED)


def d_retries(tmp_dir):
    return record_engine(RETRIES, [QD], KNOBS_RETRIES)


def e_gate_log_yesno(tmp_dir):
    return {
        "on": record_engine(GATE_ONLY, [QE], knobs(OPEN_THINK=False, GATE_LOG_YESNO=True)),
        "off": record_engine(GATE_ONLY, [QE], knobs(OPEN_THINK=False, GATE_LOG_YESNO=False)),
    }


def f_cli(tmp_dir):
    return {
        "plain_turn": record_cli(PLAIN, Q1, KNOBS_PLAIN),
        "degraded": record_cli(DEGRADED, QC1, KNOBS_DEGRADED, modes=("formatted",)),
        "restart_evict": record_cli(CONVERSATION, Q2, KNOBS_ONE_TURN_EVICT,
                                    modes=("formatted",)),
    }


def g_server(tmp_dir):
    return record_server(SERVER, [("golden", Q1), ("golden-error", QG_ERR)], KNOBS_PLAIN,
                         tmp_dir)


def h_probe_disabled(tmp_dir):
    return record_engine(GATE_ONLY, [QE], knobs(OPEN_THINK=False, PROBE_ENABLED=False))


SCENARIOS = {
    "a_conversation": (
        a_conversation,
        "Three turns with the opening thought and inline notes: gate fires and near "
        "misses, a restart, spills at the end of a turn and mid-answer, eviction after "
        "the third turn.",
    ),
    "b_plain_turn": (
        b_plain_turn,
        "One turn, no opening thought, notes not kept: a five-checkpoint block with churn "
        "(a slot refilled, later one closed); the next token cannot change.",
    ),
    "c_degraded": (
        c_degraded,
        "A 524 on every answer request after the opening thought: retried, then the turn "
        "degrades and keeps its thinking; the next turn carries on.",
    ),
    "d_retries": (
        d_retries,
        "One transient failure on each request shape (answer stream, gate batch, fan-out "
        "read, chunk), each retried once.",
    ),
    "e_gate_log_yesno": (
        e_gate_log_yesno,
        "The same turn with the logged Yes/No probe on and off.",
    ),
    "f_cli": (
        f_cli,
        "cli.py's stdout: the plain turn (--json and formatted), the degraded turn, and a "
        "turn with a restart, spills and eviction.",
    ),
    "g_server": (
        g_server,
        "server.py: /config, the /chat SSE body for the plain turn and for a turn the "
        "endpoint rejects, events.jsonl, /reset.",
    ),
    "h_probe_disabled": (
        h_probe_disabled,
        "PROBE_ENABLED=false: entropy alone opens every block, and crit is NaN on the wire.",
    ),
}

# ---------------------------------------------------------------------------
# What each golden is there to exercise
# ---------------------------------------------------------------------------


def events_of(run):
    """Every event of an engine recording, parsed."""
    return [json.loads(line) for t in run["turns"] for line in t["events"]]


def _types(evs):
    return {e["type"] for e in evs}


def _steps(evs):
    return [e for e in evs if e["type"] == "think_step"]


def _stats(evs):
    return [e["stats"] for e in evs if e["type"] == "done"]


def _requests(run):
    return [r for t in run["turns"] for r in t["requests"]]


FEATURES = {
    "a_conversation": {
        "every engine event type": lambda r: _types(events_of(r)) == {
            "start", "block_open", "frontier", "think_token", "think_step", "standings",
            "monologue", "block_close", "token", "window", "rewind", "resumed", "restart",
            "evict", "done"},
        "a narrow fan-out, read twice": lambda r: any(
            e["type"] == "frontier" and e["started"] < e["width"] for e in events_of(r)),
        "backfill and a split lineage": lambda r: any(s["backfilled"] for s in _steps(events_of(r))),
        "a gate fire and a near miss": lambda r: any(
            w["type"] == "window" and w["opened"] for w in events_of(r)) and any(
            w["type"] == "window" and not w["opened"]
            and 0.5 < w["probed"][0]["crit"] < 0.6 for w in events_of(r)),
        "a restart": lambda r: any(s["restarts"] for s in _stats(events_of(r))),
        "a spill at the end and mid-answer": lambda r: [
            (m["marker"], m["empty"]) for s in _stats(events_of(r)) for m in s["spill_markers"]
        ] == [("<|im_end|>", False), ("<think>", True)],
        "the next token changed": lambda r: any(
            s["blocks_that_changed_next_token"] for s in _stats(events_of(r))),
        "eviction on the third turn only": lambda r: [
            any(json.loads(x)["type"] == "evict" for x in t["events"]) for t in r["turns"]
        ] == [False, False, True],
        "the Yes/No probe": lambda r: any(
            "p_yes" in p for e in events_of(r) if e["type"] == "window" for p in e["probed"]),
    },
    "b_plain_turn": {
        "a refill and a closed slot": lambda r: [
            x["replaced"] for s in _steps(events_of(r)) for x in s["retired"]] == [True, False],
        "a degenerate first fan-out read": lambda r: sum(
            1 for q in _requests(r)
            if q["body"] and q["body"].get("stream") and q["body"].get("max_tokens") == 1) == 2,
        "five checkpoints": lambda r: len(_steps(events_of(r))) == 5,
        "the next token did not change": lambda r: [
            e["changed"] for e in events_of(r) if e["type"] == "resumed"] == [False],
    },
    "c_degraded": {
        "degraded after three attempts": lambda r: (
            [e["kept"] for e in events_of(r) if e["type"] == "degraded"] == [0]
            and r["usage"]["retries"] == 2),
        "the next turn completes": lambda r: r["turns"][1]["error"] is None and _stats(
            events_of(r))[1]["degraded"] is None,
    },
    "d_retries": {
        "one retry per request shape": lambda r: r["usage"]["retries"] == 4,
    },
    "e_gate_log_yesno": {
        "p_yes only when on": lambda r: (
            any("p_yes" in p for e in events_of(r["on"]) if e["type"] == "window"
                for p in e["probed"])
            and not any("p_yes" in p for e in events_of(r["off"]) if e["type"] == "window"
                        for p in e["probed"])),
    },
    "f_cli": {
        "every mode ran cleanly": lambda r: all(
            m["error"] is None for run in r.values() if isinstance(run, dict)
            for m in run.values() if isinstance(m, dict) and "stdout" in m),
        "retire, discontinue, restart and evict lines": lambda r: all(
            s in r["plain_turn"]["formatted"]["stdout"] + r["restart_evict"]["formatted"]["stdout"]
            for s in ("retired", "discontinued", "slot closed", "rewrite the answer", "EVICTED")),
        "a narrow frontier and a degraded turn": lambda r: all(
            s in r["degraded"]["formatted"]["stdout"]
            for s in ("the frontier starts narrow", "the endpoint gave out", "INCOMPLETE")),
    },
    "g_server": {
        "every stream ends with end": lambda r: all(
            c["body"].endswith('data: {"type": "end"}\n\n') for c in r["chats"]),
        "an engine error reaches the stream": lambda r: [
            '"type": "error"' in c["body"] for c in r["chats"]] == [False, True],
        "events.jsonl was written": lambda r: bool(r["events_jsonl"]),
    },
    "h_probe_disabled": {
        "NaN crit on the wire": lambda r: any(
            '"crit": NaN' in line for t in r["turns"] for line in t["events"]),
    },
}
