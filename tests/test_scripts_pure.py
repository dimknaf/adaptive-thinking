"""The scripts around the engine: the measurement scripts in experiments/
(bench_hard's grader and arms, gate_audit's two modes, the shared helpers in
experiments/_common.py, tree_probe, context_probe and degraded_check), the
demo server and the CLI.

Nothing here reaches an endpoint. The server, the CLI and the scripts' live
paths are driven by the engine-level FakeVLLM through one scripted turn
(`scripted_turn`) that exercises most event kinds: an opening thought, a gate
window that fires, a rewind, a mid-answer block and the resumed answer.
server.EVENTS, gate_audit.EVENTS and bench_hard.OUT always point into
tmp_path, so the repo's events.jsonl and bench_hard.jsonl are never read or
written.
"""

import asyncio
import json
import math
import runpy
import sys

import pytest
from fastapi.testclient import TestClient

from adaptive_thinking import Conversation, Engine, chatml, cli, config, paths, prompts, server
from experiments import _common, bench_hard, context_probe, degraded_check, gate_audit, tree_probe
from fake_endpoint import (
    Stream,
    asked_axis,
    closed,
    http_error,
    lvl,
    of_type,
    openings,
    quiet,
    run_turn,
    step,
    thought_of,
    types,
)

ln = math.log
QUESTION = "What is 2 + 2?"


# ---------------------------------------------------------------------------
# one scripted turn, used by the server and CLI tests
# ---------------------------------------------------------------------------

# The whole event sequence of scripted_turn(), in order.
SCRIPTED_TYPES = (
    ["start", "block_open", "frontier"] + ["think_token"] * 4
    + ["think_step", "standings", "monologue", "block_close"]
    + ["token"] * 4
    + ["window", "rewind", "block_open", "frontier"] + ["think_token"] * 4
    + ["think_step", "standings", "monologue", "block_close"]
    + ["resumed"] + ["token"] * 3
    + ["done"]
)


def _score(prompt):
    """Every gate axis at its top level (crit 1.0, so the gate fires); the
    judge prefers the thought that opened with "We"."""
    family, _ = asked_axis(prompt)
    if family == "gate":
        return lvl(4)
    assert family == "judge", prompt[-200:]
    return lvl(4) if thought_of(prompt).startswith("We ") else lvl(1)


def scripted_turn(cfg, fake_llm):
    """Knobs and a FakeVLLM for one full turn:

    - the opening thought fans out into four beams, each closes in its first
      chunk, and the judge picks the one that opened with "We";
    - the answer's first window is "Yes. It is 4." with one uncertain
      position (" It", right after a sentence end); the gate fires there, the
      answer rewinds three tokens and a block runs;
    - the answer resumes with " This is 4." and the model ends its turn.
    """
    cfg(OPEN_THINK=True, BEAM_WIDTH=4, THINK_BRANCH_K=5, WINDOW_TOKENS=8,
        WARMUP_TOKENS=0, MIN_GAP_TOKENS=0, ENTROPY_FLOOR=0.4,
        GATE_THRESHOLD=0.55, KEEP_BLOCKS_INLINE=True)
    return fake_llm(
        fanout=lambda prompts, k: openings("Let", "The", "We", "This"),
        chunks=lambda prompts: [closed(" check the sum.", lp=-2.0) for _ in prompts],
        score=_score,
        streams=[
            Stream([step("Yes."), step(" It", top=[(" It", ln(0.6)), (" This", ln(0.4))]),
                    step(" is"), step(" 4.")], finish="stop"),
            Stream(quiet(" This", " is", " 4."), finish="stop"),
        ],
    )


def _without_timings(ev):
    """An event with its wall-clock fields removed, for comparing two runs."""
    ev = {k: v for k, v in ev.items() if k != "secs"}
    if "stats" in ev:
        ev["stats"] = {k: v for k, v in ev["stats"].items() if k != "secs"}
    return ev


def test_the_scripted_turn_runs_as_described(cfg, fake_llm):
    fake = scripted_turn(cfg, fake_llm)
    events, conv = run_turn(fake, QUESTION)

    assert types(events) == SCRIPTED_TYPES
    (window,) = of_type(events, "window")
    assert window["opened"] is True
    assert [(p["pos"], p["tok"], p["crit"]) for p in window["probed"]] == [(1, " It", 1.0)]
    assert of_type(events, "rewind") == [{"type": "rewind", "tokens": 3}]
    assert of_type(events, "resumed") == [{"type": "resumed", "block": 1, "was": " It",
                                          "now": " This", "changed": True}]
    assert [m["text"] for m in of_type(events, "monologue")] == ["We check the sum."] * 2

    done = events[-1]
    assert done["text"] == "Yes. This is 4."
    stats = done["stats"]
    assert (stats["blocks"], stats["blocks_that_changed_next_token"]) == (2, 1)
    assert (stats["rewound_tokens"], stats["kept_tokens"]) == (3, 4)
    assert conv.turns[1].text == "Yes. This is 4."
    fake.assert_consumed()


# ---------------------------------------------------------------------------
# bench_hard.graded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, ok, expected", [
    ("The answer is **No**.", ["no"], True),
    ("It does not follow.", ["does not follow"], True),
    ("It cannot\n\n  be    determined.", ["cannot be determined"], True),
    ("Turn over `E` and `7`.", ["e and 7"], True),
    ("## Answer: _Yes_", ["yes"], True),
    ("Yes, it follows.", ["no", "does not follow"], False),
    ("", ["yes"], False),
])
def test_graded_matches_an_accepted_phrase_in_the_tail(text, ok, expected):
    assert bench_hard.graded(text, ok) is expected


def test_graded_only_reads_the_last_420_characters():
    head = "The answer is yes. "
    assert bench_hard.graded(head + "x" * 420, ["yes"]) is False
    assert bench_hard.graded(head + "x" * 400, ["yes"]) is True


def test_graded_matches_substrings_inside_words():
    # Documents current behaviour, listed in README "Limitations and known
    # issues": the grader has no word boundaries, so "no" is found inside
    # "know" and "cannot".
    assert bench_hard.graded("I don't know.", ["no"]) is True
    assert bench_hard.graded("We cannot say.", ["no"]) is True


def test_bench_hard_tasks_are_well_formed():
    ids = [tid for tid, _, _ in bench_hard.TASKS]
    assert len(ids) == len(set(ids)) == 5
    for tid, question, ok in bench_hard.TASKS:
        assert question.strip() and ok, tid
        assert all(o == o.lower() for o in ok), tid  # graded() lowercases the text only


@pytest.mark.parametrize("tid, reflex", [
    ("monty-ignorant", "Switch: C now wins with probability 2/3."),
    ("wason", "Turn over E and 4."),
    ("looking", "Cannot be determined without Bruno's status."),
    ("underdetermined", "Y is the shortest."),
])
def test_each_trap_rejects_its_reflex_answer(tid, reflex):
    ok = next(o for t, _, o in bench_hard.TASKS if t == tid)
    assert bench_hard.graded(reflex, ok) is False


def _native_prompt(q):
    return (chatml.turn("system", prompts.DEFAULT_SYSTEM) + chatml.turn("user", q)
            + chatml.assistant_header(think=True))


def test_bench_hard_native_answers_after_the_closing_tag(fake_llm):
    fake = fake_llm(streams=[quiet("Hmm", ", four", ".", "</think>", "\n\n", "Yes", ".")])

    answer, st = asyncio.run(bench_hard.native(fake, QUESTION))

    assert answer == "Yes."
    # `think` counts every character before the answer, the tag included, /4.
    assert st["think"] == len("Hmm, four.</think>") // 4
    assert (list(st), st["reqs"]) == (["secs", "think", "reqs"], 1)
    (call,) = fake.of("stream")
    # The native arm keeps its own cap and its own stop list: see the module
    # docstring, which says why its numbers are not a like-for-like comparison.
    assert call["prompt"] == _native_prompt(QUESTION)
    assert call["max_tokens"] == config.NATIVE_MAX_TOKENS
    assert call["stop"] == [chatml.IM_END]


def test_bench_hard_native_without_a_closing_tag_has_no_answer(fake_llm):
    fake = fake_llm(streams=[quiet("Hmm", ", still", " thinking")])

    answer, st = asyncio.run(bench_hard.native(fake, QUESTION))

    assert answer == ""
    assert st["think"] == len("Hmm, still thinking") // 4


def test_bench_hard_ours_reports_the_engine_arm(cfg, fake_llm):
    fake = scripted_turn(cfg, fake_llm)

    text, st = asyncio.run(bench_hard.ours(fake, QUESTION))

    assert text == "Yes. This is 4."
    assert list(st) == ["secs", "think", "reqs"]
    assert st["think"] == 2   # tree checkpoints: one per block here
    assert st["reqs"] == fake.usage["requests"]
    fake.assert_consumed()


def test_bench_hard_main_runs_both_arms_and_writes_one_row_each(
        cfg, fake_llm, monkeypatch, tmp_path, capsys):
    cfg(OPEN_THINK=False, ENTROPY_FLOOR=1e9, WINDOW_TOKENS=8)
    streams = []
    for _ in bench_hard.TASKS:
        streams.append(quiet("Let me see.", "</think>", " No."))   # native
        streams.append(quiet("Yes."))                                # ours
    fake = fake_llm(streams=streams)
    monkeypatch.setattr(bench_hard, "VLLM", lambda: fake)
    out = tmp_path / "bench_hard.jsonl"
    monkeypatch.setattr(bench_hard, "OUT", out)

    asyncio.run(bench_hard.main())

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r["id"], r["mode"]) for r in rows] == [
        (tid, mode) for tid, _, _ in bench_hard.TASKS for mode in ("native", "ours")]
    assert [list(r) for r in rows] == [
        ["id", "mode", "correct", "text", "secs", "think", "reqs"]] * len(rows)
    assert {r["text"] for r in rows} == {"No.", "Yes."}
    by = {(r["id"], r["mode"]): r["correct"] for r in rows}
    assert by[("syllogism", "native")] is True and by[("looking", "ours")] is True
    assert by[("syllogism", "ours")] is False
    printed = capsys.readouterr().out
    assert "  OK  native " in printed and "  BAD ours    " in printed
    assert fake.closed is True
    fake.assert_consumed()


# ---------------------------------------------------------------------------
# gate_audit: token classes, reading events.jsonl, the report
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tok", [None, "", " ", "\n", ",", "**", " -", " The", "the", " So", " I"])
def test_is_cheap(tok):
    assert gate_audit.is_cheap(tok) is True


@pytest.mark.parametrize("tok", ["cat", " 42", " Paris", " minus", "x2"])
def test_is_not_cheap(tok):
    assert gate_audit.is_cheap(tok) is False


@pytest.mark.parametrize("tok", ["42", " 3", " Paris", "x2", " Bell"])
def test_is_substantive(tok):
    assert gate_audit.is_substantive(tok) is True


@pytest.mark.parametrize("tok", [None, "", "I", " A", "the", " minus", ","])
def test_is_not_substantive(tok):
    assert gate_audit.is_substantive(tok) is False


def _write_events(path, records):
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


EVENT_LINES = [
    {"type": "turn", "q": "Q1", "session": "s"},
    "this line is not json",
    {"type": "window", "opened": True, "probed": [
        {"pos": 3, "tok": " 4", "h": 0.9, "crit": 0.82, "p_yes": 0.7,
         "stakes": 1.0, "difficulty": 0.5, "commitment": 0.75},
        {"pos": 5, "tok": " the", "h": 0.5, "crit": 0.4, "stakes": 0.2},
    ]},
    {"type": "window", "opened": False, "probed": [
        {"pos": 9, "tok": ",", "h": 0.6, "crit": 0.3}]},
    {"type": "turn", "q": "Q2"},
    # A line from before the gate moved to axes: p_yes, no crit.
    {"type": "window", "opened": True, "probed": [
        {"pos": 1, "tok": " Paris", "h": 0.7, "p_yes": 0.4}]},
    {"type": "done", "text": "..."},
    "",
]


def test_from_events_reads_every_probed_position(monkeypatch, tmp_path):
    path = tmp_path / "events.jsonl"
    _write_events(path, EVENT_LINES)
    monkeypatch.setattr(gate_audit, "EVENTS", str(path))

    assert gate_audit.from_events() == [
        {"q": "Q1", "tok": " 4", "h": 0.9, "crit": 0.82, "p_yes": 0.7,
         "axes": {"stakes": 1.0, "difficulty": 0.5, "commitment": 0.75},
         "fired": True, "tail": ""},
        {"q": "Q1", "tok": " the", "h": 0.5, "crit": 0.4, "p_yes": None,
         "axes": {"stakes": 0.2}, "fired": False, "tail": ""},
        {"q": "Q1", "tok": ",", "h": 0.6, "crit": 0.3, "p_yes": None,
         "axes": {}, "fired": False, "tail": ""},
        {"q": "Q2", "tok": " Paris", "h": 0.7, "crit": None, "p_yes": 0.4,
         "axes": {}, "fired": True, "tail": ""},
    ]


def test_from_events_without_a_file_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(gate_audit, "EVENTS", str(tmp_path / "missing.jsonl"))
    assert gate_audit.from_events() == []


def test_report_with_nothing_to_report(capsys):
    gate_audit.report([])
    assert "No gate decisions found" in capsys.readouterr().out


def test_report_skips_rows_without_a_crit(capsys):
    gate_audit.report([{"q": "Q", "tok": " x", "h": 0.5, "crit": None, "p_yes": 0.4,
                        "axes": {}, "fired": True, "tail": ""}])
    assert "No gate decisions found" in capsys.readouterr().out


def test_report_summarises_the_gate(cfg, monkeypatch, tmp_path, capsys):
    cfg(GATE_THRESHOLD=0.55)
    path = tmp_path / "events.jsonl"
    _write_events(path, EVENT_LINES)
    monkeypatch.setattr(gate_audit, "EVENTS", str(path))

    gate_audit.report(gate_audit.from_events())
    out = capsys.readouterr().out

    assert "3 positions considered, 1 opened a block (33%)" in out
    assert "where the threshold bites (now 0.55)" in out
    assert out.count("<- now") == 1
    assert ">= 0.55     1/3 positions" in out
    assert "the old yes/no probe, on the same 1 positions" in out
    assert "on a number or a name          1" in out


def test_report_has_no_row_for_the_shipped_threshold(cfg, monkeypatch, tmp_path, capsys):
    # Documents current behaviour: the threshold table stops at 0.75, so the
    # 0.80 that .env.example ships gets no "<- now" marker.
    cfg(GATE_THRESHOLD=0.80)
    path = tmp_path / "events.jsonl"
    _write_events(path, EVENT_LINES)
    monkeypatch.setattr(gate_audit, "EVENTS", str(path))

    gate_audit.report(gate_audit.from_events())
    out = capsys.readouterr().out
    assert "(now 0.8)" in out
    assert "<- now" not in out


def test_gate_audit_main_reads_the_events_file_offline(monkeypatch, tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    _write_events(path, EVENT_LINES)
    monkeypatch.setattr(gate_audit, "EVENTS", str(path))
    monkeypatch.setattr(sys, "argv", ["gate_audit.py"])

    def no_client():
        raise AssertionError("the offline mode must not build a client")

    monkeypatch.setattr(gate_audit, "VLLM", no_client)

    asyncio.run(gate_audit.main())
    out = capsys.readouterr().out
    assert out.startswith(f"read {path}")
    assert "3 positions considered" in out


def test_gate_audit_run_prompts_reads_the_gate_from_a_live_turn(cfg, fake_llm, monkeypatch, capsys):
    fake = scripted_turn(cfg, fake_llm)
    monkeypatch.setattr(gate_audit, "VLLM", lambda: fake)
    monkeypatch.setattr(gate_audit, "PROMPTS", [QUESTION])

    rows = asyncio.run(gate_audit.run_prompts())

    # The same row the events.jsonl route gives, except that a live run knows
    # the answer written so far: the window arrives after four tokens.
    assert rows == [{"q": QUESTION, "tok": " It", "h": 0.673, "crit": 1.0, "p_yes": 0.5,
                     "axes": {"stakes": 1.0, "difficulty": 1.0, "commitment": 1.0},
                     "fired": True, "tail": "Yes. It is 4."}]
    assert capsys.readouterr().out == f"  ran: {QUESTION}\n"
    assert fake.closed is True
    fake.assert_consumed()


def test_gate_audit_rows_match_what_the_server_logs(cfg, fake_llm, monkeypatch, tmp_path):
    """The window events the server writes are what gate_audit reads back."""
    fake = scripted_turn(cfg, fake_llm)
    path = tmp_path / "events.jsonl"
    monkeypatch.setattr(server, "EVENTS", str(path))
    monkeypatch.setattr(server, "_llm", fake)
    monkeypatch.setattr(server, "_sessions", {})
    TestClient(server.app).get("/chat", params={"session": "s", "q": QUESTION})

    monkeypatch.setattr(gate_audit, "EVENTS", str(path))
    rows = gate_audit.from_events()
    assert [(r["q"], r["tok"], r["crit"], r["fired"]) for r in rows] == [
        (QUESTION, " It", 1.0, True)]
    assert rows[0]["axes"] == {"stakes": 1.0, "difficulty": 1.0, "commitment": 1.0}
    assert rows[0]["p_yes"] == 0.5


def test_the_scripts_write_where_they_always_have():
    # The scripts moved from the repository root into experiments/; what
    # they write and read stays at the root.
    assert bench_hard.OUT == paths.ROOT / "bench_hard.jsonl"
    assert gate_audit.EVENTS == paths.EVENTS == paths.ROOT / "events.jsonl"


# ---------------------------------------------------------------------------
# experiments/_common.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out, expected", [
    ("thinking</think>answer", ("thinking", "answer", True)),
    ("a</think> b</think>c", ("a", " b</think>c", True)),
    ("</think>", ("", "", True)),
    ("still thinking", ("still thinking", "", False)),
    ("", ("", "", False)),
])
def test_split_native(out, expected):
    assert _common.split_native(out) == expected


class _Scripted:
    """An engine whose turn is a fixed list of events."""

    def __init__(self, events):
        self.events = events
        self.asked = []

    async def respond(self, conv, q):
        self.asked.append((conv, q))
        for ev in self.events:
            yield ev


def test_collect_turn_returns_the_done_event_and_shows_every_event():
    events = [{"type": "start"}, {"type": "token", "text": "Hi"},
              {"type": "done", "text": "Hi", "stats": {"blocks": 0}}]
    engine, conv, seen = _Scripted(events), object(), []

    got = asyncio.run(_common.collect_turn(engine, conv, "q?", seen.append))

    assert got == ("Hi", {"blocks": 0})
    assert seen == events
    assert engine.asked == [(conv, "q?")]


def test_collect_turn_without_a_done_event():
    engine = _Scripted([{"type": "start"}, {"type": "token", "text": "Hi"}])
    assert asyncio.run(_common.collect_turn(engine, None, "q?")) == ("", {})


def test_rows_from_window_marks_only_the_first_position_of_an_opened_window():
    ev = {"type": "window", "opened": True, "probed": [
        {"pos": 3, "tok": " 4", "h": 0.9, "crit": 0.8, "commitment": 0.5, "stakes": 1.0,
         "extra": 7},
        {"pos": 5, "tok": " the", "h": 0.5, "crit": 0.4, "p_yes": 0.3}]}

    rows = _common.rows_from_window(ev, "Q", tail="so far")

    assert rows == [
        {"q": "Q", "tok": " 4", "h": 0.9, "crit": 0.8, "p_yes": None,
         "axes": {"stakes": 1.0, "commitment": 0.5}, "fired": True, "tail": "so far"},
        {"q": "Q", "tok": " the", "h": 0.5, "crit": 0.4, "p_yes": 0.3,
         "axes": {}, "fired": False, "tail": "so far"},
    ]
    # The axes follow GATE_AXES' order, whatever order the event had.
    assert list(rows[0]["axes"]) == ["stakes", "commitment"]
    assert list(rows[0]) == ["q", "tok", "h", "crit", "p_yes", "axes", "fired", "tail"]


def test_rows_from_window_when_nothing_opened_or_nothing_was_probed():
    ev = {"type": "window", "opened": False, "probed": [
        {"pos": 1, "tok": "a", "h": 0.5, "crit": 0.1},
        {"pos": 2, "tok": "b", "h": 0.5, "crit": 0.0}]}
    assert [r["fired"] for r in _common.rows_from_window(ev, "Q")] == [False, False]
    assert all(r["tail"] == "" for r in _common.rows_from_window(ev, "Q"))
    assert _common.rows_from_window({"type": "window"}, "Q") == []


def test_utf8_stdout_reconfigures_a_text_stream(monkeypatch):
    import io

    stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", stream)
    _common.utf8_stdout()
    assert (stream.encoding, stream.errors) == ("utf-8", "replace")


def test_utf8_stdout_leaves_a_stream_it_cannot_reconfigure(monkeypatch):
    stand_in = object()
    monkeypatch.setattr(sys, "stdout", stand_in)
    _common.utf8_stdout()
    assert sys.stdout is stand_in


# ---------------------------------------------------------------------------
# experiments/tree_probe.py, context_probe.py, degraded_check.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("texts, expected", [
    ([], 0.0),
    (["abc"], 0.0),
    (["abc", "abc"], 0.0),
    (["ab", "cd"], 1.0),
    (["abc", "abd"], 0.5),
    (["ab", "ab", "cd"], 2 / 3),
])
def test_tree_probe_diversity(texts, expected):
    step = {"beams": [{"text": t} for t in texts]}
    assert tree_probe.diversity(step) == pytest.approx(expected)


def test_tree_probe_collect_keeps_the_checkpoints_and_sets_choice_every(cfg, fake_llm):
    cfg(CHOICE_EVERY=0)   # through monkeypatch, so collect()'s write is undone
    fake = scripted_turn(cfg, fake_llm)

    steps, text, stats = asyncio.run(tree_probe.collect(fake, QUESTION, 4))

    assert config.CHOICE_EVERY == 4
    assert [s["type"] for s in steps] == ["think_step", "think_step"]
    assert text == "Yes. This is 4."
    assert stats["blocks"] == 2
    fake.assert_consumed()


def test_context_probe_finds_a_note_in_place(cfg, fake_llm, capsys):
    events, conv = run_turn(scripted_turn(cfg, fake_llm), QUESTION)
    t = conv.turns[1]
    assert [n.off >= 0 for n in t.notes].count(True) == 1   # the mid-answer note

    assert context_probe.check_in_place(conv) is True
    assert capsys.readouterr().out == ""


def test_context_probe_reports_a_body_missing_from_the_render(cfg, fake_llm, capsys):
    events, conv = run_turn(scripted_turn(cfg, fake_llm), QUESTION)
    conv.render = lambda: ""

    assert context_probe.check_in_place(conv) is False
    assert "FAIL: turn body is not in render() verbatim" in capsys.readouterr().out


def test_context_probe_without_notes_has_nothing_to_check(capsys):
    assert context_probe.check_in_place(Conversation()) is True
    assert capsys.readouterr().out == ""


class _Healthy(degraded_check.Stub):
    """The stub with an answer stream that works: nothing degrades."""

    async def stream(self, prompt, max_tokens, **kw):
        from adaptive_thinking import Step

        self.usage["requests"] += 1
        self.last_finish = "stop"
        yield Step("Yes.", -0.1, [("Yes.", -0.1)], "stop")


def test_degraded_check_passes_offline(capsys):
    assert asyncio.run(degraded_check.main()) is True
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert out.rstrip().endswith("RESULT: all good")


def test_degraded_check_reports_a_failed_check(capsys):
    assert asyncio.run(degraded_check.main(_Healthy())) is False
    out = capsys.readouterr().out
    assert "FAIL a degraded event was emitted" in out
    assert out.rstrip().endswith("RESULT: SOMETHING FAILED")


def _run_degraded_check_as_a_script(monkeypatch):
    monkeypatch.delitem(sys.modules, "experiments.degraded_check", raising=False)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("experiments.degraded_check", run_name="__main__")
    return stopped.value.code


def test_degraded_check_script_exits_zero_when_every_check_passes(monkeypatch, capsys):
    assert _run_degraded_check_as_a_script(monkeypatch) == 0
    assert "RESULT: all good" in capsys.readouterr().out


def test_degraded_check_script_exits_non_zero_when_a_check_fails(monkeypatch, capsys):
    import adaptive_thinking

    # The script builds its Engine from the package; this one ignores the
    # script's stub and runs the healthy one, so the checks fail.
    monkeypatch.setattr(adaptive_thinking, "Engine", lambda stub: Engine(_Healthy()))
    assert _run_degraded_check_as_a_script(monkeypatch) == 1
    assert "RESULT: SOMETHING FAILED" in capsys.readouterr().out


def test_importing_degraded_check_runs_nothing(monkeypatch, capsys):
    # It used to run its scenario at import, with no __main__ guard.
    monkeypatch.delitem(sys.modules, "experiments.degraded_check", raising=False)
    namespace = runpy.run_module("experiments.degraded_check", run_name="imported")
    assert callable(namespace["main"])
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# server.py through FastAPI's TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def api(monkeypatch, tmp_path):
    """A TestClient on server.app with its globals isolated: no shared session
    store, no client built yet, and the event log in tmp_path.

    Named `api`, not `app`: the pytest-flask plugin, where installed, hooks
    any fixture called `app` and expects a Flask application."""
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_llm", None)
    monkeypatch.setattr(server, "EVENTS", str(tmp_path / "events.jsonl"))
    return TestClient(server.app)


def sse_events(body):
    out = []
    for block in body.split("\n\n"):
        if block.strip():
            assert block.startswith("data: "), block
            out.append(json.loads(block[len("data: "):]))
    return out


def logged(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_config_endpoint_reports_the_live_knobs(api, cfg):
    cfg(BEAM_WIDTH=3, GATE_THRESHOLD=0.8, OPEN_THINK=False)
    r = api.get("/config")
    assert r.status_code == 200
    assert r.json() == {
        "model": "test-model",
        "window_tokens": config.WINDOW_TOKENS,
        "entropy_floor": config.ENTROPY_FLOOR,
        "gate_threshold": 0.8,
        "gate_axes": ["stakes", "difficulty", "commitment"],
        "probe_enabled": config.PROBE_ENABLED,
        "beam_width": 3,
        "think_branch_k": config.THINK_BRANCH_K,
        "think_chunk": config.THINK_CHUNK,
        "choice_every": config.CHOICE_EVERY,
        "open_think": False,
        "keep_blocks_inline": config.KEEP_BLOCKS_INLINE,
        "evict_age_tokens": config.EVICT_AGE_TOKENS,
        "evict_keep_tokens": config.EVICT_KEEP_TOKENS,
    }


def test_index_serves_the_browser_ui(api):
    r = api.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<html" in r.text.lower()


def test_chat_forwards_the_engine_events_verbatim(api, cfg, fake_llm, monkeypatch, tmp_path):
    fake = scripted_turn(cfg, fake_llm)
    monkeypatch.setattr(server, "_llm", fake)

    r = api.get("/chat", params={"session": "s1", "q": QUESTION})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    events = sse_events(r.text)
    assert events[-1] == {"type": "end"}
    assert types(events[:-1]) == SCRIPTED_TYPES
    fake.assert_consumed()

    # The server adds no logic: the same script run straight through the
    # engine gives the same events, bar wall-clock timings.
    direct, _ = run_turn(scripted_turn(cfg, fake_llm), QUESTION)
    assert [_without_timings(e) for e in events[:-1]] == [_without_timings(e) for e in direct]

    # The turn is kept in the session for the next request.
    conv = server._sessions["s1"]
    assert [t.role for t in conv.turns] == ["user", "assistant"]
    assert conv.turns[1].text == "Yes. This is 4."


def test_chat_logs_the_turn_and_the_auditable_events(api, cfg, fake_llm, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_llm", scripted_turn(cfg, fake_llm))

    events = sse_events(api.get("/chat", params={"session": "s1", "q": QUESTION}).text)

    lines = logged(tmp_path / "events.jsonl")
    assert lines[0]["type"] == "turn"
    assert {k: lines[0][k] for k in ("session", "q", "inline")} == {
        "session": "s1", "q": QUESTION, "inline": True}
    kept = [e for e in events if e["type"] in server._KEEP]
    assert [line["type"] for line in lines[1:]] == [e["type"] for e in kept]
    assert all(line["session"] == "s1" and isinstance(line["t"], float) for line in lines)
    assert {"token", "think_token", "frontier", "start"}.isdisjoint(
        line["type"] for line in lines)
    assert lines[-1]["type"] == "done" and lines[-1]["text"] == "Yes. This is 4."


def test_chat_continues_the_same_session(api, cfg, fake_llm, monkeypatch):
    cfg(OPEN_THINK=False, ENTROPY_FLOOR=1e9, WINDOW_TOKENS=8)
    fake = fake_llm(streams=[quiet("Four."), quiet("Five.")])
    monkeypatch.setattr(server, "_llm", fake)

    api.get("/chat", params={"session": "a", "q": "2 + 2?"})
    api.get("/chat", params={"session": "a", "q": "And 2 + 3?"})

    first, second = fake.of("stream")
    assert "Four." not in first["prompt"]
    assert "2 + 2?" in second["prompt"] and "Four." in second["prompt"]
    assert second["prompt"].endswith(
        chatml.turn("user", "And 2 + 3?") + chatml.assistant_header())
    assert len(server._sessions["a"].turns) == 4


def test_reset_forgets_a_session(api, cfg, fake_llm, monkeypatch):
    cfg(OPEN_THINK=False, ENTROPY_FLOOR=1e9, WINDOW_TOKENS=8)
    monkeypatch.setattr(server, "_llm", fake_llm(streams=[quiet("Four.")]))
    api.get("/chat", params={"session": "a", "q": "2 + 2?"})
    assert "a" in server._sessions

    assert api.post("/reset", json={"session": "a"}).json() == {"ok": True}
    assert "a" not in server._sessions
    # Unknown or missing sessions are fine too.
    assert api.post("/reset", json={"session": "nope"}).json() == {"ok": True}
    assert api.post("/reset", json={}).json() == {"ok": True}


def test_chat_inline_flag_changes_the_setting_for_every_session(api, cfg, fake_llm, monkeypatch, tmp_path):
    # Documents current behaviour, listed in README "Limitations and known
    # issues": the UI's inline toggle writes the process-wide config value.
    cfg(OPEN_THINK=False, ENTROPY_FLOOR=1e9, WINDOW_TOKENS=8, KEEP_BLOCKS_INLINE=True)
    monkeypatch.setattr(server, "_llm", fake_llm(streams=[quiet("Four.")]))

    api.get("/chat", params={"session": "a", "q": "2 + 2?", "inline": 0})

    assert config.KEEP_BLOCKS_INLINE is False
    assert api.get("/config").json()["keep_blocks_inline"] is False
    assert logged(tmp_path / "events.jsonl")[0]["inline"] is False


def test_chat_reports_an_engine_error_in_the_stream(api, cfg, fake_llm, monkeypatch, tmp_path):
    # A 400 is not transient, so the engine raises; the stream carries the
    # error and still ends.
    cfg(OPEN_THINK=False)
    monkeypatch.setattr(server, "_llm", fake_llm(streams=[http_error(400)]))

    events = sse_events(api.get("/chat", params={"session": "a", "q": "2 + 2?"}).text)

    assert types(events) == ["start", "error", "end"]
    assert events[1]["message"].startswith("HTTPStatusError: ")
    assert "400" in events[1]["message"]
    assert "a" in server._sessions and server._sessions["a"].turns == []


def test_log_never_raises(monkeypatch, tmp_path):
    # An unwritable path: the error is swallowed and nothing is created.
    target = tmp_path / "no-such-dir" / "events.jsonl"
    monkeypatch.setattr(server, "EVENTS", str(target))
    server.log({"type": "turn", "kind": "gate"})
    assert not target.parent.exists()


def test_log_writes_a_kind_field_as_data(monkeypatch, tmp_path):
    # log() takes a dict, so an event's own `kind` field is written as data
    # rather than colliding with a parameter name.
    target = tmp_path / "events.jsonl"
    monkeypatch.setattr(server, "EVENTS", str(target))
    server.log({"type": "block_close", "kind": "open", "id": 0})
    server.log({"type": "turn", "q": "café"})
    lines = target.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert all(isinstance(r.pop("t"), float) for r in records)
    assert records == [{"type": "block_close", "kind": "open", "id": 0},
                       {"type": "turn", "q": "café"}]
    assert "café" in lines[1]  # written unescaped (ensure_ascii=False)


def test_shutdown_closes_the_client(cfg, fake_llm, monkeypatch, tmp_path):
    fake = fake_llm()
    monkeypatch.setattr(server, "_llm", fake)
    monkeypatch.setattr(server, "EVENTS", str(tmp_path / "events.jsonl"))
    with TestClient(server.app) as client:
        assert client.get("/config").status_code == 200
        assert fake.closed is False
    assert fake.closed is True


def test_the_client_is_built_once_and_reused(monkeypatch):
    built = []

    class Counting:
        def __init__(self):
            built.append(self)

    monkeypatch.setattr(server, "_llm", None)
    monkeypatch.setattr(server, "VLLM", Counting)
    assert server.llm() is server.llm()
    assert len(built) == 1


# ---------------------------------------------------------------------------
# cli.py
# ---------------------------------------------------------------------------


def run_cli(monkeypatch, fake, *argv):
    monkeypatch.setattr(cli, "VLLM", lambda: fake)
    monkeypatch.setattr(sys, "argv", ["cli.py", *argv])
    asyncio.run(cli.main())


def test_cli_json_mode_prints_every_event_as_one_json_line(cfg, fake_llm, monkeypatch, capsys):
    fake = scripted_turn(cfg, fake_llm)

    run_cli(monkeypatch, fake, "--json", "What", "is", "2", "+", "2?")

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    # Documents current behaviour: --json still prints the settings line and
    # the question, with colour codes, before the events.
    header, events = lines[:2], lines[2:]
    assert header[0].startswith(cli.DIM + "entropy floor")
    assert header[1] == f"{cli.CYA}> {QUESTION}{cli.OFF}"
    parsed = [json.loads(line) for line in events]
    assert types(parsed) == SCRIPTED_TYPES
    assert parsed[0]["user"] == QUESTION
    assert parsed[-1]["text"] == "Yes. This is 4."
    assert fake.closed is True
    fake.assert_consumed()


def test_cli_plain_mode_renders_every_event(cfg, fake_llm, monkeypatch, capsys):
    fake = scripted_turn(cfg, fake_llm)

    run_cli(monkeypatch, fake, QUESTION)

    out = capsys.readouterr().out
    for expected in [
        "opening thought on",
        f"> {QUESTION}",
        "THINKING BLOCK 0 before the answer starts",
        "fanned out into 4/4 conversations",
        "winner: We check the sum.",
        "THINKING BLOCK 1 (crit=1.0) after \"...Yes.\"",
        "<< rewind 3 tokens",
        "was about to write ' It'",
        "[resumed: ' It' -> ' This'",
        "THINKING CHANGED IT",
        "2 blocks (1 opening)",
        "1 changed the next token",
        "spills 0",
    ]:
        assert expected in out, expected
    assert "FIRE" in out and "best ' It' crit=1.0" in out
    assert fake.closed is True


def test_cli_uses_its_default_question_without_arguments(cfg, fake_llm, monkeypatch, capsys):
    cfg(OPEN_THINK=False, ENTROPY_FLOOR=1e9, WINDOW_TOKENS=8)
    fake = fake_llm(streams=[quiet("It", " arrives", " at", " 18:25.")])

    run_cli(monkeypatch, fake)

    out = capsys.readouterr().out
    assert "> A train leaves at 14:47" in out
    assert "It arrives at 18:25." in out
    assert "A train leaves at 14:47" in fake.of("stream")[0]["prompt"]


def test_cli_closes_the_client_when_the_turn_fails(cfg, fake_llm, monkeypatch, capsys):
    cfg(OPEN_THINK=False)
    fake = fake_llm(streams=[http_error(400)])
    with pytest.raises(Exception, match="400"):
        run_cli(monkeypatch, fake, "--json", QUESTION)
    assert fake.closed is True
