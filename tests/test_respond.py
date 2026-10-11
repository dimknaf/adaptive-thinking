"""A whole turn through Engine.respond(): the gate fires, the answer rewinds,
a block runs, its note goes into the context, and the model resumes.

Also covered: the restart backstop, transcript markers in the answer
(spills), age-based eviction at the end of a turn, and the KV-prefix
invariant - the conversation re-renders byte for byte what was last sent, so
the next request starts on a cached prefix.
"""

import asyncio
import json
import math
import time

import pytest

from adaptive_thinking import chatml, config, prompts
from adaptive_thinking.engine import Engine
from adaptive_thinking.state import HEAD_OFF, Conversation, Note, Turn
from adaptive_thinking.text import _whole_sentences, splice
from fake_endpoint import (
    FakeVLLM,
    Stream,
    asked_axis,
    closed,
    ln,
    lvl,
    of_type,
    openings,
    quiet,
    run_turn,
    step,
    thought_of,
    types,
)

QUESTION = "What is 3 + 1?"


def hot(text, p=0.5):
    """A token with two alternatives (entropy ln 2): the gate may stop here."""
    return step(text, top=[(text, ln(p)), (text + "~", ln(1 - p))])


# The first window: " Add" (idx 3) is the only position with any entropy, and
# it starts a new sentence after "First step." (11 characters).
FIRST = quiet("First", " step", ".") + [hot(" Add")] + quiet(" 3", " and", " 1", ".")
BEFORE = "First step."

# Both thoughts close in their first chunk; WINNER is judged best.
THOUGHTS = {"Let": " me check: 3 + 1 = 4.", "Check": " the sum again."}
WINNER = "Let" + THOUGHTS["Let"]
SEEDS = (config.INLINE_SEED, config.OPEN_SEED)


def block_chunks(prompts):
    out = []
    for p in prompts:
        opener = next(o for o in THOUGHTS for s in SEEDS if p.endswith(s + o))
        out.append(closed(THOUGHTS[opener], lp=-2.0))
    return out


def score(prompt):
    family, _ = asked_axis(prompt)
    if family == "gate":
        return lvl(4)
    assert family == "judge", prompt[-200:]
    return lvl(4) if thought_of(prompt) == WINNER else lvl(1)


def inline_note(monologue, anchor):
    """The note as _inline_note builds it: seed, monologue, the fixed closing
    line, `</think>`, then the replayed tail of the answer."""
    return (config.INLINE_SEED + " " + monologue + " " + prompts.NUDGE_PREFIX
            + " Thinking stays in here; the answer itself gets none of it."
            + "</think>" + anchor)


# "First" is dropped from the replay: the tail is cut at its first space.
NOTE = inline_note(WINNER, "step.")


def gate_turn(resume, blocks=1, **script):
    """A fake whose first window fires the gate at " Add"; the answer then
    resumes with `resume`."""
    return FakeVLLM(
        streams=[Stream(FIRST, "length"), Stream(resume, "stop")],
        fanouts=[openings("Let", "Check")] * blocks,
        chunks=block_chunks,
        score=script.pop("score", score),
        **script,
    )


@pytest.fixture
def knobs(cfg):
    return cfg(OPEN_THINK=False, WINDOW_TOKENS=8, WARMUP_TOKENS=2, MIN_GAP_TOKENS=1,
               BEAM_WIDTH=2, THINK_BRANCH_K=2, ENTROPY_FLOOR=0.4,
               GATE_THRESHOLD=0.55, GATE_LOG_YESNO=True, KEEP_BLOCKS_INLINE=True,
               EVICT_AGE_TOKENS=10**9, EVICT_KEEP_TOKENS=10**9)


def prompts_streamed(fake):
    return [c["prompt"] for c in fake.of("stream")]


# ---------------------------------------------------------------------------
# a gate fire, end to end
# ---------------------------------------------------------------------------


def test_a_gate_fire_rewinds_thinks_and_resumes(knobs):
    fake = gate_turn(quiet(" So", " 4", "."))

    events, conv = run_turn(fake, QUESTION)

    assert types(events) == (
        ["start"] + ["token"] * 8
        + ["window", "rewind", "block_open", "frontier", "think_token", "think_token",
           "think_step", "standings", "monologue", "block_close", "resumed"]
        + ["token"] * 3 + ["done"]
    )
    (window,) = of_type(events, "window")
    gate = {"crit": 1.0, "stakes": 1.0, "difficulty": 1.0, "commitment": 1.0, "p_yes": 0.5}
    assert window == {"type": "window", "opened": True,
                      "probed": [{"pos": 3, "tok": " Add", "h": 0.693, **gate}]}
    # Tokens 3..7 were written and are rewound: the block opens where " Add"
    # was about to be written.
    assert of_type(events, "rewind") == [{"type": "rewind", "tokens": 5}]
    (opened,) = of_type(events, "block_open")
    assert (opened["id"], opened["kind"], opened["tail"], opened["crit"], opened["gate"],
            opened["seed"]) == (0, "gate", BEFORE, 1.0, gate, config.INLINE_SEED)
    (close,) = of_type(events, "block_close")
    assert (close["id"], close["steps"], close["finished"], close["winner_score"],
            close["was_going_to_write"]) == (0, 1, 2, 1.0, " Add")
    assert close["requests"] == 3  # fan-out, one chunk, one judge batch
    assert of_type(events, "monologue")[0]["text"] == WINNER
    assert of_type(events, "resumed") == [
        {"type": "resumed", "block": 0, "was": " Add", "now": " So", "changed": True}]

    done = events[-1]
    assert done["text"] == "First step. So 4."
    stats = done["stats"]
    assert stats["blocks"] == 1
    assert stats["rewound_tokens"] == 5
    assert stats["kept_tokens"] == 6            # 3 before the block, 3 after
    assert stats["blocks_that_changed_next_token"] == 1
    assert stats["think_steps"] == 1
    assert stats["notes_in_context"] == 1
    assert (stats["restarts"], stats["spills"], stats["degraded"]) == (0, 0, None)
    assert stats["inline_blocks"] is True and stats["opening_thought"] is False
    assert stats["resume_samples"] == 3
    # 2 streams, gate scores, Yes/No probe, fan-out, chunk, judge scores
    assert stats["requests"] == 7
    assert fake.names() == ["stream", "labels", "answer_probs", "p_yes",
                            "fanout", "chunk", "labels", "answer_probs", "stream"]
    fake.assert_consumed()


def test_the_note_sits_at_the_offset_where_the_gate_fired(knobs):
    fake = gate_turn(quiet(" So", " 4", "."))

    events, conv = run_turn(fake, QUESTION)

    turn = conv.turns[1]
    assert turn.text == "First step. So 4."
    # Offset 11 = len("First step."); born at clock 0 + 3 tokens standing.
    assert turn.notes == [Note(off=len(BEFORE), text=NOTE, born=3)]
    assert [b.kind for b in turn.blocks] == ["gate"]
    assert turn.blocks[0].monologue == WINNER
    assert conv.clock == 6
    # The reader's text holds each word once; the model's holds the note.
    assert splice(turn.text, turn.notes) == BEFORE + NOTE + " So 4."

    prefix = Conversation().render(pending_user=QUESTION, think=False)
    first, second = prompts_streamed(fake)
    assert first == prefix
    assert fake.of("fanout")[0]["prompts"] == [prefix + BEFORE + config.INLINE_SEED]
    assert second == prefix + BEFORE + NOTE


def test_the_next_turn_starts_on_the_prompt_last_streamed(knobs, cfg):
    fake = gate_turn(quiet(" So", " 4", "."))
    _, conv = run_turn(fake, QUESTION)
    last = prompts_streamed(fake)[-1]

    # The finished turn renders exactly what the last request sent, plus the
    # tokens that request produced.
    assert conv.render() == last + " So 4." + chatml.IM_END + "\n"
    assert conv.render(pending_user="And 4 + 1?").startswith(last)

    cfg(ENTROPY_FLOOR=1e9)  # a plain second turn
    fake2 = FakeVLLM(streams=[Stream(quiet("5", "."), "stop")])
    run_turn(fake2, "And 4 + 1?", conv)
    (sent,) = prompts_streamed(fake2)
    assert sent.startswith(last)
    assert conv.render().startswith(sent)


def test_a_resume_on_the_same_token_is_not_counted_as_a_change(knobs):
    fake = gate_turn(quiet(" Add", " 3", "."))

    events, conv = run_turn(fake, QUESTION)

    assert of_type(events, "resumed") == [
        {"type": "resumed", "block": 0, "was": " Add", "now": " Add", "changed": False}]
    stats = events[-1]["stats"]
    assert stats["blocks"] == 1
    assert stats["blocks_that_changed_next_token"] == 0
    assert conv.turns[1].blocks[0].changed_next_token is False
    assert events[-1]["text"] == "First step. Add 3."


def test_with_blocks_not_kept_inline_no_note_is_left(knobs, cfg):
    # The control: the block leaves no trace, so the resumed request is the
    # same prompt the first window continued from.
    cfg(KEEP_BLOCKS_INLINE=False)
    fake = gate_turn(quiet(" Add", " 3", "."))

    events, conv = run_turn(fake, QUESTION)

    prefix = Conversation().render(pending_user=QUESTION, think=False)
    assert prompts_streamed(fake)[1] == prefix + BEFORE
    assert conv.turns[1].notes == []
    assert events[-1]["stats"]["inline_blocks"] is False
    assert events[-1]["stats"]["notes_in_context"] == 0


def test_an_opening_thought_and_a_gate_block_in_one_turn(knobs, cfg):
    cfg(OPEN_THINK=True)
    fake = FakeVLLM(
        streams=[Stream(FIRST, "length"), Stream(quiet(" So", " 4", "."), "stop")],
        fanouts=[openings("Let", "Check"), openings("Let", "Check")],
        chunks=block_chunks,
        score=score,
    )

    events, conv = run_turn(fake, QUESTION)

    closes = of_type(events, "block_close")
    assert [(c["id"], c.get("kind"), c["was_going_to_write"]) for c in closes] == [
        (0, "open", None), (1, None, " Add")]
    assert [e["seed"] for e in of_type(events, "block_open")] == [
        config.OPEN_SEED, config.INLINE_SEED]
    stats = events[-1]["stats"]
    assert stats["blocks"] == 2 and stats["opening_thought"] is True
    # Only the gate block could change a token; the opening thought replaced none.
    assert stats["blocks_that_changed_next_token"] == 1
    assert stats["notes_in_context"] == 2

    turn = conv.turns[1]
    assert turn.notes == [Note(off=HEAD_OFF, text=WINNER, born=0),
                          Note(off=len(BEFORE), text=NOTE, born=3)]
    head = (Conversation().render(pending_user=QUESTION, think=True)
            + chatml.think_head(WINNER))
    first, second = prompts_streamed(fake)
    assert first == head
    assert second == head + BEFORE + NOTE
    assert conv.render().startswith(second)
    fake.assert_consumed()


def test_each_secs_spans_exactly_the_requests_it_counts(knobs, cfg, monkeypatch):
    # The goldens freeze the clock, so they cannot see where a `secs` starts
    # and stops. Here the clock moves one second per request the endpoint is
    # sent, so each `secs` is the number of requests between its two clock
    # reads: it has to equal the `requests` the same event (or the turn's
    # stats) reports, which is counted from usage over the same span.
    cfg(OPEN_THINK=True)
    fake = FakeVLLM(
        streams=[Stream(FIRST, "length"), Stream(quiet(" So", " 4", "."), "stop")],
        fanouts=[openings("Let", "Check"), openings("Let", "Check")],
        chunks=block_chunks,
        score=score,
    )
    monkeypatch.setattr(time, "time", lambda: 1000.0 + fake.usage["requests"])

    events, _ = run_turn(fake, QUESTION)

    # Each block: the fan-out read, one chunk, one judge batch.
    assert [(c.get("kind"), c["secs"], c["requests"]) for c in of_type(events, "block_close")] \
        == [("open", 3.0, 3), (None, 3.0, 3)]
    stats = events[-1]["stats"]
    # The opening block, both answer streams, the gate batch, the Yes/No
    # probe and the gate block.
    assert (stats["secs"], stats["requests"]) == (10.0, 10)
    fake.assert_consumed()


def test_the_opening_thought_is_cut_back_to_its_last_whole_sentence(cfg):
    # The winning thought closed itself on a dangling "So". In the header, that
    # would sit right before `</think>`, the position where the template is
    # strictest, so the note keeps only the whole sentences. The block's own
    # record keeps the full text.
    cfg(OPEN_THINK=True, BEAM_WIDTH=2, THINK_BRANCH_K=2, ENTROPY_FLOOR=1e9,
        WINDOW_TOKENS=8, EVICT_AGE_TOKENS=10**9, EVICT_KEEP_TOKENS=10**9)
    whole = "Let me check: 3 + 1 = 4, since three plus one is four."
    dangling = whole + " So"
    assert _whole_sentences(dangling) == whole

    def chunks(prompts):
        return [closed(dangling[len("Let"):]) if p.endswith(config.OPEN_SEED + "Let")
                else closed(" the sum.") for p in prompts]

    def judge(prompt):
        assert asked_axis(prompt)[0] == "judge", prompt[-200:]
        return lvl(4) if thought_of(prompt).startswith("Let") else lvl(1)

    fake = FakeVLLM(streams=[Stream(quiet("It", " is", " 4."), "stop")],
                    fanouts=[openings("Let", "Check")], chunks=chunks, score=judge)

    events, conv = run_turn(fake, QUESTION)

    assert of_type(events, "monologue")[0]["text"] == dangling
    assert conv.turns[1].blocks[0].monologue == dangling
    assert conv.turns[1].notes == [Note(off=HEAD_OFF, text=whole, born=0)]
    (call,) = fake.of("stream")
    assert call["prompt"] == (
        Conversation().render(pending_user=QUESTION, think=True)
        + chatml.think_head(whole))
    assert conv.render().startswith(call["prompt"])
    fake.assert_consumed()


def test_known_issue_probe_disabled_puts_nan_into_block_open(knobs, cfg):
    # Documents CURRENT behaviour, listed in the README under "Limitations and
    # known issues"; it is not a specification. Without the probe the gate
    # opens at the highest-entropy position and the block's crit is NaN, which
    # json.dumps writes as a bare NaN that strict JSON parsers reject.
    cfg(PROBE_ENABLED=False)

    def judge_only(prompt):
        assert asked_axis(prompt)[0] == "judge"
        return score(prompt)

    fake = gate_turn(quiet(" So", " 4", "."), score=judge_only)

    events, _ = run_turn(fake, QUESTION)

    (window,) = of_type(events, "window")
    assert window["probed"] == [{"pos": 3, "tok": " Add", "h": 0.693, "crit": None}]
    assert window["opened"] is True
    (opened,) = of_type(events, "block_open")
    assert math.isnan(opened["crit"]) and math.isnan(opened["gate"]["crit"])
    assert "NaN" in json.dumps(opened)
    with pytest.raises(ValueError):
        json.dumps(opened, allow_nan=False)
    # Nothing about the gate was asked: no scores, no Yes/No probe.
    assert "p_yes" not in fake.names()
    assert fake.names().count("answer_probs") == 1  # the judge only


# ---------------------------------------------------------------------------
# the restart backstop
# ---------------------------------------------------------------------------

T0 = "The answer depends on whether Bruno is married."  # 47 characters


def test_a_note_that_makes_the_model_restart_the_answer_is_dropped(knobs, cfg):
    cfg(WINDOW_TOKENS=4, WARMUP_TOKENS=1, MIN_GAP_TOKENS=1)
    fake = FakeVLLM(
        streams=[
            Stream([step(T0)] + [hot(" So")] + quiet(" yes", "."), "length"),
            # After the block the model starts the answer over: 48 new
            # characters, the last 40 of which were already written.
            Stream(quiet(" " + T0, " Again"), "length"),
            Stream(quiet(" So", " yes", "."), "stop"),
        ],
        fanouts=[openings("Let", "Check")],
        chunks=block_chunks,
        score=score,
    )

    events, conv = run_turn(fake, QUESTION)

    kinds = types(events)
    i = kinds.index("block_close")
    assert kinds[i:] == ["block_close", "resumed", "token", "restart",
                         "token", "token", "token", "done"]
    assert of_type(events, "restart") == [{"type": "restart", "at": 1, "rewrote": 1}]
    done = events[-1]
    assert done["text"] == T0 + " So yes."
    stats = done["stats"]
    assert stats["restarts"] == 1
    assert stats["restart_detail"] == [{"at": 1, "rewrote": 1}]
    assert stats["blocks"] == 1
    assert stats["kept_tokens"] == 4
    # The gate's rewind only; the rolled-back restart is reported in
    # restart_detail, not here.
    assert stats["rewound_tokens"] == 3
    assert stats["notes_in_context"] == 0
    # The note is gone from the turn and from every later prompt.
    assert conv.turns[1].notes == []
    prefix = Conversation().render(pending_user=QUESTION, think=False)
    first, second, third = prompts_streamed(fake)
    assert second == prefix + T0 + inline_note(WINNER, "answer depends on whether Bruno is married.")
    assert third == prefix + T0
    assert config.INLINE_SEED not in conv.render()
    assert conv.render().startswith(third)
    fake.assert_consumed()


def test_new_text_that_repeats_less_than_the_span_is_not_a_restart(knobs, cfg):
    cfg(WINDOW_TOKENS=4, WARMUP_TOKENS=1, MIN_GAP_TOKENS=1)
    fake = FakeVLLM(
        streams=[
            Stream([step(T0)] + [hot(" So")] + quiet(" yes", "."), "length"),
            # 39 characters of the earlier sentence: one short of the span.
            Stream(quiet(" " + T0[-39:], "."), "stop"),
        ],
        fanouts=[openings("Let", "Check")],
        chunks=block_chunks,
        score=score,
    )

    events, conv = run_turn(fake, QUESTION)

    assert "restart" not in types(events)
    assert events[-1]["stats"]["restarts"] == 0
    assert len(conv.turns[1].notes) == 1


# ---------------------------------------------------------------------------
# transcript markers in the answer
# ---------------------------------------------------------------------------


@pytest.fixture
def plain(cfg):
    # No gate: every token below is quiet.
    return cfg(OPEN_THINK=False, WINDOW_TOKENS=8, EVICT_AGE_TOKENS=10**9)


@pytest.mark.parametrize("last, kept, marker", [
    (" done.<|im_end|>", " done.", "<|im_end|>"),
    (" done.<|im_start|>user\nMore", " done.", "<|im_start|>"),
    (" done.<think>Wait", " done.", "<think>"),
    (" done.</think><|im_end|>", " done.", "</think>"),   # the earliest marker wins
])
def test_a_marker_inside_a_token_is_cut_and_ends_the_answer(plain, last, kept, marker):
    # finish="length": without the marker the engine would ask for another
    # window, and the script has none.
    fake = FakeVLLM(streams=[Stream(quiet("It", " is", last), "length")])

    events, conv = run_turn(fake, QUESTION)

    assert [e["text"] for e in of_type(events, "token")] == ["It", " is", kept]
    done = events[-1]
    assert done["text"] == "It is done."
    assert done["stats"]["spills"] == 1
    assert done["stats"]["spill_markers"] == [
        {"marker": marker, "at": 2, "kept": kept, "empty": False}]
    assert done["stats"]["kept_tokens"] == 3
    assert conv.turns[1].text == "It is done."
    fake.assert_consumed()


def test_a_marker_only_token_is_not_written(plain):
    fake = FakeVLLM(streams=[Stream(quiet("It", " is", " done.", "<|im_end|>"), "stop")])

    events, _ = run_turn(fake, QUESTION)

    assert [e["text"] for e in of_type(events, "token")] == ["It", " is", " done."]
    stats = events[-1]["stats"]
    assert stats["spill_markers"] == [
        {"marker": "<|im_end|>", "at": 3, "kept": "", "empty": True}]
    assert stats["kept_tokens"] == 3
    assert events[-1]["text"] == "It is done."


def test_a_marker_split_across_tokens_is_caught_at_the_end(plain):
    # Neither piece is a marker on its own, so both reach the answer; the last
    # line of defence cuts the assembled marker out and records it.
    fake = FakeVLLM(streams=[Stream(quiet("Done.", "<|im_", "end|>"), "stop")])

    events, conv = run_turn(fake, QUESTION)

    stats = events[-1]["stats"]
    assert stats["spill_markers"] == [
        {"marker": "<|im_end|>", "at": 3, "kept": "", "empty": False}]
    assert events[-1]["text"] == "Done."
    assert conv.turns[1].text == "Done."


# ---------------------------------------------------------------------------
# a window cut short closes its stream at once
# ---------------------------------------------------------------------------


class ClosingFake(FakeVLLM):
    """Records "stream_closed" once an answer stream is finished with: read
    to its end, or closed by the engine. A stream the engine merely abandoned
    would be closed only later, when the event loop finalises it."""

    async def stream(self, prompt, max_tokens, **kw):
        inner = super().stream(prompt, max_tokens, **kw)
        try:
            async for s in inner:
                yield s
        finally:
            await inner.aclose()
            self._record("stream_closed")


def run_watching(fake, user_text):
    """One turn, pairing each event with the endpoint calls made before it
    arrived. The fake never suspends, so nothing the event loop defers (such
    as finalising an abandoned generator) can run in between."""
    conv = Conversation()
    engine = Engine(fake)

    async def go():
        return [(ev, fake.names()) async for ev in engine.respond(conv, user_text)]

    return asyncio.run(go())


def test_a_window_cut_short_by_a_marker_closes_its_stream_at_once(plain):
    # The marker ends the answer with a token still unread. The engine closes
    # the stream there and then, before anything else happens, rather than
    # leaving a suspended stream to the garbage collector.
    fake = ClosingFake(streams=[
        Stream(quiet("It", " is", " done.<|im_end|>", " unread"), "length")])

    seen = run_watching(fake, QUESTION)

    (last_token, open_then), (done, closed_by_then) = seen[-2:]
    assert (last_token["type"], last_token["text"]) == ("token", " done.")
    assert open_then == ["stream"]
    assert (done["type"], done["text"]) == ("done", "It is done.")
    assert closed_by_then == ["stream", "stream_closed"]


def test_a_restart_closes_the_stream_it_broke_out_of_at_once(knobs, cfg):
    # The same turn as the restart test above. The restart is seen on the
    # first token after the block, so " Again" is never read, and the stream
    # is closed before the restart is reported or the next window is asked for.
    cfg(WINDOW_TOKENS=4, WARMUP_TOKENS=1, MIN_GAP_TOKENS=1)
    fake = ClosingFake(
        streams=[
            Stream([step(T0)] + [hot(" So")] + quiet(" yes", "."), "length"),
            Stream(quiet(" " + T0, " Again"), "length"),
            Stream(quiet(" So", " yes", "."), "stop"),
        ],
        fanouts=[openings("Let", "Check")],
        chunks=block_chunks,
        score=score,
    )

    seen = run_watching(fake, QUESTION)

    [names] = [n for ev, n in seen if ev["type"] == "restart"]
    assert names.count("stream") == 2
    assert names[-2:] == ["stream", "stream_closed"]
    assert seen[-1][0]["stats"]["restarts"] == 1
    assert seen[-1][1].count("stream_closed") == 3
    fake.assert_consumed()


# ---------------------------------------------------------------------------
# eviction at the end of a turn
# ---------------------------------------------------------------------------


def aged_conversation():
    """Two past turns whose opening thoughts were written at clock 0 and 95;
    the conversation is now at clock 100."""
    conv = Conversation()
    conv.turns = [
        Turn("user", "Q1"),
        Turn("assistant", "A1.", notes=[Note(HEAD_OFF, "Old thought.", 0)]),
        Turn("user", "Q2"),
        Turn("assistant", "A2.", notes=[Note(HEAD_OFF, "Recent thought.", 95)]),
    ]
    conv.clock = 100
    return conv


def test_old_notes_are_evicted_as_a_band_once_one_is_old_enough(plain, cfg):
    cfg(EVICT_AGE_TOKENS=50, EVICT_KEEP_TOKENS=10)
    conv = aged_conversation()
    fake = FakeVLLM(streams=[Stream(quiet("Yes", "."), "stop")], cache_hit=0.25)

    events, conv = run_turn(fake, "Q3", conv)

    assert types(events)[-2:] == ["evict", "done"]
    (ev,) = of_type(events, "evict")
    # Clock 102: the old note is 102 tokens old (>= 50 triggers, >= 10 goes);
    # the recent one is 7 tokens old and stays.
    after = len(conv.render()) // 4
    before = (len(conv.render()) + len("Old thought.")) // 4
    assert ev == {"type": "evict", "notes": 1, "remaining": 1, "oldest_age": 102,
                  "keep_newer_than": 10, "before_tokens": before,
                  "after_tokens": after, "cache_hit_before": 0.25}
    assert conv.turns[1].notes == []
    assert [n.text for n in conv.turns[3].notes] == ["Recent thought."]
    assert events[-1]["stats"]["notes_in_context"] == 1
    # The evicted turn's header falls back to the empty thinking block, so the
    # prefix the request was sent on no longer matches: eviction's stated cost.
    assert chatml.think_head("Old thought.") not in conv.render()
    (sent,) = prompts_streamed(fake)
    assert chatml.think_head("Old thought.") in sent
    assert not conv.render().startswith(sent)
    assert fake.names() == ["stream", "tokenize", "prefix_cache_stats", "tokenize"]


def test_nothing_is_evicted_before_any_note_reaches_the_trigger_age(plain, cfg):
    cfg(EVICT_AGE_TOKENS=200, EVICT_KEEP_TOKENS=10)
    fake = FakeVLLM(streams=[Stream(quiet("Yes", "."), "stop")])

    events, conv = run_turn(fake, "Q3", aged_conversation())

    assert "evict" not in types(events)
    assert fake.names() == ["stream"]
    assert len(conv.live_notes()) == 2


def test_a_trigger_that_finds_nothing_older_than_the_keep_band_emits_nothing(plain, cfg):
    cfg(EVICT_AGE_TOKENS=50, EVICT_KEEP_TOKENS=1000)
    fake = FakeVLLM(streams=[Stream(quiet("Yes", "."), "stop")])

    events, conv = run_turn(fake, "Q3", aged_conversation())

    assert "evict" not in types(events)
    assert len(conv.live_notes()) == 2
    # Current behaviour: the size and the cache hit rate are read before the
    # engine knows whether anything will go.
    assert fake.names() == ["stream", "tokenize", "prefix_cache_stats"]


def test_a_note_written_in_this_turn_can_be_evicted_at_its_end(knobs, cfg):
    cfg(EVICT_AGE_TOKENS=1, EVICT_KEEP_TOKENS=1)
    fake = gate_turn(quiet(" So", " 4", "."))

    events, conv = run_turn(fake, QUESTION)

    (ev,) = of_type(events, "evict")
    # Born at clock 3, the turn ends at clock 6: 3 tokens old.
    assert (ev["notes"], ev["remaining"], ev["oldest_age"]) == (1, 0, 3)
    assert conv.turns[1].notes == []
    assert events[-1]["stats"]["notes_in_context"] == 0
    assert conv.render().endswith("First step. So 4." + chatml.IM_END + "\n")


@pytest.mark.parametrize("trigger, fires", [(102, True), (103, False)])
def test_eviction_triggers_once_a_note_is_exactly_evict_age_tokens_old(plain, cfg,
                                                                       trigger, fires):
    # The turn ends at clock 102, so the note born at clock 0 is 102 tokens old.
    cfg(EVICT_AGE_TOKENS=trigger, EVICT_KEEP_TOKENS=10)
    fake = FakeVLLM(streams=[Stream(quiet("Yes", "."), "stop")])

    events, conv = run_turn(fake, "Q3", aged_conversation())

    assert conv.clock == 102
    assert ("evict" in types(events)) is fires
    left = [n.text for _, n in conv.live_notes()]
    assert left == (["Recent thought."] if fires else ["Old thought.", "Recent thought."])


@pytest.mark.parametrize("keep, left", [
    (7, []),                    # 7 tokens old, not newer than 7: it goes too
    (8, ["Recent thought."]),
])
def test_a_note_exactly_evict_keep_tokens_old_is_evicted(plain, cfg, keep, left):
    # Current behaviour: a note stays only while it is strictly younger than
    # EVICT_KEEP_TOKENS. At clock 102 the recent note (born 95) is 7 old.
    cfg(EVICT_AGE_TOKENS=50, EVICT_KEEP_TOKENS=keep)
    fake = FakeVLLM(streams=[Stream(quiet("Yes", "."), "stop")])

    events, conv = run_turn(fake, "Q3", aged_conversation())

    (ev,) = of_type(events, "evict")
    assert (ev["notes"], ev["remaining"]) == (2 - len(left), len(left))
    assert [n.text for _, n in conv.live_notes()] == left


def test_a_note_past_the_stripped_answer_is_clamped_to_its_end(knobs):
    # The block opens after "First step. " (with the space) and the model ends
    # its turn straight after it. The answer is stored right-stripped, so the
    # note, written at offset 12, is moved to the new end of the text.
    steps = quiet("First", " step", ".", " ") + [hot("Add")] + quiet(" 3", ".")
    fake = FakeVLLM(
        streams=[Stream(steps, "length"), Stream([], "stop")],
        fanouts=[openings("Let", "Check")],
        chunks=block_chunks,
        score=score,
    )

    events, conv = run_turn(fake, QUESTION)

    turn = conv.turns[1]
    assert turn.text == "First step."
    (note,) = turn.notes
    assert note.text == inline_note(WINNER, "step. ")
    assert note.off == len(turn.text) == 11
    # The prompt the resume was sent on had the space before the note.
    assert prompts_streamed(fake)[1].endswith("First step. " + note.text)
    assert conv.render().endswith("First step." + note.text + chatml.IM_END + "\n")
    fake.assert_consumed()
