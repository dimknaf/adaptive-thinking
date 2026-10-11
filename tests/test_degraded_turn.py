"""An endpoint failure part-way through a turn degrades it instead of losing it.

Ported from the script experiments/degraded_check.py (formerly test_degraded.py
at the repository root), which prints OK/FAIL lines instead of asserting. The
scenario is the one that motivated the degraded path: the opening thought
completes, then the answer request fails with a 524 gateway timeout. The turn
must finish, say that it is incomplete, and keep the thinking - in the
conversation and in the prompt the next request will send.
"""

import httpx
import pytest

from adaptive_thinking import chatml, config
from adaptive_thinking.state import Conversation
from adaptive_thinking.text import head_note
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
    thought_of,
    types,
)

QUESTION = "Is a married person looking at an unmarried person?"

# How each of the four opening beams continues. Every thought closes itself in
# its first chunk, so the block is one checkpoint long.
THOUGHTS = {
    "Let": " me see. Jack looks at Anne, so maybe.",
    "The": " question cannot be settled without Anne's status.",
    "We": " need both cases. If Anne is married, she looks at George; if not,"
          " Jack looks at her. Either way, yes.",
    "This": " is a trick question.",
}
WINNER = "We" + THOUGHTS["We"]


def _chunks(prompts):
    out = []
    for p in prompts:
        opener = next(o for o in THOUGHTS if p.endswith(config.OPEN_SEED + o))
        out.append(closed(THOUGHTS[opener], lp=-5.0))
    return out


def _score(prompt):
    """The judge rates the case-split thought best on every axis. Nothing else
    should be asked: the answer request fails before the gate ever runs."""
    family, _ = asked_axis(prompt)
    assert family == "judge", "only the thought judge should be asked anything"
    return lvl(4) if thought_of(prompt).startswith("We ") else lvl(1)


def _opening_thought_then_524(fake_llm):
    return fake_llm(
        fanouts=[openings(*THOUGHTS)],
        chunks=_chunks,
        score=_score,
        streams=[http_error(524)],
    )


def test_a_524_after_the_opening_thought_keeps_the_thinking(cfg, fake_llm):
    cfg(OPEN_THINK=True, BEAM_WIDTH=4, THINK_BRANCH_K=5)
    fake = _opening_thought_then_524(fake_llm)

    events, conv = run_turn(fake, QUESTION)

    # The turn finished instead of raising, and said why it is short.
    kinds = types(events)
    assert list(dict.fromkeys(kinds)) == [
        "start", "block_open", "frontier", "think_token", "think_step",
        "standings", "monologue", "block_close", "degraded", "done",
    ]
    assert "token" not in kinds
    (degraded,) = of_type(events, "degraded")
    assert degraded["error"].startswith("HTTPStatusError: ")
    assert "524" in degraded["error"]
    assert degraded["kept"] == 0

    done = events[-1]
    assert done["text"] == ""
    stats = done["stats"]
    assert stats["degraded"] == degraded["error"]
    assert stats["blocks"] == 1
    assert stats["opening_thought"] is True
    assert stats["kept_tokens"] == 0
    assert stats["notes_in_context"] == 1
    assert stats["requests"] == 4  # fan-out, one chunk, one judge batch, the failed answer

    # The block really ran: four conversations, all finished, the best one won.
    (close,) = of_type(events, "block_close")
    assert close["kind"] == "open"
    assert close["finished"] == 4
    assert close["winner_score"] == 1.0
    (mono,) = of_type(events, "monologue")
    assert mono["text"] == WINNER

    # The turn was still committed, with the opening thought in its header.
    assert [t.role for t in conv.turns] == ["user", "assistant"]
    turn = conv.turns[1]
    assert turn.text == ""
    assert [b.kind for b in turn.blocks] == ["open"]
    head = head_note(turn.notes)
    assert head is not None and head.text == WINNER

    # And it is in the prompt. The failed request carried the thought in the
    # header, and the conversation now renders byte-for-byte what that request
    # sent, so a retry or the next turn re-uses the cached prefix.
    (call,) = fake.of("stream")
    sent = (Conversation().render(pending_user=QUESTION, think=True)
            + chatml.think_head(WINNER))
    assert call["prompt"] == sent
    assert conv.render() == sent + chatml.IM_END + "\n"
    assert chatml.think_head(WINNER) in conv.render(pending_user="And then?")

    assert fake.names() == ["fanout", "chunk", "labels", "answer_probs", "stream"]
    fake.assert_consumed()


def test_a_connection_lost_mid_answer_keeps_what_was_written(cfg, fake_llm):
    cfg(OPEN_THINK=False, ENTROPY_FLOOR=1e9, WINDOW_TOKENS=4)
    fake = fake_llm(streams=[
        Stream(quiet("It", " is", " yes", ","), finish="length"),
        Stream(quiet(" because"), error=httpx.ReadTimeout("read timed out")),
    ])

    events, conv = run_turn(fake, QUESTION)

    (degraded,) = of_type(events, "degraded")
    assert degraded == {"type": "degraded", "error": "ReadTimeout: read timed out",
                        "kept": 5}
    done = events[-1]
    assert done["type"] == "done"
    assert done["text"] == "It is yes, because"
    assert done["stats"]["degraded"] == "ReadTimeout: read timed out"
    assert done["stats"]["kept_tokens"] == 5
    assert conv.turns[1].text == "It is yes, because"


def test_a_non_transient_error_is_not_reported_as_a_partial_answer(cfg, fake_llm):
    # Only endpoint trouble is swallowed. A 400 means the request was wrong,
    # and that must surface rather than look like a short answer.
    cfg(OPEN_THINK=False)
    fake = fake_llm(streams=[http_error(400)])

    with pytest.raises(httpx.HTTPStatusError):
        run_turn(fake, QUESTION)


def test_a_524_inside_the_opening_thought_is_not_caught(cfg, fake_llm):
    # Current behaviour, pinned: the degraded path starts once the opening
    # thought has closed. A transient failure while its beams are still
    # being advanced propagates, and nothing is committed to the conversation.
    cfg(OPEN_THINK=True, BEAM_WIDTH=4, THINK_BRANCH_K=5)
    fake = fake_llm(fanouts=[openings(*THOUGHTS)], chunks=[http_error(524)])
    conv = Conversation()

    with pytest.raises(httpx.HTTPStatusError):
        run_turn(fake, QUESTION, conv)
    assert conv.turns == []
