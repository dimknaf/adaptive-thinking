"""The events the engine yields, and the builders for the structured ones.

Events are a CONTRACT. static/index.html, cli.py and
experiments/gate_audit.py read them, server.py forwards them verbatim over
SSE and appends several to events.jsonl, and the measurement scripts fold
them into their results. So the type names, the field names, which keys are
present, their order and the rounding of every number are part of the
behaviour: a builder here changes only together with every reader of what it
builds.

Each event is a dict with a "type"; EVENT_TYPES lists every type the engine
yields. The builders below cover the events with a non-trivial shape; the
rest are written where they are yielded.
"""

from . import config, prompts

# Every "type" Engine.respond can yield, in the order a turn meets them: the
# turn's start; for the opening thought and each triggered block, its events
# from block_open to block_close; the answer's tokens and each gate window,
# the rewind, resumption and restart around a block; a degraded answer;
# eviction; and the turn's done. server.py adds "turn" (events.jsonl only),
# and "error" and "end" (the SSE stream only), which are not engine events.
EVENT_TYPES = (
    "start",
    "block_open",
    "frontier",
    "think_token",
    "think_step",
    "standings",
    "monologue",
    "block_close",
    "token",
    "window",
    "rewind",
    "resumed",
    "restart",
    "degraded",
    "evict",
    "done",
)


def _beam_json(b, step=None):
    """One conversation, as the UI needs it. Shared so the frontier event and
    the checkpoint event cannot drift apart."""
    return {
        "i": b.index,
        "uid": b.uid,
        "from_uid": b.from_uid,
        "parent": b.parent,
        "tok": b.last,
        "text": b.text,
        "score": round(b.rank_score, 3),
        "total": round(b.score, 2),
        "p": round(b.p, 4) if b.p is not None else None,
        "done": b.done,
    }


def _block_open_event(block, seed):
    """The block_open event: which block, how wide, where, and its seed."""
    return {
        "type": "block_open",
        "id": block.id,
        "kind": block.kind,
        "width": config.BEAM_WIDTH,
        "branch_k": config.THINK_BRANCH_K,
        "tail": block.tail,
        "crit": block.crit,
        "gate": block.gate,
        "seed": seed,
    }


def _think_step_event(block, step, cands, used_choice, beams, finished, before,
                      retired, entered, leader, dry, closed, distinct):
    """The think_step event: the frontier after one checkpoint.

    `cands` are the live candidates that competed for the frontier, `before`
    the uids on it when the checkpoint began, `leader` and `dry` the block's
    progress as it stood BEFORE this checkpoint was judged, and `distinct`
    the number of slots holding a chunk no other slot holds.
    """
    alive = {b.uid for b in beams}
    return {
        "type": "think_step",
        "block": block.id,
        "step": step,
        "considered": len(cands),
        "ranked_by": "choice" if used_choice else "logprob",
        "beams": [_beam_json(b) for b in beams],
        # Thoughts that have closed themselves. They are not on the
        # frontier any more but they are still in the running, and they
        # need cards of their own: at 500-token chunks most thoughts
        # finish in the first one, so without this the whole block
        # would be a pool and nothing would ever be drawn.
        "pool": [_beam_json(b) for b in finished],
        # Lineages that were alive last step and are not carried
        # forward. FINISHING is not being dropped - a thought that
        # closed itself left the frontier on its own terms and is
        # still scored, so it must not be shown as discontinued.
        "ended": sorted(before - alive - {b.uid for b in finished}),
        # Retired for scoring clearly below the field, and the
        # variation of a good branch that took the slot. Always the
        # same length: a retirement never costs the block a slot.
        "retired": [{"uid": b.uid, "overall": round(b.overall, 3),
                     "replaced": bool(entered)} for b in retired],
        "entered": [{"uid": b.uid, "from_uid": b.from_uid,
                     "tok": b.last} for b in entered],
        # Is the block still getting anywhere? `dry` is checkpoints
        # since the last new best and it is what decides whether a
        # retired slot is refilled or shut, so it is shown rather
        # than left implicit.
        "leader": None if leader is None else round(leader, 3),
        "dry": dry,
        "closed": closed,
        "dropped": len(cands) - len(beams),
        "distinct": distinct,
        "backfilled": len(beams) - distinct,
    }


def _standings_event(block, contenders, beams, finished):
    """The standings event: every contender, in medal order, with its scores."""
    return {
        "type": "standings",
        "block": block.id,
        "ranked": [
            {"uid": c.uid, "rank": i + 1, "medal": c.medal,
             "done": c.done,
             "overall": round(c.overall, 3) if c.overall is not None else None,
             # Absolute 0..1, comparable across groups. This is the
             # number the churn runs on, so CHURN_MARGIN has to be
             # set from the spread of THIS, not of `overall`.
             "raw": round(c.raw, 3) if c.raw is not None else None,
             "axes": {a: round(c.axes.get(a, 0.0), 3) for a in prompts.AXES}}
            for i, c in enumerate(contenders)
        ],
        "live": len(beams),
        "finished": len(finished),
    }


def _block_close_event(block, was):
    """The block_close event, for the opening thought and a gate block alike.

    `was` is the token the answer was about to write when the block opened
    (None for the opening thought: the answer had not started).

    `kind` is on the OPENING thought's close only. A gate block's close has
    never carried it, and consumers tell the two apart by its presence, so it
    is added only for that block: the key set of each stays what it was.
    """
    ev = {"type": "block_close", "id": block.id}
    if block.kind == "open":
        ev["kind"] = block.kind
    ev.update({
        "width": config.BEAM_WIDTH,
        "steps": block.steps,
        # The fan-out read, then two requests per checkpoint whatever the
        # width: one to advance every live beam by a chunk, one to judge the
        # thoughts on the axes.
        "requests": block.requests,
        "winner_score": block.winner_score,
        "finished": block.finished_count,
        "explored_tokens": block.explored_tokens,
        "secs": round(block.secs, 2),
        "was_going_to_write": was,
    })
    return ev


def _window_event(opened):
    """Every position the gate considered, fired or not.

    The whole score goes on the wire, not just the number that made the
    decision: the per-axis breakdown is what makes it possible to ask later
    whether the gate stops for good reasons, and to set GATE_THRESHOLD from a
    measured distribution rather than a guess.
    """
    return {
        "type": "window",
        "probed": [
            {"pos": c.idx, "tok": c.step.text, "h": round(c.step.entropy, 3), **s}
            for c, s in opened["probed"]
        ],
        "opened": opened["pick"] is not None,
    }


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def _done_stats(st, secs, used, notes_in_context):
    """The `stats` of a turn's done event. Key order is part of the event.

    `secs` is the turn's wall time, `used` its usage deltas (UsageMeter.used)
    and `notes_in_context` the notes the whole conversation still holds.
    Rounding happens here, as it always did.
    """
    return {
        "secs": round(secs, 2),
        "kept_tokens": st.idx,
        "rewound_tokens": st.wasted,
        "blocks": len(st.blocks),
        # Did the model think before it started? The opening thought is
        # counted in `blocks` because it costs what a block costs, but it is
        # excluded from the metric below - it replaced no token, so counting
        # it could only dilute the answer.
        "opening_thought": config.OPEN_THINK,
        # The one number that decides whether any of this matters: after
        # thinking, did the model write something other than what it was
        # about to write? If this is near zero the tree changes nothing and
        # its cost buys nothing.
        "blocks_that_changed_next_token": sum(
            1 for b in st.blocks if b.kind == "gate" and b.changed_next_token
        ),
        "think_steps": sum(b.steps for b in st.blocks),
        "explored_tokens": sum(b.explored_tokens for b in st.blocks),
        # Transcript markers cut out of the answer. Must be 0 in a clean run;
        # anything else means the model started writing the transcript
        # instead of the answer and we caught it.
        # A marker arriving in the SAME chunk as the answer's last words is
        # the model ending its turn and costs the reader nothing. One
        # arriving mid-answer means it started writing the transcript. The
        # count alone could not tell those apart.
        "spills": len(st.spills),
        "spill_markers": st.spills,
        # Blocks whose note made the model begin the answer again. Must be 0:
        # the anchor in _inline_note exists to prevent it, and anything here
        # means the anchor is not holding.
        "restarts": len(st.restarts),
        "restart_detail": st.restarts,
        # Set when the endpoint gave out mid-answer. The thinking and
        # whatever was written survive; the answer is incomplete and says so
        # rather than the turn vanishing into an error.
        "degraded": st.degraded,
        "retries": used["retries"],
        "notes_in_context": notes_in_context,
        "requests": used["requests"],
        "prompt_tokens": used["prompt_tokens"],
        "completion_tokens": used["completion_tokens"],
        # Can it pick up where it left off? `resume_logprob` is the mean
        # logprob of the tokens right after a block; `logprob` is the mean
        # over the whole answer. A model thrown by the interruption is more
        # surprised by its own resumed text, so the delta goes negative. Near
        # zero means the seam is invisible to it.
        "inline_blocks": config.KEEP_BLOCKS_INLINE,
        "logprob": _mean(st.logprobs_all),
        "resume_logprob": _mean(st.logprobs_resume),
        "resume_logprob_delta": (
            None
            if not st.logprobs_resume or not st.logprobs_all
            else round(_mean(st.logprobs_resume) - _mean(st.logprobs_all), 4)
        ),
        "resume_samples": len(st.logprobs_resume),
    }
