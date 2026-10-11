"""The gate: where in the answer is a block worth opening?

Mid-answer, every streamed token is a candidate position (_Cand). _eligible
drops the positions where branching could change nothing or would do harm -
the entropy floor, fewer than two alternatives, too soon after the last
block, inside the opening clause, inside text already written, mid-sentence.
choose() scores the PROBE_CANDIDATES highest-entropy survivors on the
GATE_AXES in one batched request (gate_scores), records the Yes/No probe
beside them while GATE_LOG_YESNO is on, and picks the best position whose
`crit` reaches GATE_THRESHOLD, if any. `crit` is absolute, 0..1: it is never
standardised across the positions in a window (see config.GATE_THRESHOLD).
"""

from dataclasses import dataclass

from . import config, prompts
from .scoring import levelled_read
from .text import _already_written, _at_boundary, splice


@dataclass
class _Cand:
    idx: int  # token index within this response
    off: int  # character offset in `out` BEFORE this token
    step: object


def _eligible(c, last_block_at, vis):
    """May the gate consider this position at all?

    Six rules, checked in this order and short-circuiting: the entropy
    floor, at least two distinct alternatives, MIN_GAP_TOKENS since the last
    block, past WARMUP_TOKENS, not inside text already written, and at a
    sentence or clause boundary. `vis` is the visible answer the candidate's
    offset points into.
    """
    return (
        c.step.entropy >= config.ENTROPY_FLOOR
        and len(c.step.alternatives) >= 2
        and c.idx - last_block_at >= config.MIN_GAP_TOKENS
        # Past the opening clause. `</think>` reads as "the answer begins
        # here", so a block closing inside the opener makes the model
        # restart it - see the note on WARMUP_TOKENS.
        and c.idx >= config.WARMUP_TOKENS
        # Not inside text the answer has already written. A block rewinds
        # and the model restarts the sentence; the gate then fires again on
        # the restatement, blocks again, restarts again - the answer loops
        # and, with no cap, never ends. Repetition means the model is
        # stuck, and thinking harder at that point deepens the loop rather
        # than breaking it.
        and not _already_written(vis, c.off)
        # A new sentence starts here anyway, so `</think>` meaning "the
        # answer begins" costs nothing - see _at_boundary.
        and _at_boundary(vis, c.off)
    )


async def choose(llm, window, last_block_at, prefix, vis, notes):
    """Pick where in this window to open a block, if anywhere."""
    cands = [c for c in window if _eligible(c, last_block_at, vis)]
    if not cands:
        return None
    cands.sort(key=lambda c: -c.step.entropy)
    cands = cands[: config.PROBE_CANDIDATES]

    # What the model would be reading at each candidate position.
    ctxs = [
        prefix + splice(vis[: c.off], [n for n in notes if n.off <= c.off])
        for c in cands
    ]

    if not config.PROBE_ENABLED:
        # Ablation: entropy alone decides. Kept so the gate's contribution
        # can be measured rather than asserted.
        return {"probed": [(c, {"crit": None}) for c in cands],
                "pick": (cands[0], {"crit": float("nan")})}

    scored = await gate_scores(llm, ctxs)
    # The old binary probe, recorded alongside for calibration. One extra
    # batched request per window, and only while GATE_LOG_YESNO is on.
    if config.GATE_LOG_YESNO:
        ps = await llm.p_yes(
            [prompts.probe_prompt(x, prompts.GATE_PROBE) for x in ctxs]
        )
        for s, p in zip(scored, ps):
            s["p_yes"] = round(p, 4)

    probed = sorted(zip(cands, scored), key=lambda cs: -cs[1]["crit"])
    best, s = probed[0]
    return {
        "probed": probed,
        "pick": (best, s) if s["crit"] >= config.GATE_THRESHOLD else None,
    }


async def gate_scores(llm, ctxs):
    """How critical is the next step, at each candidate position?

    Every (position, axis) pair is one closed question answered at a single
    position, and they all ride in ONE batched request - six positions on
    three axes is 18 reads in the round trip the old Yes/No probe spent on
    six. There are no labelled options spanning the positions, so there is
    nothing positional to anchor on.

    The result is ABSOLUTE, 0..1, and is not standardised across the
    positions. See config.GATE_THRESHOLD for why that matters.
    """
    read = await levelled_read(llm, prompts.GATE_AXES, ctxs, prompts.gate_prompt)
    return [{"crit": round(crit, 4), **{a: round(v, 3) for a, v in axes.items()}}
            for axes, crit in read]
