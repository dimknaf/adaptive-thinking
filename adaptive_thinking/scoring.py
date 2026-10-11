"""Levelled reads: how the model's judgement becomes a number.

Both the gate and the thought judge ask levelled closed questions - one axis,
one item, one answer position, the letter labels read with
logprob_token_ids - and turn the answer into an expected level, 0..1
(_expected). levelled_read batches every (item, axis) pair into ONE request.
score_axes is the thought judge built on it: `raw` is the absolute weighted
mean, and `overall` is standardised within a group (_standardise_group).
_in_contention is the one test of whether a live conversation is still worth
running, and it asks only about `raw`.
"""

from . import config, prompts

def _in_contention(beam, leader):
    """Is this conversation still worth letting run?

    The one test used everywhere a live beam might be dropped, so that the
    answer is always the same: SCORE. Not how long it has run, not whether
    other thoughts have finished, not whether the block as a whole has stopped
    improving. A conversation that is still scoring near the best is kept,
    however stale the block around it has become.

    Unjudged counts as in contention - it has not had its say yet.
    """
    if beam.raw is None or leader is None:
        return True
    return leader - beam.raw < config.CHURN_MARGIN


def _expected(probs, n_levels):
    """A levelled answer as one number: worst level 0, best level 1.

    The probabilities come back renormalised over the letter labels only, so
    this is a plain expected value over the levels. Shared by the two places
    that ask a levelled question - the gate and the thought judge - because
    they are the same read, asked about different things.
    """
    p = probs[:n_levels]
    tot = sum(p) or 1.0
    return sum(i * v for i, v in enumerate(p)) / (tot * (n_levels - 1))


async def levelled_read(llm, table, items, prompt):
    """Ask every axis of `table` about every item, in ONE batched request.

    The read the gate and the thought judge share: each (item, axis) pair
    is a levelled closed question, `prompt(item, axis)`, answered at a
    single position against the letter labels. The batch holds the
    prompts item by item and axis by axis within an item, in the order of
    `table`, which is the order the rows are read back in.

    Returns, per item: (axes, raw) - each axis as the expected level, 0..1
    (see _expected), and their mean weighted by the table's weights.
    Nothing is rounded here; the callers round what they report.
    """
    names = list(table)
    n_levels = max(len(table[a]["levels"]) for a in names)
    labels = await llm.labels(n_levels)
    ids = [i for _, i in labels]

    batch = [prompt(x, a) for x in items for a in names]
    rows = await llm.answer_probs(batch, ids)

    wsum = sum(table[a]["weight"] for a in names) or 1.0
    out = []
    for ci in range(len(items)):
        axes = {
            a: _expected(rows[ci * len(names) + ai], len(table[a]["levels"]))
            for ai, a in enumerate(names)
        }
        out.append((axes, sum(table[a]["weight"] * axes[a] for a in names) / wsum))
    return out


def _standardise_group(group, names):
    """Turn a group's raw axes into `overall`, standardised within the group.

    Standardise each axis ACROSS the candidates before combining, within
    each group. Without standardising, an axis that merely varies more
    silently dominates one that matters more and the ranking follows
    whichever axis is noisiest. Within the group, because the groups are
    different kinds of thing - see score_axes.

    Writes `<axis>_z` into each candidate's `axes` and sets its `overall`,
    in place. `names` are the axes, in prompts.AXES order.
    """
    for a in names:
        vals = [c.axes[a] for c in group]
        spread = (max(vals) - min(vals)) or 1.0
        mid = sum(vals) / len(vals)
        for c in group:
            # Range, not standard deviation - with a handful of
            # candidates an sd estimate is too noisy to divide by.
            z = (c.axes[a] - mid) / spread
            c.axes[a + "_z"] = max(-config.AXIS_CLIP, min(config.AXIS_CLIP, z))
    for c in group:
        # A sum of clipped terms, so no single axis can kill a candidate.
        c.overall = sum(
            prompts.AXES[a]["weight"] * c.axes[a + "_z"] for a in names
        )


async def score_axes(llm, ctx, *groups):
    """Score every candidate on every axis, in ONE batched request.

    Each (candidate, axis) pair is a closed question answered at a single
    position, so four candidates on four axes is 16 reads in the same round
    trip a single Choice costs. There are no labelled options spanning the
    candidates, so there is nothing positional to anchor on - which is what
    made the Choice unreliable (4 of 8 reversed-order probes flipped).

    Several GROUPS may be passed. The raw read is the same for all of them
    and rides in the one request, but the standardisation that turns raw
    axes into `overall` happens WITHIN each group. Live beams are compared
    against live beams and finished thoughts against finished thoughts,
    because the two are not the same kind of thing: a half-written thought
    scores badly on "progress" for being half-written, and mixing the
    groups would retire live beams for being alive.

    Writes `axes` (raw 0..1 per axis) and `overall` onto each candidate.
    """
    names = list(prompts.AXES)
    groups = [g for g in groups if g]
    cands = [c for g in groups for c in g]
    if not cands or not names:
        return
    # Expected value over the levels: worst level scores 0, best scores 1.
    # `raw` is absolute, 0..1, no standardisation. This is the one that
    # can be compared against a thought in another group.
    read = await levelled_read(
        llm, prompts.AXES, cands, lambda c, a: prompts.axis_prompt(ctx, c.text, a))
    for c, (axes, raw) in zip(cands, read):
        c.axes, c.raw = axes, raw

    # Then `overall`, standardised within each group - see
    # _standardise_group.
    for group in groups:
        _standardise_group(group, names)
