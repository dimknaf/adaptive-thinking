"""Is the thinking block a real tree, and would a labelled Choice earn a place?

Four questions, in the order that matters. The first two are cheap structural
assertions. The last two decide whether the labelled Choice - a single-prompt
ranking inspired by jevfire: one prompt, every candidate against a verified
single-token label, read at one answer position - is actually better than the
free cumulative-logprob ranking it costs a round trip to improve on.

A measurement script, not an offline unit test: it runs three full answers
against the live endpoint.

CAVEAT: config.CHOICE_EVERY is not read by the engine at present - _run_block
always ranks by logprob and judges on the axes - so runs A, B and C below use
the same configuration, and question 4 compares nothing but run-to-run
variation. Question 3 still measures something real: it calls Engine._rank
directly.

    python -m experiments.tree_probe

(Formerly test_tree.py at the repository root; renamed because it is a
measurement against a live endpoint, not a unit test.)
"""

import asyncio
import statistics
from collections import Counter

from adaptive_thinking import VLLM, Beam, Conversation, Engine, config, prompts

from ._common import collect_turn, utf8_stdout

utf8_stdout()


def diversity(step):
    """Mean pairwise dissimilarity of the running conversations.

    Jaccard distance over character sets - crude, but it moves the right way and
    needs no dependencies. High means the beams are exploring different things;
    near zero means beam search has done what beam search does and converged on
    one string.
    """
    t = [b["text"] for b in step["beams"]]
    pairs = [(a, c) for i, a in enumerate(t) for c in t[i + 1:]]
    if not pairs:
        return 0.0
    return statistics.mean(
        1 - (len(set(a) & set(c)) / max(1, len(set(a) | set(c)))) for a, c in pairs
    )


Q = ("A train leaves at 14:47 and the journey takes 3 hours 38 minutes. "
     "When does it arrive? Show your working.")


async def collect(llm, q, choice_every):
    """Run one response and return its think_step events plus the answer."""
    config.CHOICE_EVERY = choice_every
    steps = []

    def keep(ev):
        if ev["type"] == "think_step":
            steps.append(ev)

    text, stats = await collect_turn(Engine(llm), Conversation(), q, keep)
    return steps, text, stats


async def main():
    llm = VLLM()
    try:
        print(f"width {config.BEAM_WIDTH}  k {config.THINK_BRANCH_K}  "
              f"shortlist {config.CHOICE_SHORTLIST}\n")

        print("--- run A: Choice every step ---", flush=True)
        steps_a, text_a, stats_a = await collect(llm, Q, 1)

        # 1. It is a tree: lineages are born, advance, and are discontinued.
        widest = max((len(s["beams"]) for s in steps_a), default=0)
        considered = sorted({s["considered"] for s in steps_a if s["step"] > 0})
        uids = {b["uid"] for s in steps_a for b in s["beams"]}
        ended = sum(len(s.get("ended", [])) for s in steps_a)
        splits = sum(1 for s in steps_a for b in s["beams"]
                     if b["from_uid"] != b["uid"] and s["step"] > 0)
        parents = Counter(len({b["parent"] for b in s["beams"]}) for s in steps_a)
        print(f"\n1. candidates per step: {considered}   "
              f"(expect {config.BEAM_WIDTH * config.THINK_BRANCH_K})")
        print(f"   lineages born {len(uids)}, discontinued {ended}, splits {splits}"
              f"   {'OK' if ended and splits else 'FAILED - nothing is being pruned or split'}")
        print("   distinct parents surviving per step: "
              + "  ".join(f"{k}:{parents[k]}" for k in sorted(parents)))
        print("   (all mass at 1 would mean the frontier collapsed onto one lineage)")

        # 2. Width never explodes.
        print(f"\n2. widest frontier seen: {widest}   "
              f"{'OK' if widest <= config.BEAM_WIDTH else 'FAILED'}")

        # 2b. THE number the old assertion missed. "Two beams share a parent"
        #     is satisfied by TOTAL COLLAPSE, so it passed 31/32 while the five
        #     conversations (width 5 at the time) were converging into
        #     paraphrases of one sentence.
        #     Frontier diversity is what "going in different ways" means.
        print("\n2b. frontier diversity (1 = nothing shared, 0 = identical)")
        for lab, s_ in (("first step", steps_a[0]), ("quarter", steps_a[len(steps_a)//4]),
                        ("middle", steps_a[len(steps_a)//2]), ("last step", steps_a[-1])):
            print(f"   {lab:11s} {diversity(s_):.3f}")
        nudge = sum(1 for s in steps_a for b in s["beams"]
                    if prompts.NUDGE_PREFIX[:12] in b["text"])
        print(f"   beams carrying our own closing line: {nudge}   "
              f"{'OK' if nudge == 0 else 'FAILED - the tree is copying our boilerplate'}")
        worst = min((b["score"] for s in steps_a for b in s["beams"]), default=0)
        best = max((b["score"] for s in steps_a for b in s["beams"]), default=0)
        print(f"   score range {worst:.3f} to {best:.3f}   "
              f"{'OK' if best < -0.05 else 'suspicious - a near-zero score means a copy'}")

        # 3. Label-position bias, a known trap: an earlier measurement found a
        #    field scoring 3/100 forward and 98/100 reversed. If the winner
        #    changes when the options are listed backwards, the model is reading
        #    the letter, not the text, and the Choice is worthless here.
        print("\n3. label-position bias in the RAW choice "
              "(same candidates listed in reverse)", flush=True)
        engine, conv = Engine(llm), Conversation()
        ctx = conv.render(pending_user=Q)
        flips = same = 0
        # Eight probes, because one flip in three says nothing. Each is five
        # plausible next-steps of the same deliberation, which is exactly the
        # shape the Choice sees in the real loop.
        probe = [
            ["Let me check the minutes first.", "First convert to minutes.",
             "I should add the hours.", "Wait, 47 plus 38 is 85.", "The answer is 18:25."],
            ["14:47 plus 3 hours is 17:47.", "Let me recompute.", "Hmm, that is wrong.",
             "Adding 38 minutes gives 18:25.", "So the train arrives late."],
            ["I need to carry the hour.", "85 minutes is 1h25.", "Let me just add.",
             "The journey is long.", "Convert everything to minutes."],
            ["So 17:47 plus 38 minutes.", "That gives 18:25.", "Let me verify that.",
             "Trains are often delayed.", "The minutes overflow past 60."],
            ["47 + 38 = 85 minutes.", "85 - 60 = 25, carry one hour.",
             "I will guess 18:00.", "Probably around six.", "Let me restate the question."],
            ["Check: 14:47 + 3:38.", "Hours: 14 + 3 = 17.", "It is definitely morning.",
             "Minutes: 47 + 38 = 85.", "I am not sure."],
            ["The arrival is 18:25.", "Actually let me redo this.",
             "Three hours and thirty-eight minutes.", "Skip the working.",
             "Add hours then minutes."],
            ["Convert 14:47 to minutes since midnight.", "That is 887 minutes.",
             "Add 218 minutes.", "I think it is late evening.", "Round to the nearest hour."],
        ]
        for texts in probe:
            cands = [Beam(text=t, score=0.0, parent=0, index=i) for i, t in enumerate(texts)]
            # Measure the RAW choice. Debiasing averages both orders and is
            # symmetric by construction, so testing it this way would pass
            # trivially and tell us nothing about the underlying signal.
            fresh = [Beam(text=t, score=0.0, parent=0, index=i)
                     for i, t in enumerate(texts)]
            fwd = await engine._rank(ctx, cands, debias=False)
            rev = await engine._rank(ctx, fresh, reverse=True, debias=False)
            if fwd[0].text == rev[0].text:
                same += 1
            else:
                flips += 1
            print(f"   forward -> {fwd[0].text[:40]!r}")
            print(f"   reversed-> {rev[0].text[:40]!r}   "
                  f"{'same' if fwd[0].text == rev[0].text else 'FLIPPED'}")
        print(f"   {same}/{same + flips} agreed. "
              + ("OK - the Choice reads the text" if flips == 0 else
                 "position bias present; prefer logprob pruning and say so"))

        # 4. Does the Choice earn its round trip, and does spacing it out help?
        #    At every step the candidates differ by ONE token, which is close to
        #    nothing to judge semantically - a plausible reason the choice adds
        #    little. Every 4th step they have diverged enough that there may
        #    actually be something to choose between. (That reasoning dates
        #    from the one-token-per-step engine. With CHOICE_EVERY unread, as
        #    it is now, runs A, B and C are the same configuration - see the
        #    module docstring.)
        print("\n--- run C: choice every 4 steps ---", flush=True)
        steps_c, text_c, stats_c = await collect(llm, Q, 4)
        print("\n--- run B: logprob only, no choice ---", flush=True)
        steps_b, text_b, stats_b = await collect(llm, Q, 0)

        print("\n4. what the choice costs and what it buys")
        print(f"   {'mode':16s} {'reqs':>5s} {'secs':>7s} {'changed':>9s} {'spills':>7s}")
        for lab, st in (("every step", stats_a), ("every 4 steps", stats_c),
                        ("logprob only", stats_b)):
            changed = f"{st['blocks_that_changed_next_token']}/{st['blocks']}"
            print(f"   {lab:16s} {st['requests']:5d} {st['secs']:7.1f} "
                  f"{changed:>9s} {st['spills']:7d}")
        for lab, txt in (("every step", text_a), ("every 4", text_c), ("logprob", text_b)):
            print(f"\n   {lab:11s}: {txt[-130:]!r}")
    finally:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
