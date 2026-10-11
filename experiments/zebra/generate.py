"""Make a zebra puzzle of size N, certified to have exactly one solution.

N houses in a row, N attribute categories, each category a permutation of N
distinct values. Clues use the standard vocabulary. A backtracking solver
counts solutions (stopping at 2), and clues are stripped while uniqueness
holds so the final set is minimal.

    python -m experiments.zebra.generate [N [SEED]]

prints the puzzle, its solution and the clue count, and writes
zebra_NxN_sSEED.json next to this file, where run.py looks for it. N defaults
to 4 and SEED to 1. Generation is deterministic in (N, SEED), so regenerating
a committed puzzle rewrites its file with identical content.

(Formerly zebra/zebra.py. Its __main__ wrote the JSON into the current
directory, so a run from anywhere but zebra/ left the puzzle where run.py
would never find it.)
"""
import json
import os
import random
import sys
from itertools import permutations

from . import zsolve

# This directory: the committed puzzles live here, and run.py reads them
# from here.
HERE = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = {
    "name":  ["Alice", "Bob", "Carol", "Dan", "Erin", "Frank", "Grace"],
    "drink": ["tea", "coffee", "milk", "water", "juice", "cocoa", "cider"],
    "pet":   ["cat", "dog", "bird", "fish", "horse", "rabbit", "snake"],
    "color": ["red", "blue", "green", "white", "yellow", "purple", "orange"],
    "hobby": ["chess", "painting", "hiking", "baking", "reading", "sewing", "running"],
    "job":   ["doctor", "teacher", "farmer", "pilot", "chef", "nurse", "artist"],
    "music": ["jazz", "rock", "folk", "opera", "blues", "pop", "metal"],
}


def make_solution(n, cats, rng):
    """{category: [value in house 1..n]}"""
    return {c: rng.sample(CATEGORIES[c][:n], n) for c in cats}


def house_of(sol, cat, val):
    return sol[cat].index(val) + 1


# --- clue types -------------------------------------------------------------
# Each clue is (text, kind, payload). The solver interprets payload directly.

def all_true_clues(sol, cats, rng):
    out = []
    items = [(c, v) for c in cats for v in sol[c]]
    for c, v in items:
        h = house_of(sol, c, v)
        out.append((f"The {c} {v} is in house {h}.", "pos", (c, v, h)))
    for (c1, v1) in items:
        for (c2, v2) in items:
            if c1 >= c2:
                continue
            h1, h2 = house_of(sol, c1, v1), house_of(sol, c2, v2)
            if h1 == h2:
                out.append((f"The {c1} {v1} and the {c2} {v2} are the same house.",
                            "same", (c1, v1, c2, v2)))
            if h2 - h1 == 1:
                out.append((f"The {c1} {v1} is immediately left of the {c2} {v2}.",
                            "immleft", (c1, v1, c2, v2)))
            if abs(h1 - h2) == 1:
                out.append((f"The {c1} {v1} is next to the {c2} {v2}.",
                            "next", (c1, v1, c2, v2)))
            if h1 < h2:
                out.append((f"The {c1} {v1} is somewhere left of the {c2} {v2}.",
                            "left", (c1, v1, c2, v2)))
            if h1 != h2:
                out.append((f"The {c1} {v1} is not the same house as the {c2} {v2}.",
                            "diff", (c1, v1, c2, v2)))
    rng.shuffle(out)
    return out


def holds(clue, assign):
    """True if `clue` is satisfiable given partial assignment {(cat,val): house}.
    Unknown positions never falsify - this only prunes."""
    kind, pay = clue[1], clue[2]
    g = assign.get
    if kind == "pos":
        c, v, h = pay
        return g((c, v)) in (None, h)
    c1, v1, c2, v2 = pay
    h1, h2 = g((c1, v1)), g((c2, v2))
    if h1 is None or h2 is None:
        return True
    if kind == "same":
        return h1 == h2
    if kind == "diff":
        return h1 != h2
    if kind == "immleft":
        return h2 - h1 == 1
    if kind == "next":
        return abs(h1 - h2) == 1
    if kind == "left":
        return h1 < h2
    raise ValueError(kind)


def count_solutions(clues, cats, n, cap=2):
    """The CSP solver. Verified to agree with the brute-force one below."""
    return zsolve.solve(clues, cats, {c: CATEGORIES[c][:n] for c in cats}, n, cap)


def count_solutions_bruteforce(clues, cats, n, cap=2):
    """Backtracking over categories: assign each category a permutation.
    Correct but hopeless past 4x4; kept only as the cross-check oracle."""
    vals = {c: CATEGORIES[c][:n] for c in cats}
    found = [0]

    def rec(i, assign):
        if found[0] >= cap:
            return
        if i == len(cats):
            found[0] += 1
            return
        c = cats[i]
        for perm in permutations(range(1, n + 1)):
            a2 = dict(assign)
            for v, h in zip(vals[c], perm):
                a2[(c, v)] = h
            if all(holds(cl, a2) for cl in clues):
                rec(i + 1, a2)
                if found[0] >= cap:
                    return

    rec(0, {})
    return found[0]


def generate(n, seed):
    rng = random.Random(seed)
    cats = list(CATEGORIES)[:n]
    sol = make_solution(n, cats, rng)
    pool = all_true_clues(sol, cats, rng)

    # grow until unique
    clues = []
    for cl in pool:
        clues.append(cl)
        if count_solutions(clues, cats, n) == 1:
            break
    else:
        raise RuntimeError("never became unique")

    # strip back to minimal
    i = 0
    while i < len(clues):
        trial = clues[:i] + clues[i + 1:]
        if trial and count_solutions(trial, cats, n) == 1:
            clues = trial
        else:
            i += 1
    return sol, cats, clues


def render(n, cats, clues):
    lines = [
        f"There are {n} houses in a row, numbered 1 to {n} from left to right.",
        "Each house has a different " + ", ".join(cats[:-1]) + f" and {cats[-1]}.",
        "",
        "The attributes are:",
    ]
    for c in cats:
        lines.append(f"  {c}: " + ", ".join(CATEGORIES[c][:n]))
    lines += ["", "Clues:"]
    for i, (text, _, _) in enumerate(clues, 1):
        lines.append(f"  {i}. {text}")
    lines += ["",
              "Work out the complete grid. Give your final answer as one line "
              "per house, in the form:",
              "  House 1: " + ", ".join(f"{c}=..." for c in cats)]
    return "\n".join(lines)


def puzzle_path(n, seed, directory=None):
    """Where the puzzle for (n, seed) is kept: in `directory`, by default
    this file's own."""
    return os.path.join(HERE if directory is None else directory,
                        f"zebra_{n}x{n}_s{seed}.json")


def puzzle_record(n, seed, sol, cats, clues):
    """The JSON a puzzle is kept as: the prompt both arms are sent, the
    solution grid it is graded against, and the clue count."""
    return {"n": n, "seed": seed, "cats": cats,
            "prompt": render(n, cats, clues),
            "solution": {c: sol[c] for c in cats},
            "n_clues": len(clues)}


def write_puzzle(path, record):
    """Write a puzzle record as the committed files are written."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    n = int(argv[0]) if len(argv) > 0 else 4
    seed = int(argv[1]) if len(argv) > 1 else 1
    sol, cats, clues = generate(n, seed)
    print(render(n, cats, clues))
    print()
    print("SOLUTION")
    for h in range(1, n + 1):
        print(f"  House {h}: " + ", ".join(f"{c}={sol[c][h-1]}" for c in cats))
    print(f"\nclues={len(clues)}  unique={count_solutions(clues, cats, n) == 1}")
    write_puzzle(puzzle_path(n, seed), puzzle_record(n, seed, sol, cats, clues))


if __name__ == "__main__":
    main()
