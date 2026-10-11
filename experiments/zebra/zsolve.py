"""A real CSP solver for zebra puzzles.

The permutation-based version was fine at 4x4 and hopeless at 5x5. Here each
(category, value) pair is a variable whose domain is the set of houses, with
all-different inside a category. Propagate, then branch on the smallest
domain. Counts solutions and stops at `cap`.
"""


def solve(clues, cats, values, n, cap=2):
    """clues: list of (text, kind, payload). values: {cat: [v, ...]}"""
    var_list = [(c, v) for c in cats for v in values[c]]
    full = frozenset(range(1, n + 1))
    dom = {var: set(full) for var in var_list}

    # clue -> the variables it touches, for cheap re-queueing
    touching = {var: [] for var in var_list}
    for cl in clues:
        kind, pay = cl[1], cl[2]
        vs = [(pay[0], pay[1])] if kind == "pos" else [(pay[0], pay[1]), (pay[2], pay[3])]
        for v in vs:
            touching[v].append(cl)

    def prop(dom):
        """Return False on contradiction. Mutates dom."""
        queue = list(clues)
        while queue:
            cl = queue.pop()
            kind, pay = cl[1], cl[2]
            changed = []
            if kind == "pos":
                c, v, h = pay
                a = (c, v)
                if h not in dom[a]:
                    return False
                if len(dom[a]) > 1:
                    dom[a] = {h}
                    changed.append(a)
            else:
                c1, v1, c2, v2 = pay
                a, b = (c1, v1), (c2, v2)
                da, db = dom[a], dom[b]
                if kind == "same":
                    na, nb = da & db, db & da
                elif kind == "diff":
                    na = da - (db if len(db) == 1 else set())
                    nb = db - (da if len(da) == 1 else set())
                elif kind == "immleft":
                    na = {h for h in da if h + 1 in db}
                    nb = {h for h in db if h - 1 in da}
                elif kind == "next":
                    na = {h for h in da if (h - 1 in db or h + 1 in db)}
                    nb = {h for h in db if (h - 1 in da or h + 1 in da)}
                elif kind == "left":
                    na = {h for h in da if h < max(db)} if db else set()
                    nb = {h for h in db if h > min(da)} if da else set()
                else:
                    raise ValueError(kind)
                if not na or not nb:
                    return False
                if na != da:
                    dom[a] = na
                    changed.append(a)
                if nb != db:
                    dom[b] = nb
                    changed.append(b)

            # all-different within a category, for any var now fixed
            for var in changed:
                if len(dom[var]) == 1:
                    (h,) = tuple(dom[var])
                    cat = var[0]
                    for other in var_list:
                        if other[0] == cat and other != var and h in dom[other]:
                            dom[other].discard(h)
                            if not dom[other]:
                                return False
                            changed.append(other)
                queue.extend(touching[var])
        # every category must still be able to cover all houses (Hall check)
        for c in cats:
            union = set()
            for v in values[c]:
                union |= dom[(c, v)]
            if len(union) < n:
                return False
        return True

    found = [0]

    def rec(dom):
        if found[0] >= cap:
            return
        if not prop(dom):
            return
        unfixed = [v for v in var_list if len(dom[v]) > 1]
        if not unfixed:
            found[0] += 1
            return
        var = min(unfixed, key=lambda v: len(dom[v]))
        for h in sorted(dom[var]):
            child = {k: set(s) for k, s in dom.items()}
            child[var] = {h}
            rec(child)
            if found[0] >= cap:
                return

    rec(dom)
    return found[0]
