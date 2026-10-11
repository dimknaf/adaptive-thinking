"""The zebra experiment's offline parts: solver, generator, committed puzzles,
and the runner in experiments/zebra/run.py - its exact-grid grader, and both
arms against a fake endpoint.

run.py's main() talks to the endpoint and runs only under __main__; the `run`
fixture imports the module fresh and checks that importing it starts nothing.
"""

import asyncio
import importlib
import json
import os
import random
import re
import sys
from pathlib import Path

import httpx
import pytest

from adaptive_thinking import chatml, config, prompts
from experiments.zebra import generate, zsolve
from fake_endpoint import Router, closed, json_response, mock_transport, openings, quiet, sse

ROOT = Path(__file__).resolve().parent.parent
ZEBRA_DIR = ROOT / "experiments" / "zebra"

COMMITTED = [(3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2)]
# Regenerating a 5x5 puzzle takes minutes (the generator re-counts solutions
# after every clue it adds and removes), so those two only run under -m slow.
REGENERATE = [
    (3, 1), (3, 2), (4, 1), (4, 2),
    pytest.param(5, 1, marks=pytest.mark.slow),
    pytest.param(5, 2, marks=pytest.mark.slow),
]
# Certifying a puzzle from its prompt is fast except for 5x5 seed 1, whose
# minimality check alone takes around 20 seconds.
CERTIFY = [
    (3, 1), (3, 2), (4, 1), (4, 2),
    pytest.param(5, 1, marks=pytest.mark.slow),
    (5, 2),
]


def puzzle_path(n, seed):
    return ZEBRA_DIR / f"zebra_{n}x{n}_s{seed}.json"


def load_puzzle(n, seed):
    return json.loads(puzzle_path(n, seed).read_text(encoding="utf-8"))


def values(cats, n):
    return {c: generate.CATEGORIES[c][:n] for c in cats}


# The six clue sentences generate.all_true_clues writes, inverted.
_CLUE_FORMS = [
    (re.compile(r"The (\w+) (\w+) is in house (\d+)\."), "pos"),
    (re.compile(r"The (\w+) (\w+) and the (\w+) (\w+) are the same house\."), "same"),
    (re.compile(r"The (\w+) (\w+) is immediately left of the (\w+) (\w+)\."), "immleft"),
    (re.compile(r"The (\w+) (\w+) is next to the (\w+) (\w+)\."), "next"),
    (re.compile(r"The (\w+) (\w+) is somewhere left of the (\w+) (\w+)\."), "left"),
    (re.compile(r"The (\w+) (\w+) is not the same house as the (\w+) (\w+)\."), "diff"),
]


def clues_from_prompt(prompt):
    """Read the (text, kind, payload) clues back out of a rendered prompt."""
    out = []
    for line in prompt.split("\n"):
        m = re.fullmatch(r"  \d+\. (.*)", line)
        if not m:
            continue
        text = m.group(1)
        for pattern, kind in _CLUE_FORMS:
            mm = pattern.fullmatch(text)
            if mm:
                g = mm.groups()
                payload = (g[0], g[1], int(g[2])) if kind == "pos" else g
                out.append((text, kind, tuple(payload)))
                break
        else:
            raise AssertionError(f"unrecognised clue: {text!r}")
    return out


def assignment(solution, cats):
    """{(cat, value): house} for a solution grid."""
    return {(c, v): h + 1 for c in cats for h, v in enumerate(solution[c])}


def answer_text(d):
    """The committed solution written in the answer format the prompt asks for."""
    return "\n".join(
        f"House {h}: " + ", ".join(f"{c}={d['solution'][c][h - 1]}" for c in d["cats"])
        for h in range(1, d["n"] + 1)
    )


# ---------------------------------------------------------------------------
# zsolve.solve on hand-built puzzles
# ---------------------------------------------------------------------------

NAMES3 = {"name": ["Alice", "Bob", "Carol"]}
NAMES2 = {"name": ["Alice", "Bob"]}
TWO_BY_TWO = {"name": ["Alice", "Bob"], "drink": ["tea", "coffee"]}


def pos(c, v, h):
    return ("t", "pos", (c, v, h))


def rel(kind, c1, v1, c2, v2):
    return ("t", kind, (c1, v1, c2, v2))


@pytest.mark.parametrize("clues, cats, vals, n, expected", [
    ([], ["name"], NAMES3, 3, 6),
    ([pos("name", "Alice", 1)], ["name"], NAMES3, 3, 2),
    ([pos("name", "Alice", 1), pos("name", "Bob", 1)], ["name"], NAMES3, 3, 0),
    ([pos("name", "Alice", 1), pos("name", "Bob", 2)], ["name"], NAMES3, 3, 1),
    ([], ["name", "drink"], TWO_BY_TWO, 2, 4),
    ([rel("same", "drink", "tea", "name", "Alice")], ["name", "drink"], TWO_BY_TWO, 2, 2),
    ([rel("diff", "drink", "tea", "name", "Alice")], ["name", "drink"], TWO_BY_TWO, 2, 2),
    ([rel("same", "drink", "tea", "name", "Alice"),
      rel("diff", "drink", "tea", "name", "Alice")], ["name", "drink"], TWO_BY_TWO, 2, 0),
    ([rel("immleft", "name", "Bob", "name", "Alice")], ["name"], NAMES2, 2, 1),
    ([rel("left", "name", "Carol", "name", "Alice")], ["name"], NAMES3, 3, 3),
    ([rel("next", "name", "Alice", "name", "Bob")], ["name"], NAMES3, 3, 4),
    ([rel("immleft", "name", "Alice", "name", "Bob"),
      rel("immleft", "name", "Bob", "name", "Carol")], ["name"], NAMES3, 3, 1),
])
def test_solve_counts_small_puzzles(clues, cats, vals, n, expected):
    assert zsolve.solve(clues, cats, vals, n, cap=10) == expected


def test_solve_stops_counting_at_the_cap():
    assert zsolve.solve([], ["name"], NAMES3, 3) == 2  # default cap
    assert zsolve.solve([], ["name"], NAMES3, 3, cap=1) == 1
    assert zsolve.solve([], ["name"], NAMES3, 3, cap=100) == 6


def test_solve_counts_every_3x3_grid_without_clues():
    cats = list(generate.CATEGORIES)[:3]
    assert zsolve.solve([], cats, values(cats, 3), 3, cap=1000) == 6 ** 3


def test_solve_rejects_an_unknown_clue_kind():
    with pytest.raises(ValueError, match="bogus"):
        zsolve.solve([rel("bogus", "name", "Alice", "name", "Bob")], ["name"], NAMES3, 3)


# ---------------------------------------------------------------------------
# the CSP solver against the brute-force oracle
# ---------------------------------------------------------------------------


def _random_clue_set(rng, n, k_range, solutions=2):
    """k clues drawn from the true clues of random solutions. From two
    solutions some sets contradict each other; from one they never do, so
    they are either unique or open."""
    cats = list(generate.CATEGORIES)[:n]
    pool = []
    for _ in range(solutions):
        pool += generate.all_true_clues(generate.make_solution(n, cats, rng), cats, rng)
    return cats, rng.sample(pool, rng.randint(*k_range))


def test_csp_solver_agrees_with_brute_force_on_random_3x3_clue_sets():
    exact = 6 ** 3 + 1  # above the number of 3x3 grids, so both count exactly
    counts = []
    for seed in range(150):
        rng = random.Random(seed)
        cats, clues = _random_clue_set(rng, 3, (0, 7))
        csp = generate.count_solutions(clues, cats, 3, cap=exact)
        brute = generate.count_solutions_bruteforce(clues, cats, 3, cap=exact)
        assert csp == brute, (seed, clues)
        counts.append(csp)
    # The sample really covered every regime.
    assert 0 in counts and 1 in counts and max(counts) > 1


def test_csp_solver_agrees_with_brute_force_on_random_4x4_clue_sets():
    counts = []
    for seed in range(24):
        rng = random.Random(1000 + seed)
        if seed % 2 == 0:
            cats, clues = _random_clue_set(rng, 4, (10, 24), solutions=1)
        else:
            cats, clues = _random_clue_set(rng, 4, (4, 12), solutions=2)
        csp = generate.count_solutions(clues, cats, 4, cap=2)
        brute = generate.count_solutions_bruteforce(clues, cats, 4, cap=2)
        assert csp == brute, (seed, clues)
        counts.append(csp)
    assert set(counts) == {0, 1, 2}


# ---------------------------------------------------------------------------
# holds(): the brute-force oracle's clue check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind, h1, h2, expected", [
    ("same", 2, 2, True), ("same", 1, 2, False),
    ("diff", 1, 2, True), ("diff", 3, 3, False),
    ("immleft", 1, 2, True), ("immleft", 2, 1, False), ("immleft", 1, 3, False),
    ("next", 2, 1, True), ("next", 2, 3, True), ("next", 1, 3, False),
    ("left", 1, 3, True), ("left", 3, 1, False), ("left", 2, 2, False),
])
def test_holds_on_known_positions(kind, h1, h2, expected):
    clue = rel(kind, "name", "Alice", "drink", "tea")
    assign = {("name", "Alice"): h1, ("drink", "tea"): h2}
    assert generate.holds(clue, assign) is expected


@pytest.mark.parametrize("kind", ["same", "diff", "immleft", "next", "left"])
def test_holds_never_falsifies_an_unknown_position(kind):
    clue = rel(kind, "name", "Alice", "drink", "tea")
    assert generate.holds(clue, {}) is True
    assert generate.holds(clue, {("name", "Alice"): 3}) is True
    assert generate.holds(clue, {("drink", "tea"): 1}) is True


def test_holds_for_a_position_clue():
    clue = pos("name", "Alice", 2)
    assert generate.holds(clue, {}) is True
    assert generate.holds(clue, {("name", "Alice"): 2}) is True
    assert generate.holds(clue, {("name", "Alice"): 1}) is False


def test_holds_rejects_an_unknown_kind_once_both_positions_are_known():
    with pytest.raises(ValueError):
        generate.holds(rel("bogus", "name", "Alice", "drink", "tea"),
                    {("name", "Alice"): 1, ("drink", "tea"): 2})


# ---------------------------------------------------------------------------
# generate / render
# ---------------------------------------------------------------------------


def test_render_bytes():
    clues = [("The name Alice is in house 1.", "pos", ("name", "Alice", 1))]
    assert generate.render(2, ["name", "drink"], clues) == (
        "There are 2 houses in a row, numbered 1 to 2 from left to right.\n"
        "Each house has a different name and drink.\n"
        "\n"
        "The attributes are:\n"
        "  name: Alice, Bob\n"
        "  drink: tea, coffee\n"
        "\n"
        "Clues:\n"
        "  1. The name Alice is in house 1.\n"
        "\n"
        "Work out the complete grid. Give your final answer as one line per "
        "house, in the form:\n"
        "  House 1: name=..., drink=..."
    )


def test_generate_is_deterministic_per_seed():
    assert generate.generate(3, 7) == generate.generate(3, 7)
    assert generate.generate(3, 7) != generate.generate(3, 8)


def test_every_true_clue_is_true_of_its_solution():
    rng = random.Random(5)
    cats = list(generate.CATEGORIES)[:4]
    sol = generate.make_solution(4, cats, rng)
    clues = generate.all_true_clues(sol, cats, rng)
    full = assignment(sol, cats)
    assert {k for _, k, _ in clues} == {"pos", "same", "diff", "immleft", "next", "left"}
    assert all(generate.holds(cl, full) for cl in clues)


def test_the_committed_puzzle_files_are_exactly_the_known_set():
    found = sorted(p.name for p in ZEBRA_DIR.glob("zebra_*x*_s*.json"))
    assert found == sorted(puzzle_path(n, s).name for n, s in COMMITTED)


@pytest.mark.parametrize("n, seed", REGENERATE)
def test_regenerating_a_committed_puzzle_reproduces_it(n, seed):
    sol, cats, clues = generate.generate(n, seed)
    d = load_puzzle(n, seed)

    assert d["n"] == n and d["seed"] == seed
    assert d["cats"] == cats
    assert d["prompt"] == generate.render(n, cats, clues)
    assert d["solution"] == {c: sol[c] for c in cats}
    assert d["n_clues"] == len(clues)

    # Byte for byte, as the generator writes it. Line endings are normalised
    # because git may check the file out with CRLF.
    written = json.dumps({"n": n, "seed": seed, "cats": cats,
                          "prompt": generate.render(n, cats, clues),
                          "solution": {c: sol[c] for c in cats},
                          "n_clues": len(clues)}, indent=1).encode()
    on_disk = puzzle_path(n, seed).read_bytes().replace(b"\r\n", b"\n")
    assert on_disk == written
    # ...and the record the generator and the runner both build is that dict.
    record = generate.puzzle_record(n, seed, sol, cats, clues)
    assert json.dumps(record, indent=1).encode() == written


@pytest.mark.parametrize("n, seed", CERTIFY)
def test_a_committed_puzzle_has_one_solution_and_no_spare_clue(n, seed):
    d = load_puzzle(n, seed)
    cats = d["cats"]
    clues = clues_from_prompt(d["prompt"])

    # The parser read the prompt exactly.
    assert len(clues) == d["n_clues"]
    assert generate.render(n, cats, clues) == d["prompt"]

    # Every clue is true of the committed solution...
    full = assignment(d["solution"], cats)
    assert all(generate.holds(cl, full) for cl in clues)
    # ...which is therefore the one solution.
    assert generate.count_solutions(clues, cats, n) == 1
    # And the set is minimal: without any single clue it is no longer unique.
    for i in range(len(clues)):
        trial = clues[:i] + clues[i + 1:]
        assert generate.count_solutions(trial, cats, n) != 1, clues[i][0]


@pytest.mark.parametrize("n, seed", [(3, 1), (3, 2), (4, 1), (4, 2)])
def test_brute_force_confirms_small_committed_puzzles_are_unique(n, seed):
    d = load_puzzle(n, seed)
    clues = clues_from_prompt(d["prompt"])
    assert generate.count_solutions_bruteforce(clues, d["cats"], n) == 1


def test_committed_solutions_are_permutations_of_each_category():
    for n, seed in COMMITTED:
        d = load_puzzle(n, seed)
        assert d["cats"] == list(generate.CATEGORIES)[:n]
        for c in d["cats"]:
            assert sorted(d["solution"][c]) == sorted(generate.CATEGORIES[c][:n])


# ---------------------------------------------------------------------------
# experiments/zebra/run.py, imported without running main()
# ---------------------------------------------------------------------------


@pytest.fixture
def run(monkeypatch):
    """experiments/zebra/run.py as a freshly imported module.

    It used to call asyncio.run(main()) at import, with no __main__ guard.
    asyncio.run is replaced by a stand-in that records and closes anything it
    is given, and the import must give it nothing - nor touch sys.path, which
    the old runner extended to find its imports. The module in sys.modules
    and sys.argv are restored after the test."""
    monkeypatch.setattr(sys, "argv", ["run.py"])
    monkeypatch.delitem(sys.modules, "experiments.zebra.run", raising=False)
    path_before = list(sys.path)
    started = []

    def close_unstarted(coro):
        started.append(coro.__qualname__)
        coro.close()

    with monkeypatch.context() as m:
        m.setattr(asyncio, "run", close_unstarted)
        module = importlib.import_module("experiments.zebra.run")
    assert started == []
    assert sys.path == path_before
    assert module.generate is generate  # the generator module next to it
    assert os.path.samefile(module.HERE, ZEBRA_DIR)
    return module


def test_running_the_module_as_a_script_starts_main(monkeypatch):
    import runpy

    monkeypatch.setattr(sys, "argv", ["run.py"])
    monkeypatch.delitem(sys.modules, "experiments.zebra.run", raising=False)
    started = []

    def close_unstarted(coro):
        started.append(coro.__qualname__)
        coro.close()

    monkeypatch.setattr(asyncio, "run", close_unstarted)
    runpy.run_module("experiments.zebra.run", run_name="__main__")
    assert started == ["main"]


TWO = {"n": 2, "cats": ["name", "drink"],
       "solution": {"name": ["Alice", "Bob"], "drink": ["tea", "coffee"]}}


def test_grade_accepts_the_exact_grid(run):
    text = "House 1: name=Alice, drink=tea\nHouse 2: name=Bob, drink=coffee"
    assert run.grade(text, TWO) == (
        True,
        {1: {"name": "alice", "drink": "tea"}, 2: {"name": "bob", "drink": "coffee"}},
        "4/4",
    )


def test_grade_is_case_and_spacing_insensitive(run):
    text = "house 1: NAME=ALICE, Drink = Tea\nHOUSE 2 : name =bob , drink= coffee"
    ok, rows, score = run.grade(text, TWO)
    assert (ok, score) == (True, "4/4")
    assert rows[2] == {"name": "bob", "drink": "coffee"}


def test_grade_lets_a_later_mention_override_a_draft(run):
    text = ("Draft:\nHouse 1: name=Bob, drink=coffee\nHouse 2: name=Alice, drink=tea\n"
            "Final:\nHouse 1: name=Alice, drink=tea\nHouse 2: name=Bob, drink=coffee")
    assert run.grade(text, TWO)[0] is True
    # ...and the other way round the draft's correction is what counts.
    swapped = ("House 1: name=Alice, drink=tea\nHouse 2: name=Bob, drink=coffee\n"
               "Actually:\nHouse 1: name=Bob, drink=tea")
    assert run.grade(swapped, TWO)[2] == "3/4"


def test_grade_a_later_mention_without_fields_changes_nothing(run):
    text = ("House 1: name=Alice, drink=tea\nHouse 2: name=Bob, drink=coffee\n"
            "So House 1 is settled.")
    assert run.grade(text, TWO)[0] is True


def test_grade_ignores_houses_out_of_range(run):
    text = ("House 0: name=Bob\nHouse 1: name=Alice, drink=tea\n"
            "House 2: name=Bob, drink=coffee\nHouse 3: name=Carol, drink=milk")
    ok, rows, score = run.grade(text, TWO)
    assert ok is True
    assert sorted(rows) == [1, 2]


def test_grade_needs_key_value_pairs(run):
    assert run.grade("House 1: Alice, tea\nHouse 2: Bob, coffee", TWO) == (False, {}, "0/4")


def test_grade_counts_partial_answers(run):
    assert run.grade("House 1: name=Alice, drink=tea", TWO) == (
        False, {1: {"name": "alice", "drink": "tea"}}, "2/4")
    assert run.grade("House 1: name=Bob, drink=tea\nHouse 2: name=Alice, drink=coffee",
                     TWO)[2] == "2/4"
    assert run.grade("", TWO) == (False, {}, "0/4")


def test_grade_reads_through_markdown(run):
    text = "**House 1:** name=Alice, drink=tea\n- House 2: name=Bob, drink=coffee"
    assert run.grade(text, TWO)[0] is True


@pytest.mark.parametrize("n, seed", COMMITTED)
def test_grade_gives_full_marks_to_each_committed_solution(run, n, seed):
    d = load_puzzle(n, seed)
    ok, rows, score = run.grade(answer_text(d), d)
    assert ok is True
    assert score == f"{n * n}/{n * n}"


@pytest.mark.parametrize("n, seed", COMMITTED)
def test_grade_gives_nothing_for_echoing_the_puzzle(run, n, seed):
    d = load_puzzle(n, seed)
    ok, rows, score = run.grade(d["prompt"], d)
    assert ok is False
    assert rows == {}
    assert score == f"0/{n * n}"


def test_effort_sentences(run):
    assert set(run.EFFORT) == {"low", "medium", "xhigh"}
    assert run.EFFORT["medium"] == ""
    assert run.EFFORT["low"].startswith("Reasoning effort is set to low.")
    assert run.EFFORT["xhigh"].startswith("Reasoning effort is set to xhigh.")
    assert all(s == "" or s.endswith("\n\n") for s in run.EFFORT.values())


def test_system_for_puts_the_effort_sentence_before_the_system_prompt(run):
    assert run.system_for("medium") == prompts.DEFAULT_SYSTEM
    assert run.system_for("low") == run.EFFORT["low"] + prompts.DEFAULT_SYSTEM
    with pytest.raises(KeyError):
        run.system_for("maximum")


@pytest.mark.parametrize("n, seed", COMMITTED)
def test_puzzle_reads_the_committed_file(run, n, seed):
    before = puzzle_path(n, seed).read_bytes()
    assert run.puzzle(n, seed) == load_puzzle(n, seed)
    assert puzzle_path(n, seed).read_bytes() == before


def _lf(path):
    # Line endings normalised: git may check a file out with CRLF, and the
    # generator writes in text mode, so with the platform's line ending.
    return path.read_bytes().replace(b"\r\n", b"\n")


def _committed_bytes(n, seed):
    return _lf(puzzle_path(n, seed))


def test_puzzle_generates_and_caches_a_missing_puzzle_next_to_the_runner(run, monkeypatch, tmp_path):
    monkeypatch.setattr(run, "HERE", str(tmp_path))

    d = run.puzzle(3, 2)

    assert d == load_puzzle(3, 2)
    assert _lf(tmp_path / "zebra_3x3_s2.json") == _committed_bytes(3, 2)


def test_the_generator_keeps_puzzles_in_its_own_directory():
    assert os.path.samefile(os.path.dirname(generate.puzzle_path(3, 1)), ZEBRA_DIR)
    assert os.path.basename(generate.puzzle_path(3, 1)) == "zebra_3x3_s1.json"
    assert generate.puzzle_path(4, 2, "somewhere") == os.path.join("somewhere", "zebra_4x4_s2.json")


def test_generate_main_writes_next_to_itself_not_into_the_cwd(monkeypatch, tmp_path, capsys):
    # zebra.py's __main__ wrote into the current directory; generate.py
    # writes where run.py reads the puzzles from.
    here, cwd = tmp_path / "here", tmp_path / "cwd"
    here.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(generate, "HERE", str(here))

    generate.main(["3", "1"])

    assert list(cwd.iterdir()) == []
    assert _lf(here / "zebra_3x3_s1.json") == _committed_bytes(3, 1)
    d = load_puzzle(3, 1)
    out = capsys.readouterr().out
    assert out.startswith(d["prompt"] + "\n\nSOLUTION\n")
    assert out.endswith(f"\nclues={d['n_clues']}  unique=True\n")
    assert "  House 1: " + ", ".join(f"{c}={d['solution'][c][0]}" for c in d["cats"]) in out


def test_generate_main_defaults_to_4x4_seed_1(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(generate, "HERE", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["generate.py"])

    generate.main()

    assert [p.name for p in tmp_path.iterdir()] == ["zebra_4x4_s1.json"]
    assert _lf(tmp_path / "zebra_4x4_s1.json") == _committed_bytes(4, 1)


# ---------------------------------------------------------------------------
# the native arm, against a mock endpoint
# ---------------------------------------------------------------------------


class _Recorder:
    """A MockTransport handler: a Router that also keeps each request's
    raw body and timeout."""

    def __init__(self, router):
        self.router = router
        self.bodies, self.timeouts = [], []

    def __call__(self, request):
        self.bodies.append(request.content)
        self.timeouts.append(request.extensions.get("timeout"))
        return self.router(request)


def _native_endpoint(body, max_model_len=4096, status=200):
    router = (Router()
              .add("GET", "/v1/models",
                   json_response({"data": [{"id": "test-model", "max_model_len": max_model_len}]}))
              .add("POST", "/v1/completions",
                   httpx.Response(status, content=body,
                                  headers={"content-type": "text/event-stream"})))
    return _Recorder(router)


def _piece(text, finish=None):
    return {"choices": [{"index": 0, "text": text, "finish_reason": finish}]}


def _native_prompt(system, prompt_text):
    return (chatml.turn("system", system) + chatml.turn("user", prompt_text)
            + chatml.assistant_header(think=True))


def test_native_sends_the_same_request_and_keeps_the_raw_stream(run, monkeypatch, tmp_path):
    monkeypatch.setattr(run, "HERE", str(tmp_path))
    # A comment line (skipped), the pieces, [DONE], and a chunk after it that
    # must never be read.
    body = (b": keep-alive\n\n"
            + sse(_piece("Case A fails."), _piece(" So B.</think>"),
                  _piece("\n\nHouse 1: name=Alice", finish="stop"))
            + sse(_piece(" never read"), done=False))
    handler = _native_endpoint(body)

    with mock_transport(handler):
        answer, meta = asyncio.run(run.native("Solve it.", "SYS"))

    prompt = _native_prompt("SYS", "Solve it.")
    budget = 4096 - len(prompt) // 2 - 64
    assert answer == "House 1: name=Alice"
    assert meta == {"secs": meta["secs"], "finish": "stop", "closed": True,
                    "think_chars": len("Case A fails. So B."), "budget": budget}
    assert list(meta) == ["secs", "finish", "closed", "think_chars", "budget"]
    assert (tmp_path / "native_raw.txt").read_text(encoding="utf-8") == (
        "Case A fails. So B.</think>\n\nHouse 1: name=Alice")

    models, completion = handler.router.requests
    assert (models.method, models.url) == ("GET", "http://vllm.test/v1/models")
    assert (completion.method, completion.url) == ("POST", "http://vllm.test/v1/completions")
    for seen in (models, completion):
        assert seen.headers["authorization"] == "Bearer test-key"
        assert seen.headers["user-agent"] == config.USER_AGENT
    assert "content-type" not in models.headers
    assert completion.headers["content-type"] == "application/json"
    # The payload, byte for byte: key order, and temperature the int 0.
    payload = {"model": "test-model", "prompt": prompt, "temperature": 0,
               "stop": [chatml.IM_END], "stream": True, "max_tokens": budget}
    assert handler.bodies[1] == httpx.Request("POST", completion.url, json=payload).content
    assert handler.bodies[0] == b""
    assert [t["read"] for t in handler.timeouts] == [60, 7200]
    assert [t["connect"] for t in handler.timeouts] == [60, 7200]
    handler.router.assert_consumed()


def test_native_without_a_closing_tag_has_no_answer(run, monkeypatch, tmp_path):
    monkeypatch.setattr(run, "HERE", str(tmp_path))
    handler = _native_endpoint(sse(_piece("Case A. Case B. Dr", finish="stop")))

    with mock_transport(handler):
        answer, meta = asyncio.run(run.native("Solve it.", "SYS"))

    assert answer == ""
    assert (meta["closed"], meta["finish"]) == (False, "stop")
    assert meta["think_chars"] == len("Case A. Case B. Dr")


def _ours_fake(cfg, fake_llm, d):
    """One turn: an opening thought of two beams that close at once, then the
    committed solution as the answer, one house per step."""
    cfg(OPEN_THINK=True, BEAM_WIDTH=2, THINK_BRANCH_K=2, ENTROPY_FLOOR=1e9,
        WINDOW_TOKENS=8, KEEP_BLOCKS_INLINE=True)
    lines = answer_text(d).split("\n")
    steps = [lines[0]] + ["\n" + line for line in lines[1:]]
    return fake_llm(fanout=lambda prompts, k: openings("Let", "We"),
                    chunks=lambda prompts: [closed(" check each clue.") for _ in prompts],
                    streams=[quiet(*steps)])


def _use_fake_client(monkeypatch, run, fake):
    """ours() builds its own client; make it the fake."""
    monkeypatch.setattr(run, "VLLM", lambda: fake)


def test_ours_runs_the_engine_and_reports_its_blocks(run, cfg, fake_llm, monkeypatch, capsys):
    d = load_puzzle(3, 1)
    fake = _ours_fake(cfg, fake_llm, d)
    _use_fake_client(monkeypatch, run, fake)

    text, meta = asyncio.run(run.ours(d["prompt"], "SYS"))

    assert text == answer_text(d)
    assert list(meta) == ["secs", "blocks", "explored", "restarts", "retries", "degraded",
                          "closed", "think_chars"]
    assert (meta["blocks"], meta["closed"], meta["think_chars"]) == (1, True, None)
    assert (meta["restarts"], meta["degraded"]) == (0, None)
    (line,) = capsys.readouterr().out.splitlines()
    assert "] block 0 " in line and " checkpoints=1 finished=2 explored=" in line
    assert fake.closed is True
    # The system prompt reaches the model as given.
    assert fake.of("stream")[0]["prompt"].startswith(chatml.turn("system", "SYS"))
    fake.assert_consumed()


def test_main_grades_the_answer_and_appends_one_result(run, cfg, fake_llm, monkeypatch,
                                                       tmp_path, capsys):
    d = load_puzzle(3, 1)
    (tmp_path / puzzle_path(3, 1).name).write_bytes(puzzle_path(3, 1).read_bytes())
    monkeypatch.setattr(run, "HERE", str(tmp_path))
    fake = _ours_fake(cfg, fake_llm, d)
    _use_fake_client(monkeypatch, run, fake)
    monkeypatch.setattr(sys, "argv", ["run.py", "ours", "3", "1", "medium"])

    asyncio.run(run.main())

    out = capsys.readouterr().out
    assert out.startswith(f"=== ours  3x3 seed1  ({d['n_clues']} clues)  effort=medium\n")
    assert "    RESULT PASS   cells 9/9\n" in out
    assert out.endswith("    --- answer tail ---\n    " + " ".join(answer_text(d).split()) + "\n")
    (line,) = (tmp_path / "zebra_results.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert list(record)[:5] == ["arm", "n", "seed", "ok", "cells"]
    assert {k: record[k] for k in ("arm", "n", "seed", "ok", "cells", "effort", "blocks")} == {
        "arm": "ours", "n": 3, "seed": 1, "ok": True, "cells": "9/9", "effort": "medium",
        "blocks": 1}
    assert list(record)[-1] == "effort"


def test_native_raises_on_an_error_status(run, monkeypatch, tmp_path):
    monkeypatch.setattr(run, "HERE", str(tmp_path))
    handler = _native_endpoint(b"overloaded", status=503)

    with mock_transport(handler), pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run.native("Solve it.", "SYS"))
    assert len(handler.router.requests) == 2
