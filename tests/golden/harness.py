"""Golden characterisation tests: the machinery.

A golden is a byte-exact recording of what the engine does in one scripted
situation: every request it sends, every event it emits, and the conversation
it leaves behind. A behaviour-preserving refactor must reproduce every one of
them. tests/test_golden.py compares a fresh recording with the stored one;
scenarios.py holds the scripts.

HOW IT RUNS
    The REAL llm.VLLM and the REAL engine.Engine talk to an
    httpx.MockTransport (fake_endpoint.mock_transport swaps the transport of
    every httpx.AsyncClient built while it is active, and nothing else), so the
    request payloads are captured exactly as the client serialises them. The
    handler is FakeServer below: a stand-in for a vLLM server that answers
    from a script.

THE FAKE SERVER IS A PURE FUNCTION
    Each response depends only on the scenario script, the request bytes and,
    for failure injection and the flaky fan-out read, how many times those
    exact bytes were requested before. Nothing reads a clock or an unseeded
    random source; wherever the script leaves a detail open (the
    alternatives at an answer token, a filler thought, a default gate score)
    the detail is drawn from hashlib.sha256 of the request text.

    It reads each prompt the way a model would: which question is being
    answered, what of the answer is visible, where the inline notes sit, and
    whether a thought is being written. The script says what the model
    writes next from there. So when the engine sends a different prompt, the
    fake answers differently, and the golden diff shows both.

WHAT IS PINNED
    KNOBS below sets every environment-driven config value explicitly, so
    neither a developer's .env nor a change of code default can move a golden.
    RETIRED_KNOBS lists knobs removed from config since the goldens were
    recorded; they appear in the recorded header but are not pinned.
    Prompt wording (prompts.AXES, prompts.GATE_AXES, chatml.NO_THINK, ...) is
    deliberately NOT pinned: it is part of the code under test. time.time is
    frozen, so every `secs` is 0.0 and events.jsonl's `t` is a constant. The
    labels come from the fake server's /tokenize: llm.labels() has no other
    source, and the label package it used to try first stays blocked while
    recording (tests/conftest.py blocks it for the whole session as well).

WHAT IS RECORDED
    Every recording starts with the knobs it ran under. For the engine, per
    turn:
    requests  method, path, api ("v1" or "root"), the Authorization /
              User-Agent / Content-Type headers, and the JSON body as parsed
              (compared as json.dumps(sort_keys=True), so 0 and 0.0 differ)
    events    json.dumps(event, ensure_ascii=False) - key order included
    error     the exception that ended the turn, if one did
    render    Conversation.render() after the turn
    notes     (turn, off, text, born) for every note still in context
    clock     Conversation.clock
    and the client's usage counters at the end of the run. For cli.py: the
    requests and stdout (ANSI codes included) per output mode. For
    server.py: /config, the requests and SSE body of each /chat, the
    events.jsonl it appended, and /reset.

    The first difference is reported in recording order, and within a turn
    the requests come before the events, so it usually points at the cause.

ONE NORMALISATION
    httpx appends "For more information check: <link>" to the message of an
    HTTPStatusError. That line is httpx's wording, not the engine's, and it
    may change with the httpx version, so it is removed wherever it appears.

WHERE THE ENGINE IS IMPORTED
    From the adaptive_thinking package, in exactly one place: the ENGINE
    IMPORTS block below, plus the imports of adaptive_thinking.cli and
    adaptive_thinking.server inside record_cli and record_server. A move
    inside the package should only need those lines to change.
"""

import ast
import asyncio
import contextlib
import hashlib
import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path as FsPath
from unittest import mock

import httpx

# -- ENGINE IMPORTS ------------------------------------------------------------
from adaptive_thinking import config, llm
from adaptive_thinking.engine import Engine
from adaptive_thinking.state import Conversation
from fake_endpoint import mock_transport

GOLDEN_DIR = FsPath(__file__).resolve().parent
FORMAT = 1

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

# Every config value that comes from the environment, with the value every
# scenario starts from. Types matter as much as values: WINDOW_TOKENS goes
# into a request body as max_tokens, and 8 and 8.0 are different bytes.
KNOBS = {
    # endpoint
    "BASE_URL": "http://vllm.test/v1",
    "ROOT_URL": "http://vllm.test",      # derived from BASE_URL in config
    "MODEL": "test-model",
    "API_KEY": "test-key",
    "USER_AGENT": "adaptive-thinking/0.1",
    "TIMEOUT": 300.0,
    "LLM_RETRIES": 2,
    "LLM_RETRY_BACKOFF": 0.0,
    # the gate
    "ENTROPY_FLOOR": 0.4,
    "MIN_GAP_TOKENS": 6,
    "WARMUP_TOKENS": 6,
    "WINDOW_TOKENS": 8,
    "PROBE_CANDIDATES": 3,
    "PROBE_THRESHOLD": 0.6,
    "PROBE_ENABLED": True,
    "GATE_THRESHOLD": 0.6,
    "GATE_LOG_YESNO": True,
    # the tree
    "BEAM_WIDTH": 4,
    "THINK_CHUNK": 64,
    "THINK_BRANCH_K": 5,
    "CHOICE_SHORTLIST": 8,
    "MAX_CHILDREN_PER_PARENT": 1,
    "CHURN_MARGIN": 0.12,
    "CHURN_PATIENCE": 2,
    "SIBLING_PENALTY": 0.35,
    "COPY_PENALTY": 0.6,
    "COPY_NGRAM": 24,
    "CHOICE_EVERY": 0,
    "CHOICE_DEBIAS": True,
    "AXIS_CLIP": 2.0,
    "STOP_AFTER_FINISHED": 3,
    "STOP_MARGIN": 0.35,
    "RAMBLE_RATIO": 2.5,
    # the answer and its notes
    "BRANCH_K": 5,
    "KEEP_BLOCKS_INLINE": True,
    "INLINE_SEED": "<think>Hmm, hold on.",
    "OPEN_THINK": True,
    "OPEN_SEED": "<think>\n",
    "ANCHOR_CHARS": 90,
    "RESUME_WATCH_TOKENS": 4,
    "NATIVE_MAX_TOKENS": 1600,
    # context
    "EVICT_AGE_TOKENS": 100000,
    "EVICT_KEEP_TOKENS": 20000,
    "PORT": 8099,
}
# Pinned although no environment variable sets it directly.
DERIVED_KNOBS = {"ROOT_URL"}

# Knobs config.py no longer defines, each with the KNOBS entry it followed
# and the value the stored goldens were recorded under. Nothing ever read
# them, so removing them changed no behaviour. They are not pinned, since
# config has no such attribute to set; they are only written back into each
# recording's "knobs" header at their old position, so that the stored
# goldens stay byte-identical. Drop an entry here, together with its key in
# the goldens, the next time the goldens are regenerated on purpose.
RETIRED_KNOBS = {"THINK_ALOUD": ("KEEP_BLOCKS_INLINE", True)}


def recorded_knobs(values):
    """The "knobs" header of a recording: `values`, with each of
    RETIRED_KNOBS put back after the knob it used to follow."""
    out = {}
    for name, value in values.items():
        out[name] = value
        for retired, (after, old_value) in RETIRED_KNOBS.items():
            if name == after:
                out[retired] = old_value
    return out


def env_knob_names():
    """Names config assigns from the environment, read from config's source:
    any module-level assignment whose value calls _f/_i/_b or os.getenv."""
    src = FsPath(config.__file__).read_text(encoding="utf-8")
    names = set()
    for node in ast.parse(src).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Call) and (
                (isinstance(sub.func, ast.Name) and sub.func.id in ("_f", "_i", "_b"))
                or (isinstance(sub.func, ast.Attribute) and sub.func.attr == "getenv")
            ):
                names.add(node.targets[0].id)
                break
    return names


def knobs(**overrides):
    """KNOBS with some values changed. Unknown names are an error."""
    unknown = sorted(set(overrides) - set(KNOBS))
    if unknown:
        raise KeyError(f"not a pinned knob: {unknown}")
    return {**KNOBS, **overrides}


def check_knobs(values):
    """Every environment knob is pinned, exists, and keeps config's type."""
    missing = sorted(env_knob_names() - set(values))
    if missing:
        raise AssertionError(f"KNOBS does not pin {missing}; add them to tests/golden/harness.py")
    for name, value in values.items():
        if not hasattr(config, name):
            raise AssertionError(f"config has no attribute {name!r}")
        have = getattr(config, name)
        if type(have) is not type(value):
            raise AssertionError(
                f"{name}: pinned {value!r} ({type(value).__name__}) but config holds "
                f"{type(have).__name__}")


@contextlib.contextmanager
def pinned(values):
    """Set every knob for the duration, then restore. Anything that assigns
    config attributes directly in between (server.py's `inline`) is undone
    too, because patch.multiple restores the values it saw on entry."""
    check_knobs(values)
    with mock.patch.multiple(config, **values):
        yield


FROZEN_TIME = 1700000000.0


@contextlib.contextmanager
def frozen_clock():
    """time.time() is a constant, so no wall-clock value reaches a golden."""
    with mock.patch("time.time", return_value=FROZEN_TIME):
        yield


@contextlib.contextmanager
def no_optional_labels():
    """Blocks the label package llm.labels() used to try before /tokenize.
    labels() no longer imports it, and the goldens were recorded with it
    blocked; the block stays as a guard. tests/conftest.py blocks it too."""
    with mock.patch.dict(sys.modules, {"jevfire": None}):
        yield


# ---------------------------------------------------------------------------
# Reading a prompt, as the fake model sees it
# ---------------------------------------------------------------------------
# The fake knows the ChatML format and the prompt wording independently of
# config, so a change on the engine side shows up as a different answer from
# the fake rather than being followed silently.

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
USER = IM_START + "user\n"
ASSISTANT = IM_START + "assistant\n"
HEAD_OPEN, HEAD_CLOSE = "<think>\n", "\n</think>\n\n"
THINK_CLOSE = "</think>"
MARKERS = (IM_START, IM_END, "<think>", "</think>")
REASONING = "Here is a piece of reasoning:\n\n"

GATE_QUESTIONS = {
    "stakes": "what rests on getting it right?",
    "difficulty": "How hard is the next step actually going to be to get right?",
    "commitment": "How far does the next step lock the answer in?",
}
JUDGE_QUESTIONS = {
    "looping": "Read the reasoning above. How much of it is repetition?",
    "progress": "How far has this reasoning actually got?",
    "confusion": "Is the reasoning above internally sound?",
    "concreteness": "How specific is this reasoning?",
}
YESNO_QUESTION = "is the next thing you are about to say a point where getting it wrong"


class OffScript(Exception):
    """The engine sent something the fake cannot read. Not transient, so the
    turn fails loudly and the recording says where."""


def _h(*parts):
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()


def cut_markers(text):
    at = min([i for i in (text.find(m) for m in MARKERS) if i != -1], default=len(text))
    return text[:at]


def tail_words(vis):
    """The anchor the note replays after `</think>`: the last ANCHOR_CHARS
    characters of the answer, from the first word boundary."""
    t = vis[-240:][-KNOBS["ANCHOR_CHARS"]:]
    i = t.find(" ")
    return t[i + 1:] if i > 0 else t


def at_boundary(vis):
    before = vis.rstrip()
    return (not before) or before[-1] in ".!?:;" or vis.endswith("\n")


@dataclass
class View:
    question: str
    head: str = None        # the header thought; None before the answer has a header
    vis: str = ""           # the visible answer, notes and their anchors removed
    notes: tuple = ()       # offsets in `vis` where inline notes sit
    thinking: str = None    # an unfinished thought at the end of the prompt


def parse_prompt(prompt):
    """What an assistant turn in progress looks like to the model."""
    a = prompt.rfind(ASSISTANT)
    if a == -1:
        raise OffScript(f"no assistant turn in prompt ending {prompt[-80:]!r}")
    u = prompt.rfind(USER, 0, a)
    question = prompt[u + len(USER): prompt.find(IM_END, u)] if u != -1 else ""
    body = prompt[a + len(ASSISTANT):]
    if not body:
        return View(question)                       # the opening thought's context
    if not body.startswith(HEAD_OPEN):
        raise OffScript(f"assistant turn does not open with a thinking slot: {body[:80]!r}")
    e = body.find(HEAD_CLOSE)
    if e == -1:
        return View(question, thinking=body[len(HEAD_OPEN):])   # the opening thought
    view = View(question, head=body[len(HEAD_OPEN):e])
    rest, vis, notes = body[e + len(HEAD_CLOSE):], "", []
    seed = KNOBS["INLINE_SEED"]
    while True:
        k = rest.find(seed)
        if k == -1:
            break
        vis += rest[:k]
        c = rest.find(THINK_CLOSE, k)
        if c == -1:                                  # a mid-answer thought
            view.thinking = rest[k + len(seed):]
            rest = ""
            break
        notes.append(len(vis))
        rest = rest[c + len(THINK_CLOSE):]
        anchor = tail_words(vis)
        if anchor and rest.startswith(anchor):
            rest = rest[len(anchor):]
    view.vis, view.notes = vis + rest, tuple(notes)
    return view


def split_question(prompt):
    """A closed question is the context, the turn closed, and a user turn
    asking about it. Returns (context, question body) or (None, None)."""
    i = prompt.rfind(IM_END + "\n" + USER)
    if i == -1:
        return None, None
    body = prompt[i + len(IM_END + "\n" + USER):]
    return prompt[:i], body[: body.find(IM_END)]


def classify(body):
    """("judge", axis, thought) / ("gate", axis) / ("yesno",) / ("unknown",)."""
    if body.startswith(REASONING):
        for axis, q in JUDGE_QUESTIONS.items():
            j = body.rfind("\n\n" + q)
            if j != -1:
                return ("judge", axis, body[len(REASONING):j])
        return ("unknown",)
    for axis, q in GATE_QUESTIONS.items():
        if q in body:
            return ("gate", axis)
    if YESNO_QUESTION in body:
        return ("yesno",)
    return ("unknown",)


# ---------------------------------------------------------------------------
# The script: what the model writes
# ---------------------------------------------------------------------------


@dataclass
class Tok:
    text: str
    gate: str = ""      # "^" the gate rates the position before this token as
                        # critical, "~" a near miss, "" the default (low)
    flat: bool = False  # no alternatives at all (entropy 0)


class Path:
    """One way an answer can go, written "The| answer| is|^ 4|.".

    "|" separates tokens. "‖" separates tokens AND says a thinking note sits
    there: this path is what the model writes when the context holds a note
    at that offset. A token may be prefixed "^" (the gate fires at the
    position before it), "~" (the gate scores it just under the threshold) or
    "=" (the token has no alternatives).
    """

    # The ChatML markers contain "|", so they are shielded from the split.
    _SHIELD = {IM_START: "\x00s", IM_END: "\x00e"}

    def __init__(self, spec):
        self.spec = spec
        self.tokens, self.cum, self.notes = [], [""], set()
        for marker, stand_in in self._SHIELD.items():
            spec = spec.replace(marker, stand_in)
        for piece in re.split(r"(\||‖)", spec):
            if piece in ("|", ""):
                continue
            if piece == "‖":
                self.notes.add(len(self.cum[-1]))
                continue
            for marker, stand_in in self._SHIELD.items():
                piece = piece.replace(stand_in, marker)
            gate, flat = "", False
            while piece[:1] in ("^", "~", "="):
                if piece[0] == "=":
                    flat = True
                else:
                    gate = piece[0]
                piece = piece[1:]
            self.tokens.append(Tok(piece, gate, flat))
            self.cum.append(self.cum[-1] + piece)
        self._at = {s: i for i, s in enumerate(self.cum)}

    def index(self, vis):
        return self._at.get(vis)


@dataclass
class Chunk:
    """One chunk of a thought: `alts` are the alternatives at its LAST token,
    best first, and the first is normally that token itself. finish "stop"
    means the thought closed itself (with or without `</think>` in the text);
    no_top sends no alternatives at all."""

    text: str
    lp: float
    alts: tuple = ()
    finish: str = "length"
    no_top: bool = False


FILLERS = (
    " So the carry goes into the hour.",
    " That gives the time directly.",
    " Checking once more, it holds.",
    " The minutes add up without a carry.",
    " Nothing else changes the result.",
    " Counting on from the hour works too.",
)
# A thought that is still only its opening word goes on from that word.
OPENING_FILLERS = {
    "Let": (" me check the minutes once more.", " me count on from the hour."),
    "First": (", the hours: that part is fixed.", ", add the whole hours."),
    "We": (" can count on from the hour.", " need the minutes first."),
    "So": (" the carry goes into the hour.", " the hour changes once."),
}


def fallback_chunk(text, salt):
    """How any thought the script does not know carries on: one sentence, and
    it closes. So every block ends, whatever the engine does."""
    h = _h("fallback", salt, text)
    options = OPENING_FILLERS.get(text, FILLERS)
    sentence = options[h[0] % len(options)]
    closing = THINK_CLOSE if h[1] & 1 else ""
    last = closing or sentence.split(" ")[-1]
    return Chunk(sentence + closing, -(1.5 + (h[2] % 8) * 0.25),
                 alts=((last, -0.05),), finish="stop")


class BlockScript:
    """What happens inside one thinking block.

    openings    the fan-out read: [(token, logprob)], any order
    first_read  if set, the FIRST read of this block's fan-out returns this
                instead (a truncated, degenerate read); the retry gets
                `openings`
    lineages    {opening token: [Chunk, ...]} - the chunks a conversation
                that opened with that token writes, in order, as long as it
                stays on script. Anything off script continues with
                fallback_chunk().
    grades      {opening token: [levels after chunk 1, after chunk 2, ...]}
                where levels is 0..4 for every judge axis or a 4-tuple
                (looping, progress, confusion, concreteness). A thought that
                is a variation of a scripted one (an alternative last token)
                scores one level lower on every axis. Unknown thoughts get
                levels 1..3 drawn from the hash.
    """

    def __init__(self, openings, lineages=None, grades=None, first_read=None):
        self.openings = list(openings)
        self.first_read = list(first_read) if first_read else None
        self.lineages = {o: list(c) for o, c in (lineages or {}).items()}
        self.grades = dict(grades or {})

    def next_chunk(self, text, salt):
        for o, chunks in self.lineages.items():
            s = o
            for ch in chunks:
                if text == s:
                    return ch
                s += ch.text
        return fallback_chunk(text, salt)

    def levels(self, thought, salt):
        best = None
        for o, chunks in self.lineages.items():
            s = o
            for d in range(len(chunks) + 1):
                clean = cut_markers(s).strip()
                if clean and thought.startswith(clean):
                    if best is None or len(clean) > len(best[2]):
                        best = (o, d, clean)
                if d < len(chunks):
                    s += chunks[d].text
        if best is not None and best[0] in self.grades:
            o, d, clean = best
            table = self.grades[o]
            g = table[min(max(d, 1), len(table)) - 1]
            lv = [g] * 4 if isinstance(g, int) else list(g)
            if thought != clean:
                lv = [max(0, x - 1) for x in lv]
            return lv
        h = _h("grade", salt, thought)
        return [1 + h[i] % 3 for i in range(4)]


# Twenty openings, as many as a fan-out read asks for at width 4 x 5: junk
# ("!", whitespace, `</think>`) among them, which the fan-out must skip. The
# probabilities sum to just under 1.
GENERIC_OPENINGS = (
    ("Let", -1.0), ("!", -1.7), ("First", -2.1), (" ", -2.4), ("</think>", -2.8),
    ("We", -3.1), ("So", -3.4), ("The", -3.7), ("Wait", -4.0), ("\n", -4.3),
    ("Okay", -4.6), ("Hmm", -4.9), ("I", -5.2), ("Check", -5.5), ("-", -5.8),
    ("Now", -6.1), ("Since", -6.4), ("If", -6.7), ("Right", -7.0), ("To", -7.3),
)
GENERIC_BLOCK = BlockScript(GENERIC_OPENINGS)

# Logprobs for a chunk's last token and up to four alternatives (sum < 1).
ALT_LPS = (-0.2, -2.4, -3.1, -3.7, -4.4)


def alts(*tokens):
    """The alternatives at a chunk's last token, best first: alts(".", ",")."""
    return tuple(zip(tokens, ALT_LPS))


CLOSED = (("</think>", -0.05),)

FIRE = {"stakes": 4, "difficulty": 3, "commitment": 4}
NEAR = {"stakes": 2, "difficulty": 2, "commitment": 3}


class TurnScript:
    """How the model answers one question.

    paths   Path specs. For a prompt, the model follows a path whose text
            starts with the visible answer and whose notes are all present in
            the context; of those, the one expecting the most notes.
    blocks  {"open": BlockScript, "<end of the visible answer>": BlockScript}
            - the opening thought, and mid-answer blocks by where they open.
            Unlisted blocks use GENERIC_BLOCK.
    """

    def __init__(self, question, *paths, blocks=None):
        self.question = question
        self.paths = [Path(p) for p in paths]
        self.blocks = dict(blocks or {})
        self.gate = {}
        for p in self.paths:
            for i, t in enumerate(p.tokens):
                if t.gate:
                    at = p.cum[i]
                    if self.gate.get(at, t.gate) != t.gate:
                        raise ValueError(f"conflicting gate marks at {at!r}")
                    self.gate[at] = t.gate

    def follow(self, vis, notes):
        best = None
        for p in self.paths:
            i = p.index(vis)
            if i is None or not p.notes <= set(notes):
                continue
            if best is None or len(p.notes) > len(best[0].notes):
                best = (p, i)
        return best

    def block(self, view):
        if view.head is None:
            return self.blocks.get("open", GENERIC_BLOCK)
        for key, b in self.blocks.items():
            if key != "open" and view.vis.endswith(key):
                return b
        return GENERIC_BLOCK

    def gate_levels(self, vis, salt):
        mark = self.gate.get(vis)
        if mark == "^":
            return FIRE
        if mark == "~":
            return NEAR
        h = _h("gate", salt, vis)
        return {a: h[i] % 2 for i, a in enumerate(GATE_QUESTIONS)}


@dataclass
class Fail:
    """Answer requests of `kind` for which when(body) holds with an HTTP
    error, the first `times` times the same bytes arrive (None = always)."""

    kind: str
    status: int
    when: object = None
    times: int = 1

    def applies(self, kind, body, nth):
        return (kind == self.kind
                and (self.times is None or nth < self.times)
                and (self.when is None or self.when(body)))


class Model:
    """A scenario's script: the turns by question, plus injected failures.
    stream_usage adds a usage chunk to streamed chunk responses. vLLM sends
    one only when a request asks for it, which the engine's never do, but the
    client adds it up if one arrives, and this pins how."""

    def __init__(self, *turns, failures=(), stream_usage=False):
        self.turns = {t.question: t for t in turns}
        self.failures = list(failures)
        self.stream_usage = stream_usage

    def turn(self, question):
        return self.turns.get(question)


# ---------------------------------------------------------------------------
# Distributions the fake reports
# ---------------------------------------------------------------------------

BOUNDARY_ALTS = (" So", " Then", " The", " We", "\n", " Next", " This", " It", " First", " Now")
MID_ALTS = (" the", " a", ",", " and", " to", " of", " is", " in")
# (logprob of the token written, logprobs of its alternatives). Written as
# plain decimals, never computed, and every row's probabilities sum to <= 1,
# as a real server's would.
BOUNDARY_ROWS = tuple(
    (lp0, rest) for lp0 in (-0.6, -0.7, -0.8, -0.9)
    for rest in ((-1.4, -1.9), (-1.6, -2.1, -2.8), (-1.7, -2.2, -2.9, -3.5))
)
MID_ROWS = (
    (-0.02, ()), (-0.02, (-4.2,)),
    (-0.05, (-3.3,)), (-0.05, (-3.5, -4.4)),
    (-0.1, (-2.6,)), (-0.1, (-2.8, -3.6)),
    (-0.2, (-2.0,)), (-0.2, (-2.3, -2.9)),
)


def token_top(vis, tok, k):
    """[(token, logprob)] at one answer position, best first. The token
    written is always the best (temperature 0). Where a sentence starts the
    model is unsure (entropy well above the floor); mid-sentence it mostly
    is not."""
    if tok.flat:
        return [(tok.text, 0.0)]
    h = _h("top", vis, tok.text)
    if at_boundary(vis):
        (lp0, lps), pool = BOUNDARY_ROWS[h[0] % len(BOUNDARY_ROWS)], BOUNDARY_ALTS
    else:
        (lp0, lps), pool = MID_ROWS[h[0] % len(MID_ROWS)], MID_ALTS
    others, i = [], h[2]
    for _ in lps:
        while pool[i % len(pool)] == tok.text or pool[i % len(pool)] in others:
            i += 1
        others.append(pool[i % len(pool)])
        i += 1
    return ([(tok.text, lp0)] + list(zip(others, lps)))[:k]


def level_logprobs(level, n, salt):
    """Logprobs over n labels for an answer that is mostly `level`. Now and
    then the farthest label is left out of the response, which exercises the
    client's rule that a missing id counts as 0."""
    h = _h("level", salt)
    lps = [-4.8] * 5
    lps[level] = -0.15
    up = level < 4 and (level == 0 or h[0] & 1)
    lps[level + 1 if up else level - 1] = -2.2
    if h[1] % 3 == 0:
        lps[0 if level >= 2 else 4] = None
    return (lps + [-4.8] * n)[:n]


YESNO = ((-0.24, -1.6), (-0.7, -0.69), (-1.6, -0.24), (-0.36, -1.2))

SINGLE_TOKENS = {**{chr(65 + i): 32 + i for i in range(26)}, "Yes": 9454, "No": 2753}


# ---------------------------------------------------------------------------
# The fake server
# ---------------------------------------------------------------------------


def _sse(chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(),
                          headers={"content-type": "text/event-stream"})


def _choice(index, text, logprobs, finish):
    return {"index": index, "text": text, "logprobs": logprobs,
            "finish_reason": finish, "stop_reason": None}


def _stream_chunk(model, choice):
    return {"id": "cmpl-golden", "object": "text_completion", "created": 0,
            "model": model, "choices": [choice]}


def _split_even(text, m):
    """`text` as at most m roughly equal tokens, none of them empty."""
    step = max(1, -(-len(text) // m))
    return [text[i:i + step] for i in range(0, len(text), step)] or [text]


def _chunk_pieces(ch, max_tokens):
    """A chunk as streamed pieces: [(text, [token logprobs], [top dicts], finish)].
    The last piece is the chunk's last token, carrying the alternatives."""
    last = ch.alts[0][0] if ch.alts else None
    if last and ch.text.endswith(last) and len(ch.text) > len(last):
        head, tail = ch.text[: -len(last)], [last]
    else:
        head, tail = ch.text, []
    parts = []
    if head:
        cut = head.find(" ", len(head) // 2)
        parts = [head[:cut], head[cut:]] if 0 < cut < len(head) else [head]
    parts += tail
    if tail:
        lps = [(ch.lp - ch.alts[0][1]) / max(1, len(parts) - 1)] * (len(parts) - 1)
        lps.append(ch.alts[0][1])
        if len(parts) == 1:
            lps = [ch.lp]
    else:
        lps = [ch.lp / max(1, len(parts))] * len(parts)
    pieces, used = [], 0
    for i, (text, lp) in enumerate(zip(parts, lps)):
        is_last = i == len(parts) - 1
        toks = [text] if (is_last and tail) else _split_even(text, max(1, len(text) // 4))
        used += len(toks)
        tlps = [lp / len(toks)] * len(toks)
        if ch.no_top:
            tops = [{} for _ in toks]
        elif is_last and ch.alts:
            tops = [{t: v} for t, v in zip(toks[:-1], tlps[:-1])] + [dict(ch.alts)]
        else:
            tops = [{t: v} for t, v in zip(toks, tlps)]
        pieces.append((text, toks, tlps, tops, ch.finish if is_last else None))
    if used > max_tokens:
        raise OffScript(f"scripted chunk is {used} tokens, over max_tokens={max_tokens}")
    return pieces


class FakeServer:
    """The MockTransport handler: routes, records, and answers from a Model."""

    MAX_REQUESTS = 3000

    def __init__(self, model):
        self.model = model
        self.log = []
        self._seen = {}

    # -- dispatch ---------------------------------------------------------

    def __call__(self, request):
        if request.url.host != "vllm.test":
            raise OffScript(f"request to an unexpected host: {request.url}")
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        rec = {
            "method": request.method,
            "path": path,
            "api": "v1" if path.startswith("/v1/") else "root",
            "headers": {name: request.headers.get(name)
                        for name in ("Authorization", "User-Agent", "Content-Type")},
            "body": body,
        }
        self.log.append(rec)
        if len(self.log) > self.MAX_REQUESTS:
            raise OffScript(f"more than {self.MAX_REQUESTS} requests; the engine is looping")
        key = json.dumps(rec, sort_keys=True)
        nth = self._seen.get(key, 0)
        self._seen[key] = nth + 1
        kind = self._kind(request.method, path, body)
        for f in self.model.failures:
            if f.applies(kind, body, nth):
                return httpx.Response(f.status, text=f"fake endpoint: HTTP {f.status}")
        return getattr(self, "_" + kind)(body, nth)

    @staticmethod
    def _kind(method, path, body):
        if (method, path) == ("GET", "/metrics"):
            return "metrics"
        if (method, path) == ("POST", "/tokenize"):
            return "tokenize"
        if (method, path) == ("POST", "/v1/completions"):
            if body.get("stream"):
                if isinstance(body["prompt"], str):
                    return "stream"
                return "fanout" if body.get("max_tokens") == 1 else "chunk"
            if "logprob_token_ids" in body:
                return "probs"
        raise OffScript(f"no route for {method} {path}")

    def _script(self, question):
        return self.model.turn(question)

    # -- the answer ---------------------------------------------------------

    def _stream(self, body, nth):
        prompt, n, k = body["prompt"], body["max_tokens"], body["logprobs"]
        view = parse_prompt(prompt)
        turn = self._script(view.question)
        got = turn.follow(view.vis, view.notes) if turn else None
        if got is None:
            toks, finish = [Tok(".", flat=True)], "stop"   # off script: end the answer
        else:
            p, j = got
            toks = p.tokens[j: j + n]
            finish = "stop" if j + len(toks) >= len(p.tokens) else "length"
        chunks, vis = [], view.vis
        for i, t in enumerate(toks):
            top = token_top(vis, t, k)
            lp = dict(top)[t.text]
            chunks.append(_stream_chunk(body["model"], _choice(
                0, t.text,
                {"text_offset": [len(vis)], "tokens": [t.text], "token_logprobs": [lp],
                 "top_logprobs": [dict(top)]},
                finish if i == len(toks) - 1 else None)))
            vis += t.text
        if not toks:
            chunks.append(_stream_chunk(body["model"], _choice(0, "", None, finish)))
        return _sse(chunks)

    # -- the tree -----------------------------------------------------------

    @staticmethod
    def _salt(view):
        return view.question + "\x1f" + ("open" if view.head is None else view.vis)

    def _block(self, view):
        turn = self._script(view.question)
        return turn.block(view) if turn else GENERIC_BLOCK

    def _fanout(self, body, nth):
        (prompt,) = body["prompt"]
        view = parse_prompt(prompt)
        if view.thinking != "":
            raise OffScript(f"fan-out read for a thought already begun: {view.thinking!r}")
        block = self._block(view)
        entries = block.first_read if (block.first_read and nth == 0) else block.openings
        top = sorted(entries, key=lambda kv: kv[1], reverse=True)[: body["logprobs"]]
        best, lp = top[0]
        return _sse([_stream_chunk(body["model"], _choice(
            0, best, {"tokens": [best], "token_logprobs": [lp], "top_logprobs": [dict(top)]},
            "length"))])

    def _chunk(self, body, nth):
        per = []
        for p in body["prompt"]:
            view = parse_prompt(p)
            if view.thinking is None:
                raise OffScript(f"chunk request for a prompt with no thought: {p[-80:]!r}")
            ch = self._block(view).next_chunk(view.thinking, self._salt(view))
            per.append(_chunk_pieces(ch, body["max_tokens"]))
        chunks, completion = [], 0
        for r in range(max((len(x) for x in per), default=0)):
            for i, pieces in enumerate(per):
                if r < len(pieces):
                    text, toks, tlps, tops, finish = pieces[r]
                    completion += len(toks)
                    chunks.append(_stream_chunk(body["model"], _choice(
                        i, text, {"tokens": toks, "token_logprobs": tlps, "top_logprobs": tops},
                        finish)))
        if self.model.stream_usage:
            prompt_tokens = sum(len(p) // 4 for p in body["prompt"])
            chunks.append({"id": "cmpl-golden", "object": "text_completion", "created": 0,
                           "model": body["model"], "choices": [],
                           "usage": {"prompt_tokens": prompt_tokens,
                                     "completion_tokens": completion,
                                     "total_tokens": prompt_tokens + completion}})
        return _sse(chunks)

    # -- closed questions ---------------------------------------------------

    def _probs(self, body, nth):
        ids = body["logprob_token_ids"]
        choices = []
        for i, p in enumerate(body["prompt"]):
            ctx, q = split_question(p)
            what = classify(q) if q is not None else ("unknown",)
            if what[0] == "gate":
                view = parse_prompt(ctx)
                turn = self._script(view.question)
                levels = (turn.gate_levels(view.vis, view.question) if turn
                          else {a: 0 for a in GATE_QUESTIONS})
                lps = level_logprobs(levels[what[1]], len(ids), p)
            elif what[0] == "judge":
                view = parse_prompt(ctx)
                lv = self._block(view).levels(what[2].strip(), self._salt(view))
                lps = level_logprobs(lv[list(JUDGE_QUESTIONS).index(what[1])], len(ids), p)
            elif what[0] == "yesno":
                lps = list(YESNO[_h("yesno", ctx)[0] % len(YESNO)])
                lps = (lps + [-4.8] * len(ids))[: len(ids)]
            else:
                lps = [-1.7] * len(ids)
            tl = {f"token_id:{tid}": v for tid, v in zip(ids, lps) if v is not None}
            best = max(tl, key=tl.get)
            choices.append({"index": i, "text": "",
                            "logprobs": {"tokens": [best], "token_logprobs": [tl[best]],
                                         "top_logprobs": [tl]},
                            "finish_reason": "length", "stop_reason": None})
        prompt_tokens = sum(len(p) // 4 for p in body["prompt"])
        return httpx.Response(200, json={
            "id": "cmpl-golden", "object": "text_completion", "created": 0,
            "model": body["model"], "choices": choices,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(choices),
                      "total_tokens": prompt_tokens + len(choices)},
        })

    # -- bookkeeping endpoints ----------------------------------------------

    def _tokenize(self, body, nth):
        text = body["prompt"]
        ids = [SINGLE_TOKENS[text]] if text in SINGLE_TOKENS else list(range(len(text) // 4))
        return httpx.Response(200, json={"count": len(ids), "max_model_len": 32768,
                                         "tokens": ids})

    def _metrics(self, body, nth):
        return httpx.Response(200, text=(
            "# HELP vllm:prefix_cache_queries_total Prefix cache queries.\n"
            "# TYPE vllm:prefix_cache_queries_total counter\n"
            'vllm:prefix_cache_queries_total{model_name="test-model"} 12800.0\n'
            "# HELP vllm:prefix_cache_hits_total Prefix cache hits.\n"
            "# TYPE vllm:prefix_cache_hits_total counter\n"
            'vllm:prefix_cache_hits_total{model_name="test-model"} 9600.0\n'
        ))


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

_HTTPX_HELP = re.compile(
    r"(?:\n|\\n)For more information check: https?://[^\s'\"\\]*")


def normalise(obj):
    """Drop httpx's help link from every string (see the module docstring)."""
    if isinstance(obj, str):
        return _HTTPX_HELP.sub("", obj)
    if isinstance(obj, list):
        return [normalise(x) for x in obj]
    if isinstance(obj, dict):
        return {k: normalise(v) for k, v in obj.items()}
    return obj


def _event_line(ev):
    return json.dumps(ev, ensure_ascii=False)


def _conversation(conv):
    return {
        "render": conv.render(),
        "notes": [[ti, n.off, n.text, n.born]
                  for ti, t in enumerate(conv.turns) for n in t.notes],
        "clock": conv.clock,
    }


def record_engine(model, questions, values):
    """Run the questions as one conversation. Returns the recording."""
    server = FakeServer(model)
    turns = []

    async def go():
        client = llm.VLLM()
        engine = Engine(client)
        conv = Conversation()
        try:
            for q in questions:
                start, events, error = len(server.log), [], None
                try:
                    async for ev in engine.respond(conv, q):
                        events.append(_event_line(ev))
                except Exception as exc:  # recorded, so a failure reads as a diff
                    error = f"{type(exc).__name__}: {exc}"
                turns.append({"user": q, "requests": server.log[start:], "events": events,
                              "error": error, **_conversation(conv)})
                if error:
                    break
        finally:
            await client.aclose()
        return client.usage_snapshot()

    with pinned(values), frozen_clock(), no_optional_labels(), mock_transport(server):
        usage = asyncio.run(go())
    return normalise({"knobs": recorded_knobs(values), "turns": turns, "usage": usage})


def record_cli(model, question, values, modes=("json", "formatted")):
    """cli.main() on one question, per output mode: stdout (ANSI codes and
    all) and the requests it made."""
    from adaptive_thinking import cli

    out = {"knobs": recorded_knobs(values)}
    for mode in modes:
        server = FakeServer(model)
        argv = ["cli.py"] + (["--json"] if mode == "json" else []) + [question]
        buf = io.StringIO()
        error = None
        with pinned(values), frozen_clock(), no_optional_labels(), mock_transport(server), \
                mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(buf):
            try:
                asyncio.run(cli.main())
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        out[mode] = {"argv": argv[1:], "requests": server.log, "stdout": buf.getvalue(),
                     "error": error}
    return normalise(out)


def record_server(model, chats, values, tmp_dir):
    """server.py through FastAPI's TestClient: GET /config, then GET /chat
    for each (session, question) in `chats` (the SSE body, and the requests
    that chat made), the events.jsonl it appended, and POST /reset."""
    from adaptive_thinking import server
    from fastapi.testclient import TestClient

    fake = FakeServer(model)
    events = FsPath(tmp_dir) / "events.jsonl"
    out = []
    with pinned(values), frozen_clock(), no_optional_labels(), mock_transport(fake), \
            mock.patch.object(server, "EVENTS", str(events)), \
            mock.patch.object(server, "_llm", None), \
            mock.patch.object(server, "_sessions", {}):
        with TestClient(server.app) as client:
            cfg = client.get("/config")
            for session, question in chats:
                start = len(fake.log)
                chat = client.get("/chat", params={
                    "session": session, "q": question,
                    "inline": int(values["KEEP_BLOCKS_INLINE"])})
                out.append({
                    "session": session, "q": question, "requests": fake.log[start:],
                    "status": chat.status_code,
                    "headers": {h: chat.headers.get(h) for h in
                                ("content-type", "cache-control", "x-accel-buffering")},
                    "body": chat.text,
                })
            reset = client.post("/reset", json={"session": chats[0][0]})
    # /config as parsed JSON plus its key order: the whitespace in the body is
    # FastAPI's serialiser, not server.py, and may change between versions.
    config_pairs = json.loads(cfg.text, object_pairs_hook=lambda pairs: pairs)
    return normalise({
        "knobs": recorded_knobs(values),
        "config": {"status": cfg.status_code, "keys": [k for k, _ in config_pairs],
                   "json": cfg.json()},
        "chats": out,
        "events_jsonl": events.read_text(encoding="utf-8") if events.exists() else None,
        "reset": {"status": reset.status_code, "json": reset.json()},
    })


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
# Prompts repeat the whole conversation, so a recording is mostly shared text.
# Long strings go into a table, each entry written as a prefix of an earlier
# entry + new text + a suffix of an earlier entry. Exact and lossless; the
# test decodes the file and compares the full strings, never the encoding.

INTERN_MIN = 160
_WINDOW = 64


def _lcp(a, b):
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _lcs(a, b):
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[len(a) - mid:] == b[len(b) - mid:]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class _Table:
    def __init__(self):
        self.texts, self.entries, self.index = [], [], {}

    def add(self, s):
        if s in self.index:
            return self.index[s]
        recent = range(max(0, len(self.texts) - _WINDOW), len(self.texts))
        pa, n = max(((i, _lcp(s, self.texts[i])) for i in recent),
                    key=lambda x: x[1], default=(0, 0))
        sb, m = max(((i, _lcs(s, self.texts[i])) for i in recent),
                    key=lambda x: x[1], default=(0, 0))
        m = min(m, len(s) - n)
        entry = [pa, n, s[n: len(s) - m], sb, m] if n + m >= 64 else s
        self.texts.append(s)
        self.entries.append(entry)
        self.index[s] = len(self.texts) - 1
        return self.index[s]


def encode(record):
    table = _Table()

    def walk(x):
        if isinstance(x, str) and len(x) >= INTERN_MIN:
            return {"$t": table.add(x)}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x

    return walk(record), table.entries


def decode(record, entries):
    texts = []
    for e in entries:
        if isinstance(e, str):
            texts.append(e)
        else:
            a, n, mid, b, m = e
            texts.append(texts[a][:n] + mid + (texts[b][len(texts[b]) - m:] if m else ""))

    def walk(x):
        if isinstance(x, dict):
            if set(x) == {"$t"}:
                return texts[x["$t"]]
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(record)


def golden_path(name):
    return GOLDEN_DIR / f"{name}.json"


_ONE_LINE = 600


def _compact(x):
    return json.dumps(x, ensure_ascii=False, separators=(", ", ": "))


def _layout(x, indent=0):
    """JSON with one request, event or table entry per line, so that a
    regenerated golden diffs line by line."""
    pad = " " * (indent + 1)
    if isinstance(x, dict) and x and len(_compact(x)) > _ONE_LINE:
        items = [f"{pad}{json.dumps(k)}: {_layout(v, indent + 1)}" for k, v in x.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    if isinstance(x, list) and x and (len(_compact(x)) > _ONE_LINE or len(x) > 8):
        items = [pad + (_compact(v) if not isinstance(v, (dict, list)) else _layout(v, indent + 1))
                 for v in x]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"
    return _compact(x)


def save(name, about, record):
    rec, entries = encode(record)
    doc = {"scenario": name, "about": about, "format": FORMAT,
           "record": rec, "texts": entries}
    text = _layout({k: v for k, v in doc.items() if k != "texts"})[:-2]
    text += ',\n "texts": [\n' + ",\n".join("  " + _compact(e) for e in entries) + "\n ]\n}\n"
    if json.loads(text) != json.loads(json.dumps(doc)):
        raise AssertionError("golden layout does not round-trip")
    golden_path(name).write_text(text, encoding="utf-8", newline="\n")


def load(name):
    doc = json.loads(golden_path(name).read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise AssertionError(f"{name}: golden format {doc.get('format')}, harness {FORMAT}")
    return decode(doc["record"], doc["texts"])


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------


# Subtrees compared regardless of key order: a request body (as
# json.dumps(sort_keys=True) would compare it) and the parsed JSON of a
# server response (its key order is recorded separately where it matters).
_ORDER_FREE = {"body", "json"}


def flatten(x, path="$", order_free=False):
    """[(path, kind, text)] per leaf, in recording order - so the first
    difference is usually the cause (requests come before the events of the
    same turn). A string is its own text; any other leaf is its JSON text,
    so 0 and 0.0 differ, and so do 1 and true. Every list also contributes
    its length, after its items, so an item that differs is reported before
    a count that differs."""
    if isinstance(x, dict):
        out = []
        for k in (sorted(x) if order_free else x):
            out += flatten(x[k], f"{path}.{k}", order_free or k in _ORDER_FREE)
        return out or [(path, "json", "{}")]
    if isinstance(x, list):
        out = []
        for i, v in enumerate(x):
            out += flatten(v, f"{path}[{i}]", order_free)
        return out + [(f"{path}.length", "json", str(len(x)))]
    if isinstance(x, str):
        return [(path, "str", x)]
    return [(path, "json", json.dumps(x))]


def _around(s, at, width=160):
    lo = max(0, at - width // 2)
    hi = at + width // 2
    return ("..." if lo else "") + s[lo:hi] + ("..." if hi < len(s) else "")


def first_difference(expected, actual):
    """None when the two recordings are identical; otherwise where they first
    differ, with the text around it from both sides."""
    fe, fa = flatten(expected), flatten(actual)
    if fe == fa:
        return None
    i = next(j for j, (e, a) in enumerate(zip(fe + [None], fa + [None])) if e != a)
    if i >= len(fe) or i >= len(fa):
        extra, side = (fa, "fresh recording") if i < len(fa) else (fe, "golden")
        return f"only the {side} has {extra[i][0]}"
    (pe, ke, ve), (pa, ka, va) = fe[i], fa[i]
    if pe != pa:
        return (f"the recordings have a different shape:\n  golden: {pe} = {ve[:200]!r}\n"
                f"  fresh:  {pa} = {va[:200]!r}")
    at = next((j for j, (x, y) in enumerate(zip(ve, va)) if x != y), min(len(ve), len(va)))
    what = "" if ke == ka else f" (golden is {ke}, fresh is {ka})"
    return (f"{pe}{what}, at character {at} of {len(ve)} (golden) / {len(va)} (fresh):\n"
            f"  golden: {_around(ve, at)!r}\n"
            f"  fresh:  {_around(va, at)!r}")
