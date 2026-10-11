"""The state the engine works on: beams, blocks, notes, turns, and the
conversation that renders them back into a raw prompt.

A Beam is one conversation in a thinking block's tree. A Block is one run of
that tree, and its winning thought becomes a Note: the text left in the
context at the place it was written - in the assistant header for the opening
thought (off HEAD_OFF), inside the answer for a mid-answer block. A Turn holds
the visible text and its notes, and Conversation.render() is the one place
the whole prompt is built from them.
"""

from dataclasses import dataclass, field

from . import chatml, prompts
from .text import head_note, splice

# The opening thought is a note like any other - so eviction, counting and the
# live_notes() bookkeeping need no second code path - but it does not live at a
# character offset in the answer. It lives in the assistant HEADER, before the
# answer starts, and this offset says so.
HEAD_OFF = -1


@dataclass
class Beam:
    """One live conversation in the thinking tree.

    `uid` is the lineage's identity and `from_uid` is the lineage it split off
    from. Slot indices cannot serve this purpose: slots are reused every step,
    so following `parent` across steps tracks a position, not a conversation,
    and a dropped lineage is indistinguishable from one that merely moved.
    """

    text: str
    score: float  # cumulative logprob before penalties
    parent: int  # slot of the previous step this came from
    uid: int = 0
    from_uid: int = -1
    index: int = 0
    tokens: int = 0  # length, for normalising the score
    penalty: float = 0.0  # copy + sibling penalties applied this step
    last: str = ""  # the token that extended it this step
    p: float = None  # label probability from the Choice, when one was run
    done: bool = False  # closed its own think block
    axes: dict = field(default_factory=dict)  # raw + standardised axis scores
    # Two numbers, because they answer different questions. `overall` is the
    # weighted sum of the axes STANDARDISED within a group - good for ranking
    # like against like, meaningless across groups. `raw` is the plain
    # weighted mean of the raw axes, 0..1, absolute and comparable to any
    # other thought in the block - which is what "is this a challenger?" needs.
    overall: float = None
    raw: float = None
    medal: int = 0  # 1 gold, 2 silver, 3 bronze

    @property
    def rank_score(self):
        """Length-normalised, penalised. This is what pruning sorts on.

        Cumulative logprob alone is why the tree collapsed: a short beam, or one
        that copied text already in the context, keeps a near-zero score forever
        and outranks every longer candidate. The measured worst case was a
        winner scoring -0.01 across 28 tokens, which was pure regurgitation of a
        previous block's note.
        """
        n = max(1, self.tokens)
        return (self.score / n) - self.penalty


@dataclass
class Block:
    id: int
    tail: str
    # How critical the gate judged this position, 0..1, and the per-axis
    # breakdown behind it. None for the opening thought - nothing judged it,
    # because thinking before answering is not something to be triggered.
    crit: float
    gate: dict = None
    # "open" = the opening thought at the native position, run unconditionally
    # before the answer starts. "gate" = one the gate triggered mid-answer.
    # They are the same search; only the position and what it means differ.
    kind: str = "gate"
    beams: list = field(default_factory=list)
    monologue: str = ""  # the winning deliberation, the model's own words
    winner_score: float = 0.0
    steps: int = 0
    requests: int = 0
    # Did thinking change anything? Compared against the token the gate step
    # would have emitted, which we already have - a far better question than
    # "did the winning index differ from 0".
    changed_next_token: bool = False
    finished_count: int = 0
    explored_tokens: int = 0
    secs: float = 0.0


@dataclass
class Note:
    """A thinking block as it sits in the context, at the place it happened."""

    off: int  # character offset into the turn's visible text
    text: str
    born: int  # value of Conversation.clock when it was written


@dataclass
class Turn:
    role: str
    text: str
    blocks: list = field(default_factory=list)
    notes: list = field(default_factory=list)


class Conversation:
    """The transcript, and how it is rendered back into a raw prompt.

    A thinking block stays at the offset where it was written, for the life of
    the conversation, until it is evicted. It is NOT relocated into a header at
    the top of the turn - an earlier version did that, which meant the model saw
    one string while generating and a different one afterwards, shifting
    everything downstream under the KV cache for no reason.
    """

    def __init__(self, system=None):
        self.system = system or prompts.DEFAULT_SYSTEM
        self.turns = []
        # Tokens generated across the whole conversation. A note's age is the
        # distance from its `born` to here, which is what eviction runs on.
        self.clock = 0

    def render(self, pending_user=None, think=False):
        """The raw prompt. `think` leaves the new turn's thinking slot open.

        A past turn that still holds its opening thought renders it in the
        header, where the model wrote it. Once eviction takes that note the
        header falls back to the empty NO_THINK block - the prefix changes and
        the next request re-prefills, which is eviction's stated cost.
        """
        s = chatml.turn("system", self.system)
        for t in self.turns:
            if t.role == "user":
                s += chatml.turn("user", t.text)
            else:
                s += (
                    chatml.assistant_header(think=True)
                    + self.render_assistant(t) + chatml.IM_END + "\n"
                )
        if pending_user is not None:
            s += chatml.turn("user", pending_user) + chatml.assistant_header(think)
        return s

    @staticmethod
    def render_assistant(turn):
        """An assistant turn as the model reads it: header thought, then body.

        The thinking slot holds the opening thought while the turn still has
        one, and the empty NO_THINK block once it has none; the body is the
        visible answer with its inline notes spliced in. This is everything
        between the assistant header and the closing <|im_end|>, so it is also
        what has to read the same at the end of a conversation as it did when
        the turn was written.
        """
        head = head_note(turn.notes)
        return ((chatml.think_head(head.text) if head else chatml.NO_THINK)
                + splice(turn.text, turn.notes))

    def live_notes(self):
        return [(t, n) for t in self.turns for n in t.notes]
