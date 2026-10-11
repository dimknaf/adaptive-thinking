"""events.jsonl: the audit trail the demo server appends and
experiments/gate_audit.py reads.

One JSON object per line, each stamped with the time it was written as `t`.
Written with ensure_ascii=False, so the text of a question or an answer is
kept as it was typed. The default location is paths.EVENTS, at the repository
root; both functions take the path explicitly, so a caller (or a test) can
point them anywhere.
"""

import json
import os
import time


def append(path, record):
    """Append one record, stamped with `t`. Takes a DICT, not keywords.

    It used to take **fields, which meant every field name on an event was
    also a parameter name - and the moment blocks grew a `kind` field it
    collided with the `kind` parameter and took the turn down with a
    TypeError. A dict cannot collide with anything.

    Never raises: logging must never break a turn.
    """
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": round(time.time(), 2), **record},
                                ensure_ascii=False) + "\n")
    except Exception:
        pass


def read(path):
    """Every record in the file, oldest first.

    A missing file reads as empty, and a line that does not parse - one cut
    short by a crash mid-write, say - is skipped rather than ending the read.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except Exception:
                continue
            yield record
