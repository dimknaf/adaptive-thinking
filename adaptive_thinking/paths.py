"""Where things live on disk, in one place.

ROOT is the repository root, one level above this package. It holds the .env
(config.py finds it the same way, without importing this module) and the
events.jsonl the demo server appends to. STATIC is the browser UI, which ships
inside the package.
"""

import os
from pathlib import Path

# os.path.abspath, not Path.resolve(): each path keeps the spelling the package
# was imported through (the drive letter's case, a symlinked or substituted
# checkout), as the modules at the root computed it before the package move.
# resolve() would canonicalise it, and experiments/gate_audit.py prints
# EVENTS.
PACKAGE = Path(os.path.abspath(__file__)).parent
ROOT = PACKAGE.parent
STATIC = PACKAGE / "static"
# Every gate decision and block from a browser session, as JSONL (see
# events_log.py). At the repository root, where it has always been written.
EVENTS = ROOT / "events.jsonl"
