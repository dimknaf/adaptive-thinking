"""Launcher: run the engine in a terminal.

    python cli.py "your question"
    python cli.py --json "your question"     raw events, one per line

The client is adaptive_thinking/cli.py; this file only starts it, so that the
command above keeps working from the repository root.
"""

import asyncio

from adaptive_thinking.cli import main

if __name__ == "__main__":
    asyncio.run(main())
