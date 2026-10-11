"""Launcher: the local demo server.

    python server.py      ->  http://127.0.0.1:$PORT

The server is adaptive_thinking/server.py; this file only starts it, so that
the command above keeps working from the repository root. `app` is re-exported
so that `uvicorn server:app` keeps working too.
"""

from adaptive_thinking.server import app, main  # noqa: F401

if __name__ == "__main__":
    main()
