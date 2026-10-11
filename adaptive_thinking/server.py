"""FastAPI wrapper: one SSE endpoint that forwards the engine's events verbatim.

    python server.py      ->  http://127.0.0.1:$PORT
                              (8123 with .env.example; the code default is 8099)

`python server.py` is the launcher at the repository root; `python -m
adaptive_thinking.server` runs the same thing.

A local, single-user demo: it binds 127.0.0.1, has no authentication, and the
`inline` query parameter changes a process-wide setting for every session.

The server adds no logic of its own. Every event the browser renders is one the
CLI already prints, so the UI cannot show something the engine did not do.
"""

import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse

from . import config, events_log, paths
from .engine import Engine
from .llm import VLLM
from .state import Conversation

app = FastAPI()
_llm = None
_sessions = {}

# Every gate decision and block from a browser session, appended as JSONL, so a
# run can be audited afterwards instead of being watched go past. The question
# is carried verbatim on the `turn` line and the answer on the `done` line;
# `window` lines carry every position the gate scored, fired or not, which is
# what makes it possible to ask whether the gate stops for good reasons. At
# the repository root (paths.EVENTS), and read by log() at call time, so it
# can be pointed elsewhere.
EVENTS = paths.EVENTS
_KEEP = {"turn", "window", "block_open", "block_close", "resumed", "monologue",
         "evict", "done", "error"}


def log(record):
    """Append one event to EVENTS. Takes a DICT, not keywords, and never
    raises: logging must never break a turn (see events_log.append)."""
    events_log.append(EVENTS, record)


def _record(session, ev):
    if ev.get("type") in _KEEP:
        log({**ev, "session": session})


def llm():
    global _llm
    if _llm is None:
        _llm = VLLM()
    return _llm


@app.on_event("shutdown")
async def _shutdown():
    if _llm is not None:
        await _llm.aclose()


@app.get("/")
async def index():
    return FileResponse(paths.STATIC / "index.html")


@app.get("/config")
async def settings():
    return config.summary()


@app.post("/reset")
async def reset(request: Request):
    body = await request.json()
    _sessions.pop(body.get("session", ""), None)
    return {"ok": True}


@app.get("/chat")
async def chat(request: Request, session: str = "default", q: str = "", inline: int = -1):
    # The UI can flip KEEP_BLOCKS_INLINE per request so the seam can be seen
    # with and without, side by side, without restarting anything.
    if inline >= 0:
        config.KEEP_BLOCKS_INLINE = bool(inline)
    conv = _sessions.setdefault(session, Conversation())
    engine = Engine(llm())

    async def gen():
        try:
            log({"type": "turn", "session": session, "q": q,
                 "inline": config.KEEP_BLOCKS_INLINE})
            async for ev in engine.respond(conv, q):
                if await request.is_disconnected():
                    break
                _record(session, ev)
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:  # surface it in the UI instead of a dead stream
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(exc).__name__}: {exc}'})}\n\n"
        yield "data: {\"type\": \"end\"}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main():
    """Serve the demo on 127.0.0.1:PORT until interrupted."""
    uvicorn.run(app, host="127.0.0.1", port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
