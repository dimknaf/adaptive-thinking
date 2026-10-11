"""Adaptive thinking: a model that decides mid-answer where to stop and think.

The engine runs a small tree of continuations wherever a scored gate fires,
keeps the winning thought in the context, and resumes the answer. It drives an
OpenAI-compatible vLLM endpoint with raw /v1/completions requests.

    from adaptive_thinking import VLLM, Engine, Conversation

    engine = Engine(VLLM())
    async for event in engine.respond(Conversation(), "your question"):
        ...

Every knob is an environment variable read by adaptive_thinking.config, which
loads the .env at the repository root when it is first imported.

The names below are resolved on first use rather than imported here, so that
importing the package has no side effects of its own - and so that running a
submodule with `python -m` (the endpoint smoke test is `python -m
adaptive_thinking.llm`) does not import that module twice.
"""

import importlib

# name -> the submodule that defines it
_EXPORTS = {
    "Engine": "engine",
    "EVENT_TYPES": "events",
    "Conversation": "state",
    "Beam": "state",
    "Block": "state",
    "Note": "state",
    "Turn": "state",
    "HEAD_OFF": "state",
    "cut_at_marker": "text",
    "collapse_ws": "text",
    "head_note": "text",
    "splice": "text",
    "VLLM": "llm",
    "Step": "llm",
    "UsageMeter": "llm",
    "transient": "llm",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
