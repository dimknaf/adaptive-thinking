"""Hermetic setup for the offline suite.

The module-level code below runs before any test module is imported, and so
before anything imports config. config freezes every knob at import time and
calls load_dotenv() on the repo's .env, which would export a developer's
settings (and API key) into os.environ and make the same test pass on one
machine and fail on another. So, in this order:

  1. dotenv.load_dotenv becomes a no-op. config does `from dotenv import
     load_dotenv`, so the stub is what it binds.
  2. Every environment variable config reads is removed, so the code defaults
     apply. The names are taken from adaptive_thinking/config.py's source,
     not from a list that could drift.
  3. The endpoint variables point at a host that does not exist.
  4. The label package llm.labels() used to try before /tokenize (jevfire)
     is blocked. labels() no longer imports it; the block stays so that a
     copy installed on the machine can never reach the session.
  5. config is imported (or reloaded) with all of that in place.

Per test, the autouse `network_guard` fixture makes real HTTP impossible, and
fails the test even when production code swallowed the error.
"""

import importlib
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PY = ROOT / "adaptive_thinking" / "config.py"
# pyproject.toml's pytest settings already put both on sys.path; this keeps
# the file usable under `pytest -c <other ini>` too.
for _p in (ROOT / "tests", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# -- 1. never read a .env ------------------------------------------------------
import dotenv  # noqa: E402
import dotenv.main  # noqa: E402


def _no_dotenv(*args, **kwargs):
    """Stands in for dotenv.load_dotenv for the whole test session."""
    return False


dotenv.load_dotenv = _no_dotenv
dotenv.main.load_dotenv = _no_dotenv

# -- 2. code defaults, not whatever the shell exported --------------------------
_ENV_READ = re.compile(r'(?:\b_[fib]|os\.getenv)\(\s*"([A-Z][A-Z0-9_]*)"')
CONFIG_ENV_VARS = sorted(
    set(_ENV_READ.findall(CONFIG_PY.read_text(encoding="utf-8")))
)
for _name in CONFIG_ENV_VARS:
    os.environ.pop(_name, None)

# -- 3. an endpoint that does not exist -----------------------------------------
TEST_ENDPOINT = {
    "OPENAI_BASE_URL": "http://vllm.test/v1",
    "OPENAI_MODEL": "test-model",
    "OPENAI_API_KEY": "test-key",
}
os.environ.update(TEST_ENDPOINT)

# -- 4. block the label package llm.labels() used to import -----------------------
sys.modules["jevfire"] = None

# -- 5. freeze config with the above in place ------------------------------------
_already = sys.modules.get("adaptive_thinking.config")
if _already is not None:
    _file = getattr(_already, "__file__", None)
    if _file and os.path.samefile(_file, CONFIG_PY):
        importlib.reload(_already)
    else:
        del sys.modules["adaptive_thinking.config"]

import httpx  # noqa: E402

from adaptive_thinking import config  # noqa: E402

assert os.path.samefile(config.__file__, CONFIG_PY), config.__file__
assert config.load_dotenv is _no_dotenv, "config bound the real load_dotenv"
assert config.API_KEY == TEST_ENDPOINT["OPENAI_API_KEY"]

from adaptive_thinking import chatml, prompts  # noqa: E402
from fake_endpoint import FakeVLLM  # noqa: E402

# Where the `cfg` fixture patches a name: the module the engine reads it from
# at call time. The knobs are on config; besides them, tests set a few
# constants of the chat format (SPILL, on chatml) and of the prompt wording
# (AXES, GATE_AXES, NUDGE_PREFIX, DEFAULT_SYSTEM, on prompts). The three
# modules share no name, so each name has exactly one home.
KNOB_HOMES = (config, chatml, prompts)


# -- fixtures -----------------------------------------------------------------


class NetworkGuard:
    """Refuses every real HTTP request and remembers each attempt.

    Remembering matters because some production code catches every exception
    (llm.VLLM.prefix_cache_stats returns None on any error, Engine._size falls
    back to a character count). The AssertionError alone would vanish there;
    the recorded attempt still fails the test at teardown.
    """

    MESSAGE = "real network call in a unit test"

    def __init__(self):
        self.attempts = []

    def refuse(self, request):
        self.attempts.append(f"{request.method} {request.url}")
        raise AssertionError(f"{self.MESSAGE}: {request.method} {request.url}")

    def check(self):
        if self.attempts:
            pytest.fail(f"{self.MESSAGE}: {self.attempts}", pytrace=False)


@pytest.fixture(autouse=True)
def network_guard(monkeypatch):
    """Patch the two real httpx transports; MockTransport and TestClient's
    transport are separate classes and keep working. Also drops the retry
    backoff to zero so retry paths do not sleep."""
    guard = NetworkGuard()

    def refuse_sync(self, request):
        guard.refuse(request)

    async def refuse_async(self, request):
        guard.refuse(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse_sync)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse_async)
    monkeypatch.setattr(config, "LLM_RETRY_BACKOFF", 0.0)
    yield guard
    guard.check()


@pytest.fixture
def cfg(monkeypatch):
    """Set several config knobs for one test: cfg(BEAM_WIDTH=2, OPEN_THINK=False).

    Every change goes through monkeypatch, so it is undone after the test.
    Each name is set on its home in KNOB_HOMES - a knob on config, SPILL on
    chatml, AXES on prompts - which is where the engine reads it. A name
    none of them has is an error, so a typo cannot silently set nothing.
    Returns the config module.
    """

    def set_knobs(**knobs):
        for name, value in knobs.items():
            home = next((m for m in KNOB_HOMES if hasattr(m, name)), None)
            if home is None:
                raise AttributeError(f"config has no knob named {name!r}")
            monkeypatch.setattr(home, name, value)
        return config

    return set_knobs


@pytest.fixture
def fake_llm():
    """Factory for scripted engine-level fakes: fake_llm(streams=[...], ...).

    See fake_endpoint.FakeVLLM for the script format.
    """

    def make(**script):
        return FakeVLLM(**script)

    return make
