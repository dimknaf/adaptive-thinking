"""The adaptive_thinking package itself: where things live, and what it promises.

The engine is one package. These tests pin what the move into it must not have
changed or lost: the .env is still read from the repository root, knobs and
prompt constants are still read from their module at call time, the event
types are listed in one place, importing the package does nothing by itself,
and the launchers at the root still start the same code.
"""

import ast
import importlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import adaptive_thinking
from adaptive_thinking import chatml, config, events_log, paths, prompts
from adaptive_thinking.engine import Engine
from adaptive_thinking.events import EVENT_TYPES

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "adaptive_thinking"
# The modules the engine runs on. cli.py and server.py are front ends; the
# server adds "turn", "error" and "end" records of its own.
ENGINE_MODULES = ("engine", "events", "gate", "scoring", "state", "text", "tree")


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def test_the_env_file_is_at_the_repository_root():
    assert paths.ROOT == ROOT
    assert (paths.ROOT / "pyproject.toml").is_file()
    assert config.ENV_FILE == paths.ROOT / ".env"


def test_events_jsonl_stays_at_the_root_and_the_ui_ships_in_the_package():
    assert paths.EVENTS == ROOT / "events.jsonl"
    assert paths.STATIC == PKG / "static"
    assert (paths.STATIC / "index.html").is_file()


def _same_root_spelled_differently(tmp_path):
    """The repository root under another name for the same directory: a
    symlink where the platform allows one, else (Windows) the drive letter in
    the other case. None if neither is available."""
    link = tmp_path / "checkout"
    try:
        os.symlink(ROOT, link, target_is_directory=True)
        return str(link)
    except (OSError, NotImplementedError):
        pass
    drive, rest = os.path.splitdrive(str(ROOT))
    if drive[:1].isalpha() and drive[1:] == ":":
        other = drive.swapcase() + rest
        if other != str(ROOT) and os.path.isdir(other):
            return other
    return None


@pytest.mark.parametrize("module", ["paths", "config"])
def test_paths_keep_the_spelling_the_package_was_imported_through(tmp_path, module):
    # Before the package move, config.py, server.py and gate_audit.py each
    # took their directory from os.path.abspath(__file__), and gate_audit.py
    # (now experiments/gate_audit.py) prints the events.jsonl path it reads.
    # Path.resolve() would print the canonical path instead: another
    # drive-letter case, or the target of a symlinked, substituted or mapped
    # checkout.
    alias = _same_root_spelled_differently(tmp_path)
    if alias is None:
        pytest.skip("no second spelling of the repository root on this machine")
    source = os.path.join(alias, "adaptive_thinking", f"{module}.py")
    spec = importlib.util.spec_from_file_location(f"_alias_{module}", source)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    root = os.path.dirname(os.path.dirname(os.path.abspath(source)))
    if module == "paths":
        assert str(loaded.ROOT) == root
        assert str(loaded.EVENTS) == os.path.join(root, "events.jsonl")
        assert str(loaded.STATIC / "index.html") == os.path.join(
            root, "adaptive_thinking", "static", "index.html")
    else:
        assert str(loaded.ENV_FILE) == os.path.join(root, ".env")


# ---------------------------------------------------------------------------
# knobs and constants are read at call time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(PKG.glob("*.py")), ids=lambda p: p.name)
def test_no_module_binds_a_knob_or_a_prompt_constant_at_import(path):
    # `from .config import X` would copy X at import, and a knob changed at
    # runtime (server.py's `inline`, a test's cfg) would not be seen.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    copied = [
        f"from {'.' * node.level}{node.module} import ..."
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").split(".")[-1] in ("config", "chatml", "prompts")
    ]
    assert copied == []


def test_config_imports_nothing_from_the_package_at_import():
    # So that tests/test_config.py can execute it as a standalone module.
    top = ast.parse((PKG / "config.py").read_text(encoding="utf-8")).body
    relative = [n for n in top if isinstance(n, ast.ImportFrom) and n.level]
    package = [n for n in top if isinstance(n, (ast.Import, ast.ImportFrom))
               and "adaptive_thinking" in ast.dump(n)]
    assert relative == [] and package == []


def test_config_chatml_and_prompts_share_no_public_name():
    # tests/conftest.py's cfg fixture patches each name on the one module
    # that has it.
    def public(module):
        return {n for n in vars(module) if not n.startswith("__")
                and not isinstance(vars(module)[n], type(sys))}

    assert public(config) & public(chatml) == set()
    assert public(config) & public(prompts) == set()
    assert public(chatml) & public(prompts) == set()


def test_cfg_patches_prompt_constants_where_the_engine_reads_them(cfg):
    cfg(SPILL=chatml.SPILL + ("@@",), NUDGE_PREFIX="Back to it.")
    assert chatml.SPILL[-1] == "@@"
    assert prompts.NUDGE_PREFIX == "Back to it."
    assert not hasattr(config, "SPILL") and not hasattr(config, "NUDGE_PREFIX")


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


def test_event_types_list_every_type_the_engine_yields():
    written = set()
    for name in ENGINE_MODULES:
        src = (PKG / f"{name}.py").read_text(encoding="utf-8")
        written |= set(re.findall(r'"type":\s*"(\w+)"', src))
    assert written == set(EVENT_TYPES)
    assert len(EVENT_TYPES) == len(set(EVENT_TYPES))


def test_events_log_round_trip(tmp_path, monkeypatch):
    import time

    monkeypatch.setattr(time, "time", lambda: 12.3456)
    path = tmp_path / "events.jsonl"
    assert list(events_log.read(path)) == []
    events_log.append(path, {"type": "turn", "q": "café", "kind": "gate"})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    events_log.append(path, {"type": "done"})
    assert path.read_text(encoding="utf-8").splitlines()[0] == (
        '{"t": 12.35, "type": "turn", "q": "café", "kind": "gate"}')
    assert list(events_log.read(path)) == [
        {"t": 12.35, "type": "turn", "q": "café", "kind": "gate"},
        {"t": 12.35, "type": "done"},
    ]


def test_events_log_append_never_raises(tmp_path):
    events_log.append(tmp_path / "missing-dir" / "events.jsonl", {"type": "turn"})
    events_log.append(tmp_path / "x.jsonl", {"type": "turn", "bad": object()})
    assert not (tmp_path / "missing-dir").exists()


# ---------------------------------------------------------------------------
# the package's own namespace, and the launchers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", adaptive_thinking.__all__)
def test_every_export_is_the_object_its_module_defines(name):
    home = adaptive_thinking._EXPORTS[name]
    module = importlib.import_module(f"adaptive_thinking.{home}")
    assert getattr(adaptive_thinking, name) is getattr(module, name)


def test_an_unknown_package_attribute_is_an_attribute_error():
    with pytest.raises(AttributeError, match="no_such_name"):
        getattr(adaptive_thinking, "no_such_name")


def test_importing_the_package_imports_nothing_else():
    # A fresh interpreter: `import adaptive_thinking` must not load a
    # submodule (so no .env is read) until one of its names is used.
    code = ("import sys, adaptive_thinking; "
            "print(sorted(m for m in sys.modules if m.startswith('adaptive_thinking')))")
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "['adaptive_thinking']"


def test_the_root_launchers_start_the_package_entry_points():
    from adaptive_thinking import cli, server

    assert importlib.import_module("cli").main is cli.main
    launcher = importlib.import_module("server")
    assert launcher.main is server.main
    assert launcher.app is server.app


def test_the_engine_still_offers_every_step_as_a_method():
    for name in ("respond", "maybe_evict", "_choose", "_gate_scores", "_run_block",
                 "_fan_out", "_prune", "_churn", "_score_axes", "_rank",
                 "_finalise_block", "_undo_restart", "_commit_turn"):
        assert callable(getattr(Engine, name)), name
    assert Engine.MAX_LOGPROBS == 20
