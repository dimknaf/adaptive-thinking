"""config.py: env parsing, URL derivation, prompt shapes, and invariants.

The shared `config` module is never reloaded here. Where a test needs config
evaluated against a different environment, config.py is executed into a fresh,
private module (see `fresh_config`), so the knobs every other test relies on
are untouched. tests/conftest.py has already replaced dotenv.load_dotenv with
a no-op, so a fresh copy never reads a .env file either.
"""

import ast
import importlib.util
import re
from pathlib import Path

import pytest
from dotenv import dotenv_values

from adaptive_thinking import chatml, config, prompts

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PY = ROOT / "adaptive_thinking" / "config.py"
ENV_EXAMPLE = ROOT / ".env.example"

# An environment variable name config.py does not read, for the helpers.
PROBE_VAR = "ADAPTIVE_THINKING_TEST_KNOB"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def env_knobs():
    """{ENV_NAME: reader} for every environment variable config.py reads,
    found by walking its AST: _f/_i/_b("NAME", ...) and os.getenv("NAME", ...).
    reader is "_f", "_i", "_b" or "getenv"."""
    out = {}
    for node in ast.walk(ast.parse(CONFIG_PY.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id in ("_f", "_i", "_b"):
            out[first.value] = f.id
        elif (isinstance(f, ast.Attribute) and f.attr == "getenv"
              and isinstance(f.value, ast.Name) and f.value.id == "os"):
            out[first.value] = "getenv"
    return out


def example_names():
    """Every variable .env.example mentions as an assignment, commented out or
    not: `NAME=value` or `# NAME=value`."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"(?m)^#?[ \t]*([A-Z][A-Z0-9_]*)=", text))


@pytest.fixture
def fresh_config(monkeypatch):
    """fresh_config(**env) executes config.py into a new private module with
    exactly the given environment variables set (every other variable config
    reads is removed), and returns that module. The shared config module is
    not touched."""
    count = [0]

    def load(**env):
        for name in env_knobs():
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        count[0] += 1
        spec = importlib.util.spec_from_file_location(
            f"_config_under_test_{count[0]}", CONFIG_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # The copy bound the session's no-op, not python-dotenv's loader.
        assert module.load_dotenv is config.load_dotenv
        return module

    return load


# ---------------------------------------------------------------------------
# _f / _i / _b read the environment at call time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "True", "yes", " YES ", "\tyes\n"])
def test_b_is_true_only_for_1_true_yes(monkeypatch, raw):
    monkeypatch.setenv(PROBE_VAR, raw)
    assert config._b(PROBE_VAR, False) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "on", "y", "t", "2", "enabled"])
def test_b_is_false_for_anything_else(monkeypatch, raw):
    # "on" is false: only 1, true and yes switch a flag on.
    monkeypatch.setenv(PROBE_VAR, raw)
    assert config._b(PROBE_VAR, True) is False


@pytest.mark.parametrize("default", [True, False])
def test_b_unset_or_empty_gives_the_default(monkeypatch, default):
    monkeypatch.delenv(PROBE_VAR, raising=False)
    assert config._b(PROBE_VAR, default) is default
    monkeypatch.setenv(PROBE_VAR, "")
    assert config._b(PROBE_VAR, default) is default


def test_i_parses_integers(monkeypatch):
    monkeypatch.setenv(PROBE_VAR, "7")
    assert config._i(PROBE_VAR, 3) == 7
    monkeypatch.setenv(PROBE_VAR, " 12 ")
    assert config._i(PROBE_VAR, 3) == 12
    monkeypatch.setenv(PROBE_VAR, "-4")
    assert config._i(PROBE_VAR, 3) == -4


def test_i_zero_is_a_value_not_empty(monkeypatch):
    monkeypatch.setenv(PROBE_VAR, "0")
    assert config._i(PROBE_VAR, 3) == 0


def test_i_unset_or_empty_gives_the_default(monkeypatch):
    monkeypatch.delenv(PROBE_VAR, raising=False)
    assert config._i(PROBE_VAR, 3) == 3
    monkeypatch.setenv(PROBE_VAR, "")
    assert config._i(PROBE_VAR, 3) == 3


@pytest.mark.parametrize("raw", ["7.0", "1e3", "seven", "0x10"])
def test_i_rejects_anything_but_a_plain_integer(monkeypatch, raw):
    monkeypatch.setenv(PROBE_VAR, raw)
    with pytest.raises(ValueError):
        config._i(PROBE_VAR, 3)


@pytest.mark.parametrize("raw, expected", [
    ("0.8", 0.8), ("1e-3", 0.001), ("3", 3.0), (" 2.5 ", 2.5), ("0", 0.0), ("-1.5", -1.5),
])
def test_f_parses_floats(monkeypatch, raw, expected):
    monkeypatch.setenv(PROBE_VAR, raw)
    assert config._f(PROBE_VAR, 9.0) == expected


def test_f_unset_or_empty_gives_the_default(monkeypatch):
    monkeypatch.delenv(PROBE_VAR, raising=False)
    assert config._f(PROBE_VAR, 0.55) == 0.55
    monkeypatch.setenv(PROBE_VAR, "")
    assert config._f(PROBE_VAR, 0.55) == 0.55


def test_f_rejects_text(monkeypatch):
    monkeypatch.setenv(PROBE_VAR, "high")
    with pytest.raises(ValueError):
        config._f(PROBE_VAR, 0.55)


def test_readers_return_the_right_types(monkeypatch):
    monkeypatch.delenv(PROBE_VAR, raising=False)
    # Exact types on purpose: isinstance(True, int) holds, so isinstance could
    # not tell a bool from an int.
    assert type(config._f(PROBE_VAR, 3)) is float  # noqa: E721
    assert type(config._i(PROBE_VAR, 3)) is int  # noqa: E721
    assert type(config._b(PROBE_VAR, 1)) is bool  # noqa: E721


# ---------------------------------------------------------------------------
# config evaluated against a given environment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url, base, root", [
    ("https://h/v1", "https://h/v1", "https://h"),
    ("https://h/v1/", "https://h/v1", "https://h"),
    ("https://h/api/v1", "https://h/api/v1", "https://h/api"),
    # Only a trailing /v1 is stripped.
    ("https://h/v1beta", "https://h/v1beta", "https://h/v1beta"),
    ("https://h", "https://h", "https://h"),
    ("", "", ""),
])
def test_root_url_is_the_base_url_without_its_v1(fresh_config, url, base, root):
    c = fresh_config(OPENAI_BASE_URL=url)
    assert (c.BASE_URL, c.ROOT_URL) == (base, root)


def test_an_empty_environment_gives_the_code_defaults(fresh_config):
    c = fresh_config()
    assert (c.BASE_URL, c.ROOT_URL, c.MODEL, c.API_KEY) == ("", "", "", "")
    assert c.USER_AGENT == "adaptive-thinking/0.1"
    assert c.GATE_THRESHOLD == 0.55
    assert c.PORT == 8099
    assert c.INLINE_SEED == "<think>Hmm, hold on."
    assert c.OPEN_SEED == "<think>\n"


def test_empty_string_env_values_fall_back_to_the_defaults(fresh_config):
    c = fresh_config(INLINE_SEED="", OPEN_SEED="", BEAM_WIDTH="", OPEN_THINK="",
                     LLM_USER_AGENT="")
    assert c.INLINE_SEED == "<think>Hmm, hold on."
    assert c.OPEN_SEED == "<think>\n"
    assert c.BEAM_WIDTH == 4
    assert c.OPEN_THINK is True
    assert c.USER_AGENT == "adaptive-thinking/0.1"


def test_environment_values_override_the_defaults(fresh_config):
    c = fresh_config(GATE_THRESHOLD="0.8", BEAM_WIDTH="2", OPEN_THINK="false",
                     INLINE_SEED="<think>Wait.")
    assert c.GATE_THRESHOLD == 0.8
    assert c.BEAM_WIDTH == 2
    assert c.OPEN_THINK is False
    assert c.INLINE_SEED == "<think>Wait."


def test_a_malformed_integer_knob_fails_the_import(fresh_config):
    with pytest.raises(ValueError):
        fresh_config(BEAM_WIDTH="4.0")


def test_the_shipped_env_example_loads(fresh_config):
    """Every value .env.example sets parses as its knob's type, so copying it
    to .env cannot break the import."""
    shipped = {k: v for k, v in dotenv_values(ENV_EXAMPLE).items() if v}
    c = fresh_config(**shipped)
    knobs = env_knobs()
    for name, value in shipped.items():
        if knobs.get(name) == "_b":
            assert value.strip().lower() in ("1", "true", "yes", "0", "false", "no"), (
                f"{name}={value!r} is not a recognisable boolean; anything but "
                "1/true/yes silently reads as false")
    assert c.BASE_URL.endswith("/v1")
    assert c.ROOT_URL + "/v1" == c.BASE_URL


def test_env_example_ships_no_secret_and_no_open_seed():
    shipped = dotenv_values(ENV_EXAMPLE)
    assert shipped["OPENAI_API_KEY"] == ""
    # OPEN_SEED needs a real newline, which an unquoted .env value cannot
    # carry; the file only mentions it commented out.
    assert "OPEN_SEED" not in shipped
    # When .env.example sets INLINE_SEED, it is the code default.
    assert shipped.get("INLINE_SEED", config.INLINE_SEED) == config.INLINE_SEED


# ---------------------------------------------------------------------------
# .env.example documents every knob, and nothing else
# ---------------------------------------------------------------------------


def test_the_knob_scan_finds_the_knobs():
    knobs = env_knobs()
    assert len(knobs) >= 40, sorted(knobs)
    assert knobs["BEAM_WIDTH"] == "_i"
    assert knobs["GATE_THRESHOLD"] == "_f"
    assert knobs["OPEN_THINK"] == "_b"
    assert knobs["INLINE_SEED"] == "getenv"
    assert knobs["OPENAI_BASE_URL"] == "getenv"


def test_every_env_knob_config_reads_is_in_env_example():
    missing = sorted(set(env_knobs()) - example_names())
    assert missing == [], f".env.example does not mention: {missing}"


def test_env_example_mentions_no_variable_config_does_not_read():
    stale = sorted(example_names() - set(env_knobs()))
    assert stale == [], f".env.example mentions variables config.py never reads: {stale}"


# ---------------------------------------------------------------------------
# prompt builders: exact bytes
# ---------------------------------------------------------------------------


def test_turn():
    assert chatml.turn("user", "hi") == "<|im_start|>user\nhi<|im_end|>\n"
    assert chatml.turn("system", "") == "<|im_start|>system\n<|im_end|>\n"


def test_assistant_header():
    assert chatml.assistant_header() == "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert chatml.assistant_header(False) == chatml.assistant_header()
    assert chatml.assistant_header(True) == "<|im_start|>assistant\n"


def test_think_head_strips_and_wraps():
    assert chatml.think_head("  x \n") == "<think>\nx\n</think>\n\n"
    assert chatml.think_head("a\n\nb") == "<think>\na\n\nb\n</think>\n\n"


def test_an_empty_think_head_is_the_no_think_block():
    # So a turn whose opening thought is empty renders byte-identically to one
    # with no opening thought at all.
    assert chatml.think_head("") == chatml.NO_THINK
    assert chatml.think_head(" \n ") == chatml.NO_THINK
    assert chatml.assistant_header(True) + chatml.think_head("") == chatml.assistant_header()


def test_levels_are_lettered_from_a():
    assert prompts._levels({"levels": ["a", "b"]}) == "A: a\nB: b"
    assert prompts._levels({"levels": []}) == ""


def test_probe_prompt():
    assert prompts.probe_prompt("CTX", "Q?") == (
        "CTX<|im_end|>\n<|im_start|>user\nQ?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_gate_prompt_bytes():
    a = prompts.GATE_AXES["commitment"]
    expected = (
        "CTX<|im_end|>\n<|im_start|>user\n"
        + a["question"] + "\n\n"
        + "\n".join(f"{L}: {t}" for L, t in zip("ABCDE", a["levels"]))
        + "\n\nAnswer with one letter.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert prompts.gate_prompt("CTX", "commitment") == expected


def test_axis_prompt_bytes():
    a = prompts.AXES["looping"]
    expected = (
        "CTX<|im_end|>\n<|im_start|>user\n"
        "Here is a piece of reasoning:\n\nI think.\n\nThen done.\n\n"
        + a["question"] + "\n\n"
        + "\n".join(f"{L}: {t}" for L, t in zip("ABCDE", a["levels"]))
        + "\n\nAnswer with one letter.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert prompts.axis_prompt("CTX", "  I think.\n\nThen done. \n", "looping") == expected


@pytest.mark.parametrize("axis", list(prompts.GATE_AXES))
def test_every_gate_prompt_ends_on_one_letter_and_a_fresh_answer(axis):
    p = prompts.gate_prompt("CTX", axis)
    assert p.startswith("CTX" + chatml.IM_END + "\n" + chatml.IM_START + "user\n")
    assert p.endswith("Answer with one letter." + chatml.IM_END + "\n"
                      + chatml.assistant_header())
    assert "Here is a piece of reasoning" not in p


@pytest.mark.parametrize("axis", list(prompts.AXES))
def test_every_axis_prompt_ends_on_one_letter_and_a_fresh_answer(axis):
    p = prompts.axis_prompt("CTX", "thought", axis)
    assert p.endswith("Answer with one letter." + chatml.IM_END + "\n"
                      + chatml.assistant_header())
    assert "Here is a piece of reasoning:\n\nthought\n\n" in p


def test_prompt_builders_reject_an_unknown_axis():
    with pytest.raises(KeyError):
        prompts.gate_prompt("CTX", "looping")  # a judge axis, not a gate axis
    with pytest.raises(KeyError):
        prompts.axis_prompt("CTX", "t", "stakes")


def test_choice_prompt_labels_each_option():
    p = prompts.choice_prompt("CTX", ["go left", "go right"], ["A", "B"])
    assert p == (
        "CTX<|im_end|>\n<|im_start|>user\n" + prompts.CHOICE_QUESTION
        + "\n\nA: go left\nB: go right\n\nAnswer with one letter.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_choice_prompt_stops_at_the_shorter_of_options_and_labels():
    p = prompts.choice_prompt("C", ["x", "y", "z"], ["A", "B"], question="Which?")
    assert "\n\nA: x\nB: y\n\nAnswer with one letter." in p
    assert "z" not in p.split("Which?", 1)[1]
    assert prompts.CHOICE_QUESTION not in p


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------


def test_chatml_markers():
    assert (chatml.IM_START, chatml.IM_END) == ("<|im_start|>", "<|im_end|>")
    assert chatml.NO_THINK == "<think>\n\n</think>\n\n"


def test_every_spill_marker_is_also_a_stop_sequence():
    assert set(chatml.SPILL) <= set(chatml.STOP)
    assert {chatml.IM_START, chatml.IM_END, "<think>", "</think>"} <= set(chatml.STOP)


@pytest.mark.parametrize("table", ["AXES", "GATE_AXES"])
def test_every_axis_has_five_distinct_levels_a_question_and_a_positive_weight(table):
    axes = getattr(prompts, table)
    assert axes
    for name, a in axes.items():
        assert len(a["levels"]) == 5, name
        assert len(set(a["levels"])) == 5, name
        assert all(isinstance(t, str) and t.strip() for t in a["levels"]), name
        assert isinstance(a["question"], str) and a["question"].strip(), name
        assert a["weight"] > 0, name


def test_axis_tables_do_not_share_names():
    assert not set(prompts.AXES) & set(prompts.GATE_AXES)


def test_seeds_open_a_think_block():
    assert config.INLINE_SEED.startswith("<think>")
    assert config.OPEN_SEED == "<think>\n"
    # Both contain a SPILL marker, which is why a beam's text never holds its
    # seed (see text.cut_at_marker).
    assert any(m in config.INLINE_SEED for m in chatml.SPILL)


def test_nudge_prefix_marker():
    # tree._expand cuts a copied nudge on its first 12 characters.
    assert prompts.NUDGE_PREFIX[:12] == "Right - back"
    assert not any(m in prompts.NUDGE_PREFIX for m in chatml.SPILL)


def test_yes_and_no_probes():
    assert (prompts.YES, prompts.NO) == ("Yes", "No")
    assert prompts.GATE_PROBE.endswith("Answer with one word, Yes or No.")
    assert prompts.LEAF_PROBE.endswith("Answer with one word, Yes or No.")
