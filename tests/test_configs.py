"""Tests for config loading and shipped YAML configs -- Block 12a."""
from __future__ import annotations

from pathlib import Path

import pytest

from pcadiv.utils.config import load_config


# ---- loader mechanics ---------------------------------------------

def test_load_simple_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\nb:\n  c: hello\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg == {"a": 1, "b": {"c": "hello"}}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("a: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_config(p)


def test_non_mapping_root_raises(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(p)


def test_empty_yaml_returns_empty_dict(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_config(p) == {}


def test_override_int(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    cfg = load_config(p, ["a=42"])
    assert cfg["a"] == 42


def test_override_float(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    cfg = load_config(p, ["a=3.14"])
    assert cfg["a"] == pytest.approx(3.14)


def test_override_bool_true_false(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: false\nb: true\n", encoding="utf-8")
    cfg = load_config(p, ["a=true", "b=false"])
    assert cfg["a"] is True
    assert cfg["b"] is False


def test_override_null(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 5\n", encoding="utf-8")
    cfg = load_config(p, ["a=null"])
    assert cfg["a"] is None


def test_override_string_fallback(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    cfg = load_config(p, ["a=not-a-number"])
    assert cfg["a"] == "not-a-number"


def test_override_nested_existing_path(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a:\n  b:\n    c: 1\n", encoding="utf-8")
    cfg = load_config(p, ["a.b.c=42"])
    assert cfg["a"]["b"]["c"] == 42


def test_override_nested_new_path(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    cfg = load_config(p, ["x.y.z=9"])
    assert cfg["x"]["y"]["z"] == 9


def test_override_mapping_form(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a:\n  b: 1\n", encoding="utf-8")
    cfg = load_config(p, {"a.b": 99})
    assert cfg["a"]["b"] == 99


def test_override_malformed_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be 'key=value'"):
        load_config(p, ["no_equals_sign"])


def test_override_last_wins(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    cfg = load_config(p, ["a=2", "a=3"])
    assert cfg["a"] == 3


# ---- shipped configs are valid ------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("rel_path", [
    "configs/data/writingprompts.yaml",
    "configs/embed/selection_encoder.yaml",
    "configs/embed/eval_encoder.yaml",
    "configs/experiments/main_sweep.yaml",
])
def test_shipped_config_parses(rel_path):
    cfg = load_config(PROJECT_ROOT / rel_path)
    assert isinstance(cfg, dict)


def test_data_config_has_required_fields():
    cfg = load_config(PROJECT_ROOT / "configs/data/writingprompts.yaml")
    assert "source" in cfg and "hf_dataset" in cfg["source"]
    assert "tokenizer" in cfg and "name" in cfg["tokenizer"]
    assert "passage_tokens" in cfg["passageize"]
    assert "split" in cfg and "train_frac" in cfg["split"]
    assert {"primary", "ood"} <= set(cfg["pools"])


def test_main_sweep_config_structure():
    cfg = load_config(PROJECT_ROOT / "configs/experiments/main_sweep.yaml")
    assert isinstance(cfg["selectors"], list) and len(cfg["selectors"]) > 0
    for item in cfg["selectors"]:
        assert "name" in item
    assert isinstance(cfg["budgets"], list) and all(b > 0 for b in cfg["budgets"])
    assert isinstance(cfg["seeds"], list) and len(cfg["seeds"]) >= 1
    assert {"selection", "eval"} <= set(cfg["encoders"])


def test_selection_and_eval_encoders_are_different():
    """Both must be defined; the test does not require different models
    (debug configs may use tiny models), but it does require that both
    files parse and have model_id."""
    sel = load_config(PROJECT_ROOT / "configs/embed/selection_encoder.yaml")
    ev = load_config(PROJECT_ROOT / "configs/embed/eval_encoder.yaml")
    assert "model_id" in sel
    assert "model_id" in ev