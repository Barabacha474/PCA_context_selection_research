"""YAML configuration loading with dot-path overrides.

Why not Hydra/OmegaConf
-----------------------
We only need three things: load a YAML file, apply a handful of CLI
overrides for debug, and return a plain dict that downstream code can
read without a dependency on a config DSL. Hydra/OmegaConf bring
a lot of machinery (config groups, interpolation, structured configs)
that this project does not use. A ~50-line loader is auditable and
has no transitive dependency footprint.

Override format
---------------
Each override is ``"a.b.c=value"``. ``value`` is parsed by
``json.loads``; on failure it is taken as a raw string. This gives
``true``/``false``/``null``/``123``/``1.5``/``"quoted string"``
without inventing our own type parser.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

__all__ = ["load_config"]


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _set_nested(cfg: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    if not all(keys):
        raise ValueError(f"invalid override path: {path!r}")
    *parents, leaf = keys
    node = cfg
    for k in parents:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[leaf] = value


def load_config(
    path: str | Path,
    overrides: Sequence[str] | Mapping[str, Any] | None = None,
) -> dict:
    """Load a YAML config file, apply overrides, return a plain dict.

    Parameters
    ----------
    path
        Path to the YAML file.
    overrides
        Either a sequence of ``"a.b.c=value"`` strings (CLI style) or a
        mapping ``{"a.b.c": value}`` (test / programmatic style). The
        mapping form skips value parsing -- values are used as-is.

    Returns
    -------
    dict
        The config tree, with overrides applied in order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is not valid YAML, or if the top level is not a
        mapping (list root is not a valid config for this project).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    raw = p.read_text(encoding="utf-8")
    try:
        cfg = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ValueError(f"invalid YAML in {p}: {e}") from e
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"top level of {p} must be a mapping, got {type(cfg).__name__}"
        )

    if overrides is None:
        return cfg

    if isinstance(overrides, Mapping):
        for k, v in overrides.items():
            _set_nested(cfg, k, v)
    else:
        for item in overrides:
            if "=" not in item:
                raise ValueError(
                    f"override must be 'key=value', got {item!r}"
                )
            k, raw_v = item.split("=", 1)
            _set_nested(cfg, k.strip(), _parse_value(raw_v))

    return cfg