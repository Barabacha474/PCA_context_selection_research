"""Thin logging wrapper.

Why not use :func:`logging.getLogger` directly? Two reasons:

1. We want a single, consistent format across the project that is
   configured **once**, in ``experiments/`` entry points -- not
   scattered across library modules.

2. Modules must be importable without side effects: importing
   :mod:`pcadiv.utils.logging` must not configure the root logger.
   Configuring is the caller's job (see :func:`configure`).
"""
from __future__ import annotations

import logging
import sys

__all__ = ["get_logger", "configure"]

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger without touching the root logger."""
    return logging.getLogger(name)


def configure(level: int = logging.INFO, stream=sys.stderr) -> None:
    """Configure the root logger once. Call from entry points only.

    Idempotent: repeated calls do not stack handlers. If the root logger
    already has handlers (e.g. because Jupyter or a test runner set them
    up), we leave them alone and only adjust the level.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
        root.addHandler(handler)
    root.setLevel(level)