"""Tests for pcadiv.utils.logging."""
from __future__ import annotations

import logging

from pcadiv.utils.logging import configure, get_logger


def test_get_logger_returns_logger():
    log = get_logger("pcadiv.test")
    assert isinstance(log, logging.Logger)
    assert log.name == "pcadiv.test"


def test_get_logger_does_not_configure_root():
    root = logging.getLogger()
    before = list(root.handlers)
    get_logger("pcadiv.test")
    assert list(root.handlers) == before


def test_configure_idempotent():
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    root.handlers.clear()
    try:
        configure(level=logging.DEBUG)
        configure(level=logging.DEBUG)
        assert len(root.handlers) == 1
        assert root.level == logging.DEBUG
    finally:
        root.handlers.clear()
        for h in original_handlers:
            root.addHandler(h)
        root.setLevel(original_level)