"""Tests for pcadiv.utils.io."""
from __future__ import annotations

import numpy as np

from pcadiv.utils.io import ensure_dir, load_npz, save_npz, sha1_of_texts


# ---- sha1_of_texts ----------------------------------------------------

def test_sha1_deterministic():
    a = sha1_of_texts(["hello", "world"])
    b = sha1_of_texts(["hello", "world"])
    assert a == b
    assert len(a) == 40
    assert all(c in "0123456789abcdef" for c in a)


def test_sha1_order_sensitive():
    assert sha1_of_texts(["a", "b"]) != sha1_of_texts(["b", "a"])


def test_sha1_concatenation_does_not_collide():
    # separator prevents ["ab", "c"] == ["a", "bc"]
    assert sha1_of_texts(["ab", "c"]) != sha1_of_texts(["a", "bc"])


def test_sha1_empty_vs_empty_string():
    # [] feeds nothing to the hasher; [""] feeds just the separator
    assert sha1_of_texts([]) != sha1_of_texts([""])


def test_sha1_unicode():
    assert sha1_of_texts(["привет"]) == sha1_of_texts(["привет"])
    assert sha1_of_texts(["привет"]) != sha1_of_texts(["privet"])


# ---- ensure_dir -------------------------------------------------------

def test_ensure_dir_creates_nested(tmp_path):
    p = ensure_dir(tmp_path / "a" / "b" / "c")
    assert p.is_dir()


def test_ensure_dir_idempotent(tmp_path):
    p1 = ensure_dir(tmp_path / "x")
    p2 = ensure_dir(tmp_path / "x")
    assert p1 == p2


# ---- save_npz / load_npz ---------------------------------------------

def test_save_load_roundtrip(tmp_path):
    arrays = {
        "x": np.arange(10, dtype=np.float32),
        "y": np.ones((3, 4), dtype=np.float64),
    }
    p = save_npz(tmp_path / "arr.npz", **arrays)
    loaded = load_npz(p)
    assert set(loaded) == set(arrays)
    for k in arrays:
        np.testing.assert_array_equal(loaded[k], arrays[k])


def test_save_npz_appends_extension(tmp_path):
    p = save_npz(tmp_path / "arr", x=np.arange(3))
    assert p.name == "arr.npz"
    assert p.exists()


def test_save_npz_no_leftover_tmp(tmp_path):
    save_npz(tmp_path / "arr.npz", x=np.arange(3))
    files = sorted(f.name for f in tmp_path.iterdir())
    assert files == ["arr.npz"]


def test_save_npz_overwrites_existing(tmp_path):
    save_npz(tmp_path / "arr.npz", x=np.array([1, 2]))
    save_npz(tmp_path / "arr.npz", x=np.array([3, 4]))
    loaded = load_npz(tmp_path / "arr.npz")
    np.testing.assert_array_equal(loaded["x"], np.array([3, 4]))


def test_load_npz_returns_plain_dict(tmp_path):
    p = save_npz(tmp_path / "d.npz", a=np.array([1, 2, 3]))
    d = load_npz(p)
    assert type(d) is dict
    assert list(d) == ["a"]