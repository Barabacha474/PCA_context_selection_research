"""Small I/O helpers used across the project.

Two things live here on purpose:

1. :func:`sha1_of_texts` -- the canonical hashing primitive used by the
   embedding cache. It is **order-sensitive**: the same texts in a
   different order yield a different digest. This is intentional --
   cache keys must not silently alias permutations of a pool.

2. NPZ helpers -- thin wrappers that guarantee an atomic write, so a
   crashed process cannot leave a half-written cache file that the next
   run would happily load.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "ensure_dir",
    "sha1_of_texts",
    "save_npz",
    "load_npz",
]


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create directory (and parents) if it does not exist; return it as :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha1_of_texts(texts: Sequence[str]) -> str:
    """Return a SHA-1 hex digest of a sequence of texts.

    The digest is **order-sensitive**: ``["a", "b"]`` and ``["b", "a"]``
    hash differently. Texts are separated by the ASCII unit separator
    (``\\x1f``), which cannot appear in normal text, so distinct
    sequences cannot collide via concatenation alone (``["ab", "c"]``
    and ``["a", "bc"]`` hash differently).

    Parameters
    ----------
    texts
        Sequence of strings.

    Returns
    -------
    str
        40-character hexadecimal SHA-1 digest.
    """
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def save_npz(path: str | os.PathLike[str], **arrays: np.ndarray) -> Path:
    """Atomically save arrays to a ``.npz`` file.

    If ``path`` does not end in ``.npz``, the extension is appended
    (mirrors :func:`numpy.savez`). The write goes to a temporary file in
    the same directory and is then ``os.replace``-d into place, so a
    crash mid-write cannot leave a corrupted target file.

    Parameters
    ----------
    path
        Destination path.
    **arrays
        Named arrays to save.

    Returns
    -------
    pathlib.Path
        The final path, with ``.npz`` ensured.
    """
    p = Path(path)
    if p.suffix != ".npz":
        p = p.with_name(p.name + ".npz")
    ensure_dir(p.parent)

    fd, tmp_name = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".", suffix=".npz")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, p)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return p


def load_npz(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Load a ``.npz`` file into a plain dict of arrays.

    Returns a real :class:`dict` (not an ``NpzFile``) so the underlying
    file handle is closed immediately and callers cannot accidentally
    hold it open across the rest of a pipeline.
    """
    with np.load(Path(path)) as data:
        return {k: data[k] for k in data.files}