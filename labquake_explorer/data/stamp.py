"""Which version of the package wrote an experiment file, and when: the
``labquake_explorer`` dict at the top of the file.

* ``version`` and ``saved_at``: the package that last wrote the file and the
  local time it did (ISO 8601 with the UTC offset).  Every write refreshes
  them: ``DataManager.save_file`` and the migration tool's in-place edits.
* ``git``: ``git describe`` of the checkout the package ran from
  (``v0.2.0-3-g1bf4fca``, ``-dirty`` when it had uncommitted edits), so a file
  written between releases still names its commit.  Absent when the package
  did not run from a checkout.
* ``created_version`` and ``created_at``: the package that built the
  experiment (``preprocessing.build.experiment``); later saves leave them.

A file made before the stamp existed gets one when it is next saved.
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from labquake_explorer._version import __version__

STAMP_KEY = "labquake_explorer"
#: The entries every write refreshes.
SAVED_KEYS = ("version", "saved_at", "git")
#: The entries set once, when the experiment is built.
CREATED_KEYS = ("created_version", "created_at")

_REPO = Path(__file__).resolve().parents[2]   # the checkout the package lives in, if it is one


def now_iso() -> str:
    """Local time to the second with the UTC offset: ``2026-11-01T11:03:12+08:00``."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git_description() -> Optional[str]:
    """``git describe --tags --always --dirty`` of the checkout the package runs
    from, or None when it is not one (an installed copy) or git is missing."""
    if not (_REPO / ".git").exists():          # a directory, or a file in a worktree
        return None
    try:
        done = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                              cwd=_REPO, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    text = done.stdout.strip()
    return text if done.returncode == 0 and text else None


def saved_stamp() -> dict:
    """The entries of a write happening now."""
    entry = {"version": __version__, "saved_at": now_iso()}
    git = git_description()
    if git:
        entry["git"] = git
    return entry


def stamp(data: dict, created: bool = False) -> dict:
    """Refresh ``data['labquake_explorer']`` for a write happening now and
    return it; with ``created`` also record this version as the one that built
    the experiment."""
    entry = data.get(STAMP_KEY)
    if not isinstance(entry, dict):
        entry = {}
        data[STAMP_KEY] = entry
    entry.pop("git", None)                     # not run from a checkout: no stale description
    entry.update(saved_stamp())
    if created:
        entry["created_version"] = __version__
        entry["created_at"] = entry["saved_at"]
    return entry


def write_stamp_h5(f) -> None:
    """Refresh the stamp of an HDF5 file edited in place (the small group only)."""
    from labquake_explorer.data.data_manager import CONTAINER_ATTR
    group = f.require_group(STAMP_KEY)
    group.attrs[CONTAINER_ATTR] = "dict"
    for key in SAVED_KEYS:
        if key in group:
            del group[key]
    for key, value in saved_stamp().items():
        group.create_dataset(key, data=value.encode())


__all__ = ["STAMP_KEY", "SAVED_KEYS", "CREATED_KEYS", "git_description", "stamp", "write_stamp_h5"]
