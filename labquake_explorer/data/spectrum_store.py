"""The spectrum store of a full-rate record: ``block['spectrum']``.

Version 2 keeps named result sets, so that analyses of the same channel by
different routes (a BAC displacement calibration, an EGF moment calibration)
live side by side instead of overwriting each other::

    {"version": 2, "sets": {"<name>": {"channels": {"ch<k>": record}}}}

Version 1 held one set of channels, ``{"version": 1, "channels": {...}}``:
it is read as the set ``'default'`` and becomes version 2 when next written
(:func:`ensure_set`).  The event-level ``pzt_spectrum`` map of still older
files is the ``'default'`` set of the record each entry was computed on; the
views and ``data.migrate`` move it.
"""
from __future__ import annotations

from typing import Mapping, Optional

STORE_VERSION = 2
DEFAULT_SET = "default"
SETS_KEY = "sets"
CHANNELS_KEY = "channels"


def set_name(name) -> str:
    """A validated result-set name: stripped, non-empty, without ``/`` (an HDF5
    group name).  Raises ValueError."""
    text = str(name if name is not None else "").strip()
    if not text:
        raise ValueError("the result set needs a name")
    if "/" in text:
        raise ValueError("a result-set name cannot contain '/'")
    return text


def store_sets(store) -> dict:
    """``{name: {"channels": ...}}`` of a store, whatever its version (a version-1
    store is the set ``'default'``); ``{}`` for anything that is not a store."""
    if not isinstance(store, Mapping):
        return {}
    sets = store.get(SETS_KEY)
    if isinstance(sets, Mapping):
        return {str(name): entry for name, entry in sets.items() if isinstance(entry, Mapping)}
    if CHANNELS_KEY in store:
        return {DEFAULT_SET: store}
    return {}


def set_names(store) -> list:
    """The result sets of a store, ``'default'`` first, the rest sorted."""
    names = set(store_sets(store))
    return ([DEFAULT_SET] if DEFAULT_SET in names else []) + sorted(names - {DEFAULT_SET})


def set_channels(store, name: str = DEFAULT_SET):
    """The ``channels`` value (a dict, or the list an older HDF5 file gives) of
    one result set, or None when the store has no such set."""
    entry = store_sets(store).get(name)
    return entry.get(CHANNELS_KEY) if isinstance(entry, Mapping) else None


def upgrade_store(store: dict) -> bool:
    """Bring a store dict to version 2 in place: version-1 channels become the
    set ``'default'``.  True when something changed."""
    if not isinstance(store, dict):
        return False
    changed = False
    if not isinstance(store.get(SETS_KEY), dict):
        store[SETS_KEY] = {}
        changed = True
    if CHANNELS_KEY in store:
        channels = store.pop(CHANNELS_KEY)
        store[SETS_KEY].setdefault(DEFAULT_SET, {})
        if not isinstance(store[SETS_KEY][DEFAULT_SET], dict):
            store[SETS_KEY][DEFAULT_SET] = {}
        store[SETS_KEY][DEFAULT_SET].setdefault(CHANNELS_KEY, channels)
        changed = True
    if store.get("version") != STORE_VERSION:
        store["version"] = STORE_VERSION
        changed = True
    return changed


def ensure_set(store: dict, name: str = DEFAULT_SET) -> dict:
    """Upgrade ``store`` in place and return its set ``name``, created with an
    empty ``channels`` dict when missing."""
    upgrade_store(store)
    entry = store[SETS_KEY].get(name)
    if not isinstance(entry, dict):
        entry = {}
        store[SETS_KEY][name] = entry
    if not isinstance(entry.get(CHANNELS_KEY), dict):
        entry[CHANNELS_KEY] = {}
    return entry


__all__ = ["STORE_VERSION", "DEFAULT_SET", "SETS_KEY", "CHANNELS_KEY", "set_name", "store_sets",
           "set_names", "set_channels", "upgrade_store", "ensure_set"]
