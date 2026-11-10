"""Experiment-level analysis settings: ``data['analysis']``.

Everything a view stores for the whole experiment, as opposed to one event,
goes into this one dict, so that the top level of a file keeps to the
metadata the preprocessing wrote (``name``, ``date``, ``sample``, ``fault``,
...) plus ``runs``:

* ``analysis['fault']``: ``patch_x_mm`` and ``ends_x_mm``, the bare PMMA
  patch edges and where the fault ends, in the frame of the sensor positions;
* ``analysis['medium']``: ``rho_kg_m3``, ``vp_m_s``, ``vs_m_s`` of the sample;
* ``analysis['pzt_spectrum']``: the PZT Spectrum view's last saved controls
  and per-sensor tables per result set (``sets[<name>]['settings']``,
  ``sets[<name>]['sensors']``) and the set last saved (``current_set``);
  files written before result sets keep flat ``settings``/``sensors``, read
  as the set ``default`` and moved on the next save.

Values are read with :func:`get_setting` and written with :func:`set_setting`,
which creates the dicts on the way.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

ANALYSIS_KEY = "analysis"


def analysis_dict(data) -> Optional[Mapping]:
    """``data['analysis']`` when it is a dict, else None."""
    value = data.get(ANALYSIS_KEY) if isinstance(data, Mapping) else None
    return value if isinstance(value, Mapping) else None


def get_setting(data, *keys: str, default: Any = None) -> Any:
    """``data['analysis'][keys[0]][keys[1]]...``, or ``default`` when any step is missing."""
    current: Any = analysis_dict(data)
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def set_setting(data: dict, value: Any, *keys: str) -> None:
    """Store ``value`` at ``data['analysis'][keys...]``, creating the dicts on the way."""
    if not isinstance(data, dict):
        raise TypeError("the experiment must be a dict")
    if not keys:
        raise ValueError("set_setting needs at least one key")
    current = data
    for key in (ANALYSIS_KEY,) + keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[keys[-1]] = value


def delete_setting(data, *keys: str) -> None:
    """Remove ``data['analysis'][keys...]`` and the dicts it leaves empty."""
    if not keys:
        return
    chain = []
    current = data.get(ANALYSIS_KEY) if isinstance(data, dict) else None
    for key in keys[:-1]:
        if not isinstance(current, dict):
            return
        chain.append((current, key))
        current = current.get(key)
    if isinstance(current, dict):
        current.pop(keys[-1], None)
    for parent, key in reversed(chain):
        if isinstance(parent.get(key), dict) and not parent[key]:
            del parent[key]
    if isinstance(data, dict) and isinstance(data.get(ANALYSIS_KEY), dict) and not data[ANALYSIS_KEY]:
        del data[ANALYSIS_KEY]


__all__ = ["ANALYSIS_KEY", "analysis_dict", "get_setting", "set_setting", "delete_setting"]
