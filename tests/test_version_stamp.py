"""The package version, written once and recorded in every file written."""
import re
import sys
from datetime import datetime
from pathlib import Path

import h5py

import labquake_explorer
from labquake_explorer import main as entry_point
from labquake_explorer._version import __version__
from labquake_explorer.data import migrate as M
from labquake_explorer.data import stamp as S
from labquake_explorer.preprocessing.build import experiment
from tests.synthetic import load_manager, make_experiment, save_data

ROOT = Path(__file__).resolve().parents[1]


def saved(tmp_path, data, suffix=".h5"):
    path = tmp_path / f"exp{suffix}"
    save_data(data, path)
    return path, load_manager(path).data[S.STAMP_KEY]


def test_one_version_everywhere(monkeypatch, capsys):
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
    assert labquake_explorer.__version__ == __version__
    assert "_version.py" in (ROOT / "setup.py").read_text()     # read from the one place, not repeated
    monkeypatch.setattr(sys, "argv", ["labquake-explorer", "--version"])
    entry_point.main()
    assert capsys.readouterr().out.strip() == f"labquake-explorer {__version__}"
    text = S.git_description()
    if (ROOT / ".git").exists():
        assert text and re.search(r"[0-9a-f]{7}", text)
    else:
        assert text is None
    # an experiment records the version that built it
    exp = experiment("t0001", [], date="2026-01-01")
    entry = exp[S.STAMP_KEY]
    assert entry["created_version"] == __version__ == entry["version"]
    assert entry["created_at"] == entry["saved_at"]
    assert datetime.fromisoformat(entry["saved_at"]).utcoffset() is not None
    assert list(exp) == ["name", "date", S.STAMP_KEY, "runs"]


def test_saving_stamps_the_writer_and_keeps_the_builder(tmp_path):
    data, _ = make_experiment(recurrence=12.0)
    assert S.STAMP_KEY not in data
    for suffix in (".npz", ".h5"):                                  # an older file gets a stamp when saved
        data.pop(S.STAMP_KEY, None)
        _, entry = saved(tmp_path, data, suffix)
        assert entry["version"] == __version__ and entry.get("git") == S.git_description()
        assert datetime.fromisoformat(entry["saved_at"]).year >= 2026
        assert not any(key in entry for key in S.CREATED_KEYS)      # it was not built by this package
        assert data[S.STAMP_KEY] == entry                           # the dict in memory carries the same
    # a save refreshes the writer and keeps the builder
    old = {"version": "0.0.1", "saved_at": "2024-01-16T00:00:00+00:00", "git": "gone",
           "created_version": "0.0.1", "created_at": "2024-01-16T00:00:00+00:00"}
    data[S.STAMP_KEY] = dict(old)
    path, entry = saved(tmp_path, data)
    assert entry["version"] == __version__ and entry["saved_at"] != old["saved_at"]
    assert entry["created_version"] == old["created_version"] and entry["created_at"] == old["created_at"]
    assert entry.get("git") == S.git_description()                  # never the stale one
    # an in-place geometry write refreshes the stamp too
    with h5py.File(path, "r+") as f:
        del f[f"{S.STAMP_KEY}/version"]
        f[f"{S.STAMP_KEY}/version"] = b"0.0.1"
    assert M.write_fault_geometry(path, ends_x_mm=(0, 500))
    entry = load_manager(path).data[S.STAMP_KEY]
    assert entry["version"] == __version__ and entry.get("git") == S.git_description()
    assert M.write_fault_geometry(path, ends_x_mm=(0, 500)) == []     # nothing to do: nothing touched
    assert load_manager(path).data[S.STAMP_KEY] == entry
