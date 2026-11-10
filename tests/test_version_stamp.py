"""The package version, written once and recorded in every file written."""
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import pytest

import labquake_explorer
from labquake_explorer._version import __version__
from labquake_explorer.data import migrate as M
from labquake_explorer.data import stamp as S
from labquake_explorer.data.data_manager import DataManager
from labquake_explorer.preprocessing.build import experiment
from tests.synthetic import make_experiment

ROOT = Path(__file__).resolve().parents[1]


def saved(tmp_path, data, suffix=".h5"):
    dm = DataManager()
    dm.data = data
    path = tmp_path / f"exp{suffix}"
    dm.save_file(path)
    back = DataManager()
    back.load_file(path)
    return path, back.data


def test_one_version_everywhere():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
    assert labquake_explorer.__version__ == __version__
    assert "_version.py" in (ROOT / "setup.py").read_text()     # read from the one place, not repeated
    out = subprocess.run([sys.executable, "-m", "labquake_explorer.main", "--version"],
                         capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert out.returncode == 0 and out.stdout.strip() == f"labquake-explorer {__version__}"


def test_git_description_names_this_checkout():
    text = S.git_description()
    if (ROOT / ".git").exists():
        assert text and re.search(r"[0-9a-f]{7}", text)
    else:
        assert text is None


def test_experiment_records_the_version_that_built_it():
    exp = experiment("t0001", [], date="2026-01-01")
    entry = exp[S.STAMP_KEY]
    assert entry["created_version"] == __version__ == entry["version"]
    assert entry["created_at"] == entry["saved_at"]
    assert datetime.fromisoformat(entry["saved_at"]).utcoffset() is not None
    assert list(exp) == ["name", "date", S.STAMP_KEY, "runs"]


@pytest.mark.parametrize("suffix", [".h5", ".npz"])
def test_an_older_file_gets_a_stamp_when_saved(tmp_path, suffix):
    data, _ = make_experiment(recurrence=12.0)
    assert S.STAMP_KEY not in data
    _, back = saved(tmp_path, data, suffix)
    entry = back[S.STAMP_KEY]
    assert entry["version"] == __version__
    assert datetime.fromisoformat(entry["saved_at"]).year >= 2026
    assert entry.get("git") == S.git_description()
    assert not any(key in entry for key in S.CREATED_KEYS)          # it was not built by this package
    assert data[S.STAMP_KEY] == entry                                # the dict in memory carries the same


def test_a_save_refreshes_the_writer_and_keeps_the_builder(tmp_path):
    data, _ = make_experiment(recurrence=12.0)
    old = {"version": "0.0.1", "saved_at": "2024-01-16T00:00:00+00:00", "git": "gone",
           "created_version": "0.0.1", "created_at": "2024-01-16T00:00:00+00:00"}
    data[S.STAMP_KEY] = dict(old)
    _, back = saved(tmp_path, data)
    entry = back[S.STAMP_KEY]
    assert entry["version"] == __version__ and entry["saved_at"] != old["saved_at"]
    assert entry["created_version"] == old["created_version"]
    assert entry["created_at"] == old["created_at"]
    assert entry.get("git") == S.git_description()                   # never the stale one


def test_in_place_geometry_write_refreshes_the_stamp(tmp_path):
    data, _ = make_experiment(recurrence=12.0)
    path, before = saved(tmp_path, data)
    with h5py.File(path, "r+") as f:                                 # make the stamp look old
        del f[f"{S.STAMP_KEY}/version"]
        f[f"{S.STAMP_KEY}/version"] = b"0.0.1"
    assert M.write_fault_geometry(path, ends_x_mm=(0, 500))
    back = DataManager()
    back.load_file(path)
    entry = back.data[S.STAMP_KEY]
    assert entry["version"] == __version__
    assert entry.get("git") == S.git_description()
    assert M.write_fault_geometry(path, ends_x_mm=(0, 500)) == []     # nothing to do: nothing touched
    again = DataManager()
    again.load_file(path)
    assert again.data[S.STAMP_KEY] == entry
