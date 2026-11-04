"""Data management and processing for Labquake Explorer"""
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import h5py
from labquake_explorer.data.event_processor import EventProcessor
from labquake_explorer.data.stamp import stamp

# HDF5 group attribute recording whether a group was written from a dict or a
# list, so digit-keyed dicts and lists round-trip as what they were.  Files
# written without it fall back to the "keys 0..n-1 means list" heuristic.
CONTAINER_ATTR = "container"
SUPPORTED_SUFFIXES = (".npz", ".h5", ".hdf5")

class DataManager:
    def __init__(self):
        self.data: Optional[Dict[str, Any]] = None
        self.data_path: Optional[Path] = None
        self.data: Optional[Dict[str, Any]] = None
        self.event_processor = EventProcessor()

    def load_file(self, path: Path) -> None:
        """Load data from a file"""
        self.data_path = path
        self.event_processor.set_data_path(path)  # Set the data path in EventProcessor

        if path.suffix.lower() == '.npz':
            self._load_npz(path)
        elif path.suffix.lower() in ['.h5', '.hdf5']:
            self._load_hdf5(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _load_npz(self, path: Path) -> None:
        """Load data from NPZ file"""
        with np.load(path, allow_pickle=True) as data:
            self.data = data["experiment"][()]

    def _load_hdf5(self, path: Path) -> None:
        with h5py.File(path, 'r') as h5data:
            self.data = self._load_h5_group(h5data)

    @classmethod
    def _load_h5_item(cls, item):
        if isinstance(item, h5py.Group):
            return cls._load_h5_group(item)
        return cls._load_h5_dataset(item)

    @staticmethod
    def _load_h5_dataset(item):
        """Datasets come back as numpy arrays; scalars as Python scalars.
        String datasets come back as ``str`` when 0-d and as a ``list`` of
        ``str`` otherwise (so a one-element string list stays a list)."""
        try:
            data = np.array(item)
            if data.dtype.kind in ('S', 'O'):
                if data.size == 0:
                    return []
                if isinstance(data.flat[0], bytes):
                    if data.ndim == 0:
                        return data.flat[0].decode('utf-8')
                    return [x.decode('utf-8') for x in data.flat]
            if data.ndim == 0:  # scalar dataset
                return data.item()
            return data
        except Exception as exc:
            print(f"Dataset loading error: {str(exc)}")
            return None

    _SLASH = "%2F"      # '/' is the HDF5 path separator, so dict keys containing it are escaped

    @classmethod
    def _escape_key(cls, key) -> str:
        return str(key).replace("/", cls._SLASH)

    @classmethod
    def _unescape_key(cls, name: str) -> str:
        return name.replace(cls._SLASH, "/")

    @classmethod
    def _load_h5_group(cls, group):
        """Groups written from a list come back as lists, groups written from
        a dict as dicts (``CONTAINER_ATTR``).  Without the marker (files from
        older versions) groups whose keys are exactly 0..n-1 become lists."""
        keys = list(group.keys())
        container = group.attrs.get(CONTAINER_ATTR)
        if isinstance(container, bytes):
            container = container.decode()
        if container == "list" or (container is None and keys and all(k.isdigit() for k in keys)):
            indices = sorted(int(k) for k in keys)
            if indices == list(range(len(indices))):
                return [cls._load_h5_item(group[str(i)]) for i in indices]
        result = {}
        for key in keys:
            try:
                result[cls._unescape_key(key)] = cls._load_h5_item(group[key])
            except Exception as exc:
                print(f"Error loading {key}: {str(exc)}")
        return result

    def save_file(self, path: Path) -> None:
        """Write the experiment to ``path`` (``.npz``, ``.h5`` or ``.hdf5``).

        The file is written to a temporary sibling and moved into place only
        after the whole tree has been written, so a failure part-way through
        never leaves a truncated file behind (the usual flow overwrites the
        file that was just loaded).  Any other suffix raises ``ValueError``
        instead of silently writing nothing.  The top-level ``labquake_explorer``
        dict is refreshed first: the package version writing the file and when
        (``labquake_explorer.data.stamp``).
        """
        if not self.data:
            raise ValueError("No data to save")
        stamp(self.data)
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {path.suffix!r} "
                             f"(use one of {', '.join(SUPPORTED_SUFFIXES)})")

        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}.", suffix=suffix)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            if suffix == '.npz':
                with open(tmp, 'wb') as f:  # a file object keeps numpy from appending '.npz'
                    np.savez(f, experiment=self.data)
            else:
                with h5py.File(tmp, 'w') as f:
                    f.attrs[CONTAINER_ATTR] = "dict"
                    for k, v in self.data.items():
                        self._save_h5_item(f, k, v)
            os.replace(tmp, path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    @classmethod
    def _save_h5_item(cls, group, key, value) -> None:
        """Write one value: dicts and lists of dicts become groups, arrays and
        lists of numbers/strings become datasets, None is skipped.  Lists that
        numpy cannot stack (ragged lists of arrays, mixed content) become a
        group with one entry per index."""
        key = cls._escape_key(key)
        if value is None:
            return
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            subgroup.attrs[CONTAINER_ATTR] = "dict"
            for k, v in value.items():
                cls._save_h5_item(subgroup, k, v)
            return
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                group.create_dataset(key, data=np.zeros(0))
                return
            stacked = None
            if not any(isinstance(x, (dict, list, tuple, type(None))) for x in value):
                try:
                    stacked = np.array(value)
                except (ValueError, TypeError):  # ragged arrays cannot be stacked
                    stacked = None
                if stacked is not None and stacked.dtype == object:
                    stacked = None  # inhomogeneous content
            if stacked is None:
                subgroup = group.create_group(key)
                subgroup.attrs[CONTAINER_ATTR] = "list"
                for i, item in enumerate(value):
                    cls._save_h5_item(subgroup, str(i), item)
                return
            value = stacked
        if isinstance(value, np.ndarray):
            arr = value
            if arr.dtype == object:
                if all(isinstance(x, (bool, np.bool_)) for x in arr.flat):
                    arr = arr.astype(np.int8)
                elif all(isinstance(x, (int, np.integer)) for x in arr.flat):
                    arr = arr.astype(np.int64)
                elif all(isinstance(x, (int, float, np.integer, np.floating)) for x in arr.flat):
                    arr = arr.astype(np.float64)
                elif arr.ndim == 1 and any(isinstance(x, (np.ndarray, list, tuple, dict)) for x in arr.flat):
                    cls._save_h5_item(group, key, list(arr))  # ragged object array -> per-index group
                    return
                else:
                    arr = np.array([str(x).encode() for x in arr.flat]).reshape(arr.shape)
            elif arr.dtype.kind == 'U':
                arr = np.array([x.encode() for x in arr.flat]).reshape(arr.shape)
            elif arr.dtype.kind == 'b':
                arr = arr.astype(np.int8)
            if arr.ndim == 0:
                group.create_dataset(key, data=arr)
            else:
                group.create_dataset(key, data=arr, compression="gzip" if arr.size > 1 else None)
            return
        if isinstance(value, str):
            group.create_dataset(key, data=value.encode())
        elif isinstance(value, (bool, np.bool_)):
            group.create_dataset(key, data=int(value))
        elif isinstance(value, (int, float, np.number)):
            group.create_dataset(key, data=value)
        else:
            try:
                group.create_dataset(key, data=np.array(value))
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not save {key}: {e}")

    def extract_events(self, indices: List[int], window_size: float) -> List[Dict]:
        """Extract events using provided indices"""
        if not self.data:
            raise ValueError("No data loaded")
            
        events = []
        for idx in indices:
            event = self._extract_single_event(idx, window_size)
            events.append(event)
        return events

    def _extract_single_event(self, idx: int, window: float) -> Dict:
        """Extract single event data"""
        event_time = self.data["time"][idx]
        
        idx_beg = np.argmin(np.abs(event_time - window - self.data["time"]))
        idx_end = np.argmin(np.abs(event_time + window - self.data["time"]))
        idx_event = range(idx_beg, idx_end + 1)
        
        event = {
            'event_time': event_time,
            'time': self.data['time'][idx_event]
        }
        
        for key, value in self.data.items():
            if key != "events" and isinstance(value, (np.ndarray, list)):
                try:
                    event[key] = value[idx_event]
                except IndexError:
                    event[key] = value[idx]
                    
        if 'strain' in self.data:
            event['strain'] = self._process_strain_data(event_time, window)
            
        return event

    def get_data(self, path: str) -> Any:
        """Get data at specified path (e.g. 'runs/[0]/events/[3]/shear_stress')."""
        if not self.data:
            raise ValueError("No data loaded")
        current = self.data
        for key in self._split(path):
            current = current[key]
        return current

    @staticmethod
    def _split(path: str) -> list:
        parts = []
        for part in path.replace("\\", "/").split("/"):
            if not part:
                continue
            if part.startswith('[') and part.endswith(']'):
                parts.append(int(part[1:-1]))
            else:
                parts.append(part)
        return parts

    def set_data(self, path: str, value: Any, add_key: bool = False) -> None:
        """Set data at specified path.

        With ``add_key=True`` missing intermediate dictionaries are created;
        otherwise a missing parent raises KeyError.
        """
        if not self.data:
            raise ValueError("No data loaded")
        parts = self._split(path)
        if not parts:
            raise ValueError("Cannot set the root")
        current = self.data
        for part in parts[:-1]:
            if isinstance(current, dict) and part not in current:
                if not add_key:
                    raise KeyError(f"Key '{part}' not found in path '{path}'")
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def delete_data(self, path: str) -> None:
        """Delete data at specified path (e.g. 'runs/[0]/events'); '' clears everything.

        Raises ValueError for an invalid path, KeyError/IndexError when it
        does not exist.
        """
        if not self.data:
            raise ValueError("No data loaded")
        parts = self._split(path)
        if not parts:
            self.data = None
            return
        current = self.data
        for part in parts[:-1]:
            if isinstance(part, int):
                if not isinstance(current, (list, tuple)):
                    raise ValueError(f"Cannot index non-sequence with [{part}]")
                if part >= len(current):
                    raise IndexError(f"Index {part} out of range for sequence of length {len(current)}")
            else:
                if not isinstance(current, dict):
                    raise ValueError(f"Cannot get key '{part}' from non-dictionary")
                if part not in current:
                    raise KeyError(f"Key '{part}' not found")
            current = current[part]
        last = parts[-1]
        if isinstance(last, int):
            if not isinstance(current, list):
                raise ValueError("Cannot delete index from non-list")
            if last >= len(current):
                raise IndexError(f"Index {last} out of range")
            current.pop(last)
        else:
            if not isinstance(current, dict):
                raise ValueError("Cannot delete key from non-dictionary")
            if last not in current:
                raise KeyError(f"Key '{last}' not found")
            current.pop(last)
