"""Configuration settings for Labquake Explorer"""
from dataclasses import dataclass

@dataclass
class LabquakeExplorerConfig:
    """Configuration settings for the application"""
    WINDOW_GAP: int = 100
    WINDOW_WIDTH: int = 300
    WINDOW_TITLE: str = "Labquake Explorer"
    DEFAULT_WINDOW_SIZE: float = 5.0
    FILE_TYPES: tuple = (                 # first entry is the dialogs' default
        ("HDF5 files", "*.h5 *.hdf5"),
        ("NPZ files", "*.npz"),
        ("All files", "*.*")
    )
    SAVE_SUFFIX: str = ".h5"