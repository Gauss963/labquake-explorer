import re
from pathlib import Path

from setuptools import setup, find_packages

# The one place the version is written: labquake_explorer/_version.py.
VERSION = re.search(r'__version__ = "([^"]+)"',
                    (Path(__file__).parent / "labquake_explorer" / "_version.py").read_text()).group(1)

setup(
    name="labquake-explorer",
    version=VERSION,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "h5py",
        "matplotlib",
        "scipy",
        "pandas",
    ],
    entry_points={
        'console_scripts': [
            'labquake-explorer=labquake_explorer.main:main',
        ],
    },
    python_requires='>=3.10',
)