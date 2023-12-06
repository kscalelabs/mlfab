#!/usr/bin/env python
"""Setup mlfab."""

import os
import re

from setuptools import setup

DEV_REQUIREMENTS = [
    "black",
    "darglint",
    "mypy",
    "pytest",
    "pytest-timeout",
    "ruff",
]


with open("mlfab/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in mlfab/__init__.py"
version: str = version_re.group(1)


with open("mlfab/requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()


with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("mlfab/scripts/create_mlfab_project/assets.txt") as f:
    files = [line.strip() for line in f if line]
create_mlfab_project_assets = [f"scripts/create_mlfab_project/assets/{file}" for file in files]
missing_files = [file for file in create_mlfab_project_assets if not os.path.exists(f"mlfab/{file}")]
assert not missing_files, f"Could not find all assets: {missing_files}"


setup(
    name="mlfab",
    version=version,
    description="A collection of core machine learning tools",
    author="Benjamin Bolte",
    url="https://github.com/dpshai/mlfab",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    tests_require=DEV_REQUIREMENTS,
    extras_require={"dev": DEV_REQUIREMENTS},
    entry_points={
        "console_scripts": [
            "create-mlfab-project=mlfab.scripts.create_mlfab_project.__main__:main",
        ],
    },
    package_data={
        "mlfab": [
            "py.typed",
            "requirements*.txt",
        ]
        + create_mlfab_project_assets,
    },
)
