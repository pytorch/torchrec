#!/usr/bin/env python3

from setuptools import setup, find_packages

# Minimal setup configuration.
setup(
    name="torchrec",
    # Needed because torchrec's __init__.py is at the same level as setup.py. In other
    # PyTorch domain libraries, setup.py exists at the same level as the top level
    # package. TODO: Update torchrec folder organization on github prior to OSS release
    # to be consistent with other domain libraries.
    package_dir={"torchrec": "../torchrec"},
    packages=find_packages(
        "..", include=("torchrec", "torchrec.*"), exclude=("*tests",)
    ),
)
