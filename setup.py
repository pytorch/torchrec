#!/usr/bin/env python3

from setuptools import setup, find_packages

# Minimal setup configuration.
setup(
    name="torchrec",
    packages=find_packages(exclude=("*tests",)),
)
