#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
import re
import sys
from datetime import date
from typing import List

from setuptools import find_packages, setup


def get_version():
    # get version string from version.py
    # TODO: ideally the version.py should be generated when setup is run
    version_file = os.path.join(os.path.dirname(__file__), "version.py")
    version_regex = r"__version__ = ['\"]([^'\"]*)['\"]"
    with open(version_file, "r") as f:
        version = re.search(version_regex, f.read(), re.M).group(1)
        return version


def get_nightly_version():
    today = date.today()
    return f"{today.year}.{today.month}.{today.day}"


def get_channel():
    # Channel typically takes on the following values:
    # - NIGHTLY: for nightly published binaries
    # - TEST: for binaries build from release candidate branches
    return os.getenv("CHANNEL")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec setup")
    parser.add_argument(
        "--package_name",
        type=str,
        default="torchrec",
        help="the name of this output wheel",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="override version",
    )
    return parser.parse_known_args(argv)


def main(argv: List[str]) -> None:
    args, unknown = parse_args(argv)

    # Set up package name and version
    channel = get_channel()
    name = args.package_name

    with open(
        os.path.join(os.path.dirname(__file__), "README.MD"), encoding="utf8"
    ) as f:
        readme = f.read()
    with open(
        os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="utf8"
    ) as f:
        reqs = f.read()
        install_requires = reqs.strip().split("\n")

    version = args.version
    if version is None:
        version = get_nightly_version() if channel == "nightly" else get_version()

    if channel != "nightly":
        if "fbgemm-gpu-nightly" in install_requires:
            install_requires.remove("fbgemm-gpu-nightly")
        install_requires.append("fbgemm-gpu")

    print(f"-- {name} building version: {version}")

    packages = find_packages(
        exclude=(
            "*tests",
            "*test",
            "examples",
            "*examples.*",
            "*benchmarks",
            "*build",
            "*rfc",
        )
    )
    sys.argv = [sys.argv[0]] + unknown

    setup(
        # Metadata
        name=name,
        version=version,
        author="TorchRec Team",
        author_email="packages@pytorch.org",
        description="Pytorch domain library for recommendation systems",
        long_description=readme,
        long_description_content_type="text/markdown",
        url="https://github.com/pytorch/torchrec",
        license="BSD-3",
        keywords=["pytorch", "recommendation systems", "sharding"],
        python_requires=">=3.7",
        install_requires=install_requires,
        packages=packages,
        zip_safe=False,
        # PyPI package information.
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main(sys.argv[1:])
