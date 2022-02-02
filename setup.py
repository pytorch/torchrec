#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import random
import re
import sys
from datetime import date
from subprocess import check_output
from typing import List

from setuptools import setup, find_packages


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


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec setup")
    parser.add_argument(
        "--skip_fbgemm",
        type=bool,
        default=False,
        help="if we need to skip the fbgemm_gpu installation",
    )
    parser.add_argument(
        "--pacakge_name",
        type=str,
        default="torchrec",
        help="the name of this output wheel",
    )
    parser.add_argument(
        "--TORCH_CUDA_ARCH_LIST",
        type=str,
        default="7.0;8.0",
        help="the arch list of the torch cuda, check here for more detail: https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu",
    )
    return parser.parse_known_args(argv)


def main(argv: List[str]) -> None:
    args, unknown = parse_args(argv)
    print("args: ", args)
    print("unknown: ", unknown)
    if args.skip_fbgemm:
        print("Skipping fbgemm_gpu installation")
    else:
        print("Installing fbgemm_gpu")
        print("TORCH_CUDA_ARCH_LIST: ", args.TORCH_CUDA_ARCH_LIST)
        my_env = os.environ.copy()
        cuda_arch_arg = f"-DTORCH_CUDA_ARCH_LIST={args.TORCH_CUDA_ARCH_LIST}"
        out = check_output(
            [sys.executable, "setup.py", "build", cuda_arch_arg],
            cwd="third_party/fbgemm/fbgemm_gpu",
            env=my_env,
        )
        print(out)

    name = args.pacakge_name
    print("name: ", name)
    is_nightly = "nightly" in name
    is_test = "test" in name

    with open(
        os.path.join(os.path.dirname(__file__), "README.MD"), encoding="utf8"
    ) as f:
        readme = f.read()

    with open(
        os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="utf8"
    ) as f:
        reqs = f.read()

    version = get_nightly_version() if is_nightly else get_version()
    if is_test:
        version = (f"0.0.{random.randint(0, 1000)}",)
    print(f"-- {name} building version: {version}")
    # the path to find all the packages
    fbgemm_install_base = glob.glob(
        "third_party/fbgemm/fbgemm_gpu/_skbuild/*/cmake-install"
    )[0]
    sys.argv = [sys.argv[0]] + unknown
    print("sys.argv", sys.argv)

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
        install_requires=reqs.strip().split("\n"),
        packages=find_packages(exclude=("*tests",))
        + find_packages(fbgemm_install_base),
        package_dir={
            "torchrec": "torchrec",
            # to include the fbgemm_gpu.so
            "fbgemm_gpu": glob.glob(
                "third_party/fbgemm/fbgemm_gpu/_skbuild/*/cmake-install/fbgemm_gpu"
            )[0],
        },
        zip_safe=False,
        package_data={"fbgemm_gpu": ["fbgemm_gpu_py.so"]},
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
    print(sys.argv)
    main(sys.argv[1:])
