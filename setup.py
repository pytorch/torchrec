#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import random
import re
import sys
from datetime import date

from setuptools import setup, find_packages
from subprocess import check_call

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


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        sys.exit("python >= 3.8 required for torchrec")

    if "--skip_fbgemm" in sys.argv:
        print("Skipping fbgemm_gpu installation")
        sys.argv.remove("--skip_fbgemm")

    else:
        print("Installing fbgemm_gpu")
        torchrec_dir = os.getcwd()
        os.chdir("third_party/fbgemm/fbgemm_gpu/")
        os.system(
            'CUDACXX=/usr/local/cuda-11.3/bin/nvcc TORCH_CUDA_ARCH_LIST="7.0;8.0" python setup.py build'
        )
        os.chdir(torchrec_dir)
        # check_call([sys.executable, "setup.py", "build"], cwd="third_party/fbgemm/fbgemm_gpu", env={'TORCH_CUDA_ARCH_LIST: "7.0;8.0'})

    name = "torchrec"
    NAME_ARG = "--override-name"
    if NAME_ARG in sys.argv:
        idx = sys.argv.index(NAME_ARG)
        name = sys.argv.pop(idx + 1)
        sys.argv.pop(idx)
    is_nightly = "nightly" in name
    is_test = "test" in name

    with open("README.MD", encoding="utf8") as f:
        readme = f.read()

    with open("requirements.txt") as f:
        reqs = f.read()

    version = get_nightly_version() if is_nightly else get_version()
    if is_test:
        version = (f"0.0.{random.randint(0, 1000)}",)
    print(f"-- {name} building version: {version}")
    fbgemm_install_base = glob.glob(
        "third_party/fbgemm/fbgemm_gpu/_skbuild/*/cmake-install"
    )[0]
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
        python_requires=">=3.8",
        install_requires=reqs.strip().split("\n"),
        packages=find_packages(exclude=("*tests",))
        + find_packages(fbgemm_install_base),
        package_dir={
            "torchrec": "torchrec",
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
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
