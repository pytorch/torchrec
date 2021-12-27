#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from setuptools import setup, find_packages

if "--skip_fbgemm" in sys.argv:
    print("Skipping fbgemm_gpu installation")
    sys.argv.remove("--skip_fbgemm")

else:
    print("Installing fbgemm_gpu")
    torchrec_dir = os.getcwd()
    os.chdir("third_party/fbgemm/fbgemm_gpu/")
    os.system("python setup.py build develop")
    os.chdir(torchrec_dir)

# Minimal setup configuration.
setup(
    name="torchrec",
    packages=find_packages(exclude=("*tests",)),
)
