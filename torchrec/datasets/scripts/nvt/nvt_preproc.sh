#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

INPUT_PATH="$1"
BASE_OUTPUT_PATH="$2"

python 01_nvt_preproc.py -i "$INPUT_PATH" -o "$BASE_OUTPUT_PATH"
python 02_nvt_preproc.py -b "$BASE_OUTPUT_PATH"
python 03_nvt_preproc.py -b "$BASE_OUTPUT_PATH"
