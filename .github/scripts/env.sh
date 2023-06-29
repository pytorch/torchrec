#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

conda install -n test -y cuda -c "nvidia/label/cuda-11.8.0"

echo "before LD library path"
echo $LD_LIBRARY_PATH
echo "after LD library path"

echo "before ldconfig"
{
ldconfig -p && output=$(ldconfig -p); echo "$output"
} || {
echo "ldconfig failed"
}
echo "after ldconfig"

echo "before nvidia-smi"
{
sudo nvidia-smi
} || {
nvidia-smi
} || {
echo "nvidia-smi failed"
}
echo "after nvidia-smi"

echo "before cuda"
nvcc --version
echo "after cuda"
