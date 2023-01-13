#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


export PYTORCH_CUDA_PKG=""

# Install pytorch, torchrec and fbgemm as per
# installation instructions on following page
# https://github.com/pytorch/torchrec#installations
if [[ ${MATRIX_GPU_ARCH_TYPE} = 'cuda' ]]; then
    export PYTORCH_CUDA_PKG="pytorch-cuda=${MATRIX_GPU_ARCH_VERSION}"
fi

if [[ ${MATRIX_CHANNEL} = 'nightly' ]]; then
    # shellcheck disable=SC2086
    conda install -y pytorch ${PYTORCH_CUDA_PKG} -c pytorch-nightly -c nvidia
    pip install torchrec_nightly
else
    # shellcheck disable=SC2086
    conda install -y pytorch ${PYTORCH_CUDA_PKG} -c pytorch -c nvidia
    pip install torchrec
fi

if [[ ${MATRIX_GPU_ARCH_TYPE} = 'cpu' || ${MATRIX_GPU_ARCH_TYPE} = 'rocm' ]]; then
    if [[ ${MATRIX_CHANNEL} = 'nightly' ]]; then
        pip uninstall fbgemm-gpu-nightly -y
        pip install fbgemm-gpu-nightly-cpu
    else
        pip uninstall fbgemm-gpu -y
        pip install fbgemm-gpu-cpu
    fi
fi

# Run small import test
python -c "import torch; import fbgemm_gpu; import torchrec"

# Finally run smoke test
if [[ ${MATRIX_GPU_ARCH_TYPE} = 'cuda' ]]; then
    torchx run -s local_cwd dist.ddp -j 1 --gpu 2 --script test_installation.py
else
    torchx run -s local_cwd dist.ddp -j 1 --script test_installation.py -- --cpu_only
fi
