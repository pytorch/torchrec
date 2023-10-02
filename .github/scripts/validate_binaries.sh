#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


export PYTORCH_CUDA_PKG=""

conda create -y -n build_binary python="${MATRIX_PYTHON_VERSION}"

conda run -n build_binary python --version

# Install pytorch, torchrec and fbgemm as per
# installation instructions on following page
# https://github.com/pytorch/torchrec#installations
# switch back to conda once torch nightly is fixed
# if [[ ${MATRIX_GPU_ARCH_TYPE} = 'cuda' ]]; then
#     export PYTORCH_CUDA_PKG="pytorch-cuda=${MATRIX_GPU_ARCH_VERSION}"
# fi

if [[ ${MATRIX_GPU_ARCH_TYPE} = 'rocm' ]]; then
    echo "We don't support rocm"
    exit 0
fi

if [[ ${MATRIX_GPU_ARCH_TYPE} = 'cuda' ]]; then
    if [[ ${MATRIX_GPU_ARCH_VERSION} = '11.8' ]]; then
        export CUDA_VERSION="cu118"
    else
        export CUDA_VERSION="cu121"
    fi
else
    export CUDA_VERSION="cpu"
fi

if [[ ${MATRIX_GPU_ARCH_TYPE} = 'cuda' && ${MATRIX_GPU_ARCH_VERSION} = '11.8' ]]; then
    conda install -n build_binary pytorch pytorch-cuda=11.8 -c pytorch-test -c nvidia -y
    conda run -n build_binary pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/test/cu118
    conda run -n build_binary pip install torchmetrics==1.0.3
    conda run -n build_binary pip install --pre torchrec --index-url https://download.pytorch.org/whl/test/cu118
fi 

if [[ ${MATRIX_GPU_ARCH_TYPE} = 'cuda' && ${MATRIX_GPU_ARCH_VERSION} = '12.1' ]]; then
    conda install -n build_binary pytorch pytorch-cuda=12.1 -c pytorch-test -c nvidia -y
    conda run -n build_binary pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/test/cu121
    conda run -n build_binary pip install torchmetrics==1.0.3
    conda run -n build_binary pip install --pre torchrec --index-url https://download.pytorch.org/whl/test/cu118
    exit 0
fi 

if [[ ${MATRIX_GPU_ARCH_TYPE} = 'cpu' ]]; then
    conda install -n build_binary pytorch cpuonly -c pytorch-test -y 
    conda run -n build_binary pip install fbgemm-gpu-cpu
    conda run -n build_binary pip install torchmetrics==1.0.3
    conda run -n build_binary pip install --pre torchrec --index-url https://download.pytorch.org/whl/test/cpu
    conda run -n build_binary pip uninstall fbgemm-gpu-cpu -y
    conda run -n build_binary pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/test/cpu
fi 

# Run small import test
conda run -n build_binary python -c "import torch; import fbgemm_gpu; import torchrec"

# check directory
ls -R

# Finally run smoke test
conda run -n build_binary pip install torchx-nightly iopath
if [[ ${MATRIX_GPU_ARCH_TYPE} = 'cuda' ]]; then
    conda run -n build_binary torchx run -s local_cwd dist.ddp -j 1 --gpu 2 --script test_installation.py
else
    conda run -n build_binary torchx run -s local_cwd dist.ddp -j 1 --script test_installation.py -- --cpu_only
fi
