#!/usr/bin/env bash
set -ex
eval "$(conda shell.bash hook)"

conda create -y -n ${ENV_NAME} python=${DESIRED_PYTHON} numpy
conda activate ${ENV_NAME}
export CONDA_CHANNEL="pytorch"
export PIP_DOWNLOAD_URL="https://download.pytorch.org/whl/cpu"
export TORCHREC_PIP_PREFIX=""

if [[ ${CHANNEL} = 'nightly' ]]; then
    export TORCHREC_PIP_PREFIX="--pre"
    export PIP_DOWNLOAD_URL="https://download.pytorch.org/whl/nightly/cpu"
    export CONDA_CHANNEL="pytorch-nightly"
elif [[ ${CHANNEL} = 'test' ]]; then
    export PIP_DOWNLOAD_URL="https://download.pytorch.org/whl/test/cpu"
    export CONDA_CHANNEL="pytorch-test"
fi

if [[ ${PACKAGE_TYPE} = 'conda' ]]; then
    conda install -y torchrec pytorch -c ${CONDA_CHANNEL}
else
    pip install ${TORCHREC_PIP_PREFIX} torchrec torch --extra-index-url ${PIP_DOWNLOAD_URL}
fi

python test_installation.py -- --cpu_only
