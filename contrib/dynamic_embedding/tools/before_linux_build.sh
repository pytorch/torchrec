#!/usr/bin/env bash
set -xe

distro=rhel7
arch=x86_64
CUDA_VERSION="${CUDA_VERSION:-11.6}"

CUDA_MAJOR_VERSION=$(echo "${CUDA_VERSION}" | tr '.' ' ' | awk '{print $1}')
CUDA_MINOR_VERSION=$(echo "${CUDA_VERSION}" | tr '.' ' ' | awk '{print $2}')

yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-$distro.repo
yum install -y \
        cuda-toolkit-"${CUDA_MAJOR_VERSION}"-"${CUDA_MINOR_VERSION}" \
        libcudnn8-devel
ln -s cuda-"${CUDA_MAJOR_VERSION}"."${CUDA_MINOR_VERSION}" /usr/local/cuda

pipx install cmake
pipx install ninja
python -m pip install scikit-build
python -m pip install --pre torch --extra-index-url \
  https://download.pytorch.org/whl/nightly/cu"${CUDA_MAJOR_VERSION}""${CUDA_MINOR_VERSION}"
