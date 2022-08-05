#!/usr/bin/env bash
set -xe

distro=rhel7
arch=x86_64

yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-$distro.repo
yum install -y \
        cuda-toolkit-11-6 \
        libcudnn8-devel
ln -s cuda-11.6 /usr/local/cuda

pipx install cmake
pipx install ninja
python -m pip install scikit-build
python -m pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu116
