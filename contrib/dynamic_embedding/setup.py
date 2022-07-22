import os.path

import torch

from skbuild import setup

setup(
    name="torchrec_dynamic_embedding",
    package_dir={"": "src"},
    packages=["torchrec_dynamic_embedding"],
    cmake_args=[
        f"-DCMAKE_PREFIX_PATH={os.path.dirname(torch.__file__)}",
    ],
    cmake_install_dir="src",
    version="0.0.1",
)
