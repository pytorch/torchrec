# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import shutil
import sysconfig
import sys

from codegen.embedding_backward_code_generator import emb_codegen
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

cpu_only_build = False
cur_dir = os.path.dirname(os.path.realpath(__file__))
cub_include_path = os.getenv("CUB_DIR", None)
if cub_include_path is None:
    print("CUDA CUB directory environment variable not set.  Using default CUB location.")
build_codegen_path = "build/codegen"
py_path = "python"

# Get the long description from the relevant file
with open(os.path.join(cur_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

extra_compile_args = sysconfig.get_config_var("CFLAGS").split()
extra_compile_args += ["-mavx2", "-mf16c", "-mfma", "-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl"]

OPTIMIZERS = [
    "adagrad",
    "adam",
    "approx_rowwise_adagrad",
    "approx_sgd",
    "lamb",
    "lars_sgd",
    "partial_rowwise_adam",
    "partial_rowwise_lamb",
    "rowwise_adagrad",
    "sgd",
]

cpp_asmjit_files = glob.glob("../third_party/asmjit/src/asmjit/*/*.cpp")

cpp_fbgemm_files = [
    "../src/EmbeddingSpMDMAvx2.cc",
    "../src/EmbeddingSpMDMAvx512.cc",
    "../src/EmbeddingSpMDM.cc",
    "../src/EmbeddingSpMDMNBit.cc",
    "../src/QuantUtils.cc",
    "../src/QuantUtilsAvx2.cc",
    "../src/RefImplementations.cc",
    "../src/RowWiseSparseAdagradFused.cc",
    "../src/SparseAdagrad.cc",
    "../src/Utils.cc",
]

cpp_cpu_output_files = (
    [
        "gen_embedding_forward_quantized_unweighted_codegen_cpu.cpp",
        "gen_embedding_forward_quantized_weighted_codegen_cpu.cpp",
        "gen_embedding_backward_dense_split_cpu.cpp",
    ]
    + [
        "gen_embedding_backward_split_{}_cpu.cpp".format(optimizer)
        for optimizer in OPTIMIZERS
    ]
    + [
        "gen_embedding_backward_{}_split_cpu.cpp".format(optimizer)
        for optimizer in OPTIMIZERS
    ]
)

cpp_cuda_output_files = (
    [
        "gen_embedding_forward_dense_weighted_codegen_cuda.cu",
        "gen_embedding_forward_dense_unweighted_codegen_cuda.cu",
        "gen_embedding_forward_quantized_split_unweighted_codegen_cuda.cu",
        "gen_embedding_forward_quantized_split_weighted_codegen_cuda.cu",
        "gen_embedding_forward_split_weighted_codegen_cuda.cu",
        "gen_embedding_forward_split_unweighted_codegen_cuda.cu",
        "gen_embedding_backward_split_indice_weights_codegen_cuda.cu",
        "gen_embedding_backward_dense_indice_weights_codegen_cuda.cu",
        "gen_embedding_backward_dense_split_unweighted_cuda.cu",
        "gen_embedding_backward_dense_split_weighted_cuda.cu",
    ]
    + [
        "gen_embedding_backward_{}_split_{}_cuda.cu".format(optimizer, weighted)
        for optimizer in OPTIMIZERS
        for weighted in [
            "weighted",
            "unweighted",
        ]
    ]
    + [
        "gen_embedding_backward_split_{}.cpp".format(optimizer)
        for optimizer in OPTIMIZERS
    ]
)

py_output_files = ["lookup_{}.py".format(optimizer) for optimizer in OPTIMIZERS]


def generate_jinja_files():
    abs_build_path = os.path.join(cur_dir, build_codegen_path)
    if not os.path.exists(abs_build_path):
        os.makedirs(abs_build_path)
    emb_codegen(install_dir=abs_build_path, is_fbcode=False)

    dst_python_path = os.path.join(cur_dir, py_path)
    if not os.path.exists(dst_python_path):
        os.makedirs(dst_python_path)
    for filename in py_output_files:
        shutil.copy2(os.path.join(abs_build_path, filename), dst_python_path)
    shutil.copy2(os.path.join(cur_dir, "codegen", "lookup_args.py"), dst_python_path)


class FBGEMM_GPU_BuildExtension(BuildExtension.with_options(no_python_abi_suffix=True)):
    def build_extension(self, ext):
        generate_jinja_files()
        super().build_extension(ext)

# Handle command line args before passing to main setup() method.
if "--cpu_only" in sys.argv:
    cpu_only_build = True
    sys.argv.remove("--cpu_only")

setup(
    name="fbgemm_gpu",
    install_requires=[
        "torch",
        "Jinja2",
        "click",
        "hypothesis",
    ],
    version="0.0.1",
    long_description=long_description,
    ext_modules=[
        CUDAExtension(
            name="fbgemm_gpu_py",
            sources=[
                os.path.join(cur_dir, build_codegen_path, "{}".format(f))
                for f in cpp_cuda_output_files + cpp_cpu_output_files
            ]
            + cpp_asmjit_files
            + cpp_fbgemm_files
            + [
                os.path.join(cur_dir, "codegen/embedding_forward_split_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_forward_quantized_host_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_forward_quantized_host.cpp"),
                os.path.join(cur_dir, "codegen/embedding_backward_dense_host_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_backward_dense_host.cpp"),
                os.path.join(cur_dir, "codegen/embedding_bounds_check_host.cpp"),
                os.path.join(cur_dir, "codegen/embedding_bounds_check_host_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_bounds_check.cu"),
                os.path.join(cur_dir, "src/split_embeddings_cache_cuda.cu"),
                os.path.join(cur_dir, "src/split_table_batched_embeddings.cpp"),
                os.path.join(cur_dir, "src/cumem_utils.cu"),
                os.path.join(cur_dir, "src/cumem_utils_host.cpp"),
                os.path.join(cur_dir, "src/quantize_ops_cpu.cpp"),
                os.path.join(cur_dir, "src/quantize_ops_gpu.cpp"),
                os.path.join(cur_dir, "src/sparse_ops_cpu.cpp"),
                os.path.join(cur_dir, "src/sparse_ops_gpu.cpp"),
                os.path.join(cur_dir, "src/sparse_ops.cu"),
                os.path.join(cur_dir, "src/input_combine_cpu.cpp"),
                os.path.join(cur_dir, "src/merge_pooled_embeddings_gpu.cpp"),
                os.path.join(cur_dir, "src/permute_pooled_embedding_ops.cu"),
                os.path.join(cur_dir, "src/permute_pooled_embedding_ops_gpu.cpp"),
                os.path.join(cur_dir, "src/layout_transform_ops_cpu.cpp"),
                os.path.join(cur_dir, "src/layout_transform_ops_gpu.cpp"),
                os.path.join(cur_dir, "src/layout_transform_ops.cu"),
            ],
            include_dirs=[
                cur_dir,
                os.path.join(cur_dir, "include"),
                os.path.join(cur_dir, "../include"),
                os.path.join(cur_dir, "../src"),
                os.path.join(cur_dir, "../third_party/asmjit/src"),
                os.path.join(cur_dir, "../third_party/asmjit/src/core"),
                os.path.join(cur_dir, "../third_party/asmjit/src/x86"),
                os.path.join(cur_dir, "../third_party/cpuinfo/include"),
                cub_include_path,
            ],
            extra_compile_args={"cxx": extra_compile_args + ["-DFBGEMM_GPU_WITH_CUDA"],
                                "nvcc": ["-U__CUDA_NO_HALF_CONVERSIONS__"]},
            libraries=["nvidia-ml"],
        ) if not cpu_only_build else
        CppExtension(
            name="fbgemm_gpu_py",
            sources=[
                os.path.join(cur_dir, build_codegen_path, "{}".format(f))
                for f in cpp_cpu_output_files
            ]
            + cpp_asmjit_files
            + cpp_fbgemm_files
            + [
                os.path.join(cur_dir, "codegen/embedding_forward_split_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_forward_quantized_host_cpu.cpp"),
                os.path.join(cur_dir, "codegen/embedding_backward_dense_host_cpu.cpp"),
            ],
            include_dirs=[
                cur_dir,
                os.path.join(cur_dir, "include"),
                os.path.join(cur_dir, "../include"),
                os.path.join(cur_dir, "../src"),
                os.path.join(cur_dir, "../third_party/asmjit/src"),
                os.path.join(cur_dir, "../third_party/asmjit/src/core"),
                os.path.join(cur_dir, "../third_party/asmjit/src/x86"),
                os.path.join(cur_dir, "../third_party/cpuinfo/include"),
            ],
            extra_compile_args={"cxx": extra_compile_args},
        )
    ],
    cmdclass={"build_ext": FBGEMM_GPU_BuildExtension},
)
