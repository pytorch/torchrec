# FBGEMM_GPU

FBGEMM_GPU (FBGEMM GPU kernel library) is a collection of
high-performance CUDA GPU operator library for GPU training and inference.

The library provides efficient table batched embedding bag,
data layout transformation, and quantization supports.


## Examples

The tests (in test folder) and benchmarks (in bench folder) are some great
examples of using FBGEMM_GPU.

## Build Notes
FBGEMM_GPU uses the standard CMAKE-based build flow
and [PyTorch TorchScript extension with custom C++ operator][0] build flow.

### Dependencies
FBGEMM_GPU requires nvcc and a Nvidia GPU with
compute capability of 3.5+.

+ ###### CUB
For the [CUB][1] build time dependency, if you are using conda, you can continue with
```
conda install -c bottler nvidiacub
```
Otherwise download the CUB library from https://github.com/NVIDIA/cub/releases and unpack it to a folder of your choice. Define the environment variable CUB_DIR before building and point it to the directory that contains CMakeLists.txt for CUB. For example on Linux/Mac,

```
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_DIR=$PWD/cub-1.10.0
```

+ ###### googletest
[googletest][2] is required to build and run FBGEMM_GPU's tests. **googletest is not
required** if you don't want to run FBGEMM_GPU tests. By default, building of tests
is **on**. Turn it off by setting FBGEMMGPU\_BUILD\_TESTS to off.


+ ###### PyTorch, Jinja2
[PyTorch][3] and [Jinja2][4] are **required** to build and run the table
batched embedding bag operator. One thing to note is that the implementation
of this op relies on the latest version of PyTorch (1.8+), so it requires the
installation with PyTorch Nightly:
```
conda uninstall pytorch
# update with the corresponding CUDA version
conda install pytorch cudatoolkit=9.2 -c pytorch-nightly
conda install jinja2
```

You can download [googletest][2] and set
GOOGLETEST\_SOURCE\_DIR respectively for
cmake to find these libraries. If any of these variables is not set, cmake will
build the git submodules found in the third\_party directory.

General build instructions are as follows:

```
git clone --recursive https://github.com/pytorch/FBGEMM.git
cd FBGEMM/fbgemm_gpu
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
# configure the NVCC and CUB path
export CUDACXX=/usr/local/cuda/bin/nvcc
# if using CUDA 10 or earliers set the location to the CUB installation directory
export CUB_DIR=${CUB_DIR}
# in fbgemm_gpu folder
# build the table batched embedding bag op
cd ..
python setup.py build develop
```

FBGEMM_GPU also supports a CPU-only (no CUDA dependencies) build if so desired.  To
enable the CPU-only build add the --cpu_only option to the python setup command.

```
python setup.py build develop --cpu_only
```

## Running  FBGEMM_GPU

To run the tests or benchmarks after building FBGEMM_GPU (if tests or benchmarks
are built), use the following command:
```
# run the tests and benchmarks of table batched embedding bag op,
# data layout transform op, quantized ops, etc.
cd ..
python test/split_table_batched_embeddings_test.py
python test/quantize_ops_test.py
python test/sparse_ops_test.py
python test/split_embedding_inference_converter_test.py
python bench/split_table_batched_embeddings_benchmark.py
```

## How FBGEMM_GPU works
For a high-level overview, design philosophy and brief descriptions of various
parts of FBGEMM_GPU please see our Wiki (work in progress).

## Full documentation
We have extensively used comments in our source files. The best and up-do-date
documentation is available in the source files.

## Join the FBGEMM community
See the [`CONTRIBUTING`](../CONTRIBUTING.md) file for how to help out.

## License
FBGEMM is BSD licensed, as found in the [`LICENSE`](../LICENSE) file.

[0]:https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html
[1]:https://github.com/NVIDIA/cub
[2]:https://github.com/google/googletest
[3]:https://github.com/pytorch/pytorch
[4]:https://jinja.palletsprojects.com/en/2.11.x/
