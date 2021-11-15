# FBGEMM

[![FBGEMMCI](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemmci.yml/badge.svg)](https://github.com/pytorch/FBGEMM/actions/workflows/fbgemmci.yml)
[![CircleCI](https://circleci.com/gh/pytorch/FBGEMM.svg?style=shield)](https://circleci.com/gh/pytorch/FBGEMM)

FBGEMM (Facebook GEneral Matrix Multiplication) is a low-precision,
high-performance matrix-matrix multiplications and convolution library for
server-side inference.

The library provides efficient low-precision general matrix multiplication for
small batch sizes and support for accuracy-loss minimizing techniques such as
row-wise quantization and outlier-aware quantization. FBGEMM also exploits
fusion opportunities in order to overcome the unique challenges of matrix
multiplication at lower precision with bandwidth-bound operations.

FBGEMM is used as a backend of Caffe2 and PyTorch quantized operators for x86 machines:
* Caffe2: https://github.com/pytorch/pytorch/tree/master/caffe2/quantization/server
* PyTorch: https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native/quantized/cpu

## What's New?
* [New Features and Recent Improvements](https://github.com/pytorch/FBGEMM/wiki/Recent-feature-additions-and-improvements-in-FBGEMM) (January, 2020)

## Examples

The tests (in test folder) and benchmarks (in bench folder) are some great
examples of using FBGEMM. For instance, SpMDMTest test in
test/PackedRequantizeAcc16Test.cc shows how to combine row offset calculations
with packing of A (PackAWithRowOffset), how to pack B matrix (PackBMatrix) and
construct output pipeline (sparse\_matrix\*dense\_matrix --> requantization -->
nop) fused with inner GEMM macro kernel.

## Build Notes
FBGEMM uses the standard CMAKE-based build flow.

### Dependencies
FBGEMM requires gcc 5+ and a CPU with support for avx2 instruction set or
higher. It's been tested on Mac OS X and Linux.

+ ###### asmjit
With inner kernels, FBGEMM takes a “one size doesn't fit all” approach, so the
implementation dynamically generates efficient matrix-shape specific vectorized
code using a third-party library called [asmjit][1]. **asmjit is required** to
build FBGEMM.

+ ###### cpuinfo
FBGEMM detects CPU instruction set support at runtime using cpuinfo library and
dispatches optimized kernels for the detected instruction set. Therefore,
**cpuinfo is required** to detect CPU type.

+ ###### googletest
googletest is required to build and run FBGEMM's tests. **googletest is not
required** if you don't want to run FBGEMM tests. By default, building of tests
is **on**. Turn it off by setting FBGEMM\_BUILD\_TESTS to off.

You can download [asmjit][1], [cpuinfo][2], [googletest][3] and set
ASMJIT\_SRC\_DIR, CPUINFO\_SRC\_DIR, GOOGLETEST\_SOURCE\_DIR respectively for
cmake to find these libraries. If any of these variables is not set, cmake will
build the git submodules found in the third\_party directory.

FBGEMM, in general, does not have any dependency on Intel MKL. However, for
performance comparison, some benchmarks use MKL functions. If MKL is found or
MKL path is provided with INTEL\_MKL\_DIR benchmarks are built with MKL and
performance numbers are reported for MKL functions as well. However, if MKL is
not found, the benchmarks are not built.

General build instructions are as follows:

```
git clone --recursive https://github.com/pytorch/FBGEMM.git
cd FBGEMM
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
mkdir build && cd build
cmake ..
make
```

To run the tests after building FBGEMM (if tests are built), use the following
command:
```
make test
```

## Installing  FBGEMM
```
make install
```

## How FBGEMM works
For a high-level overview, design philosophy and brief descriptions of various
parts of FBGEMM please see [our blog][4].

## Full documentation
We have extensively used comments in our source files. The best and up-do-date
documentation is available in the source files.

You can also turn on the option to generate the documentation (using [Doxygen][5]
and [Sphinx][6] by setting FBGEMM\_BUILD\_DOCS to ON, and then follow the above
cmake build process.

## Citation
For those looking for the appropriate article to cite regarding FBGEMM, we
recommend citing our
[paper](https://arxiv.org/pdf/2101.05615.pdf):

```
@article{fbgemm,
  title={FBGEMM: Enabling High-Performance Low-Precision Deep Learning Inference},
  author={Khudia, Daya and Huang, Jianyu and Basu, Protonu and Deng, Summer and Liu, Haixin and Park, Jongsoo and Smelyanskiy, Mikhail},
  journal={arXiv preprint arXiv:2101.05615},
  year={2021}
}
```

## Join the FBGEMM community
See the [`CONTRIBUTING`](CONTRIBUTING.md) file for how to help out.

## License
FBGEMM is BSD licensed, as found in the [`LICENSE`](LICENSE) file.


[1]:https://github.com/asmjit/asmjit
[2]:https://github.com/pytorch/cpuinfo
[3]:https://github.com/google/googletest
[4]:https://code.fb.com/ml-applications/fbgemm
[5]:https://www.doxygen.nl/index.html
[6]:https://www.sphinx-doc.org/en/master/
