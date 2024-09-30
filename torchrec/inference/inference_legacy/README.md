# TorchRec Inference Library (**Experimental** Release)

## Overview
TorchRec Inference is a C++ library that supports **multi-gpu inference**. The Torchrec library is used to shard models written and packaged in Python via [torch.package](https://pytorch.org/docs/stable/package.html) (an alternative to TorchScript). The [torch.deploy](https://pytorch.org/docs/stable/deploy.html) library is used to serve inference from C++ by launching multiple Python interpreters carrying the packaged model, thus subverting the GIL.

Follow the instructions below to package a DLRM model in Python, run a C++ inference server with the model on a GPU and send requests to said server via a python client.

## Example

C++ 17 is a requirement.

<br>

### **1. Install Dependencies**

Follow the instructions at: https://github.com/pytorch/pytorch/blob/master/docs/source/deploy.rst to ensure torch::deploy
is working in your environment. Use the Dockerfile in the docker directory to install all dependencies. Run it via:

```
sudo nvidia-docker build -t torchrec .
sudo nvidia-docker run -it torchrec:latest
```

### **2. Set variables**

Replace these variables with the relevant paths in your system. Check `CMakeLists.txt` and `server.cpp` to see how they're used throughout the build and runtime.

```
# provide the cmake prefix path of pytorch, folly, and fmt.
# fmt and boost are pulled from folly's installation in this example.
export FOLLY_CMAKE_DIR="~/folly-build/installed/folly/lib/cmake/folly"
export FMT_CMAKE_DIR="~/folly-build/fmt-WuK9LOk2P03KJKCt1so6VofoXa4qtyD5kQk6cZMphjg/lib64/cmake/fmt"
export BOOST_CMAKE_DIR="~/folly-build/installed/boost-4M2ZnvEM4UWTqpsEJRQTB4oejmX3LmgYC9pcBiuVlmA/lib/cmake/Boost-1.78.0"

#  provide fmt from pytorch for torch deploy
export PYTORCH_FMT_INCLUDE_PATH="~/pytorch/third_party/fmt/include/"
export PYTORCH_LIB_FMT="~/pytorch/build/lib/libfmt.a"

#  provide necessary info to link to torch deploy
export DEPLOY_INTERPRETER_PATH="/pytorch/build/torch/csrc/deploy"
export DEPLOY_SRC_PATH="~/pytorch/torch/csrc/deploy"

#  provide common.cmake from grpc/examples, makes linking to grpc easier
export GRPC_COMMON_CMAKE_PATH="~/grpc/examples/cpp/cmake"
export GRPC_HEADER_INCLUDE_PATH="~/.local/include/"

# provide libfbgemm_gpu_py.so to enable fbgemm_gpu c++ operators
export FBGEMM_LIB="~/anaconda3/envs/inference/lib/python3.8/site-packages/fbgemm_gpu-0.1.0-py3.8-linux-x86_64.egg/fbgemm_gpu/libfbgemm_gpu_py.so"

# provide path to python packages for torch deploy runtime
export PYTHON_PACKAGES_PATH="~/anaconda3/envs/inference/lib/python3.8/site-packages/"
```

Update `$LD_LIBRARY_PATH` and `$LIBRARY_PATH` to enable linker to locate libraries.

```
# double-conversion, fmt and gflags are pulled from folly's installation in this example
export DOUBLE_CONVERSION_LIB_PATH="~/folly-build/installed/double-conversion-skGL6pOaPHjtDwdXY-agzdwT1gvTXP0bD-7P4gKJD9I/lib"
export FMT_LIB_PATH="~/folly-build/fmt-WuK9LOk2P03KJKCt1so6VofoXa4qtyD5kQk6cZMphjg/lib64/"
export GFLAGS_LIB_PATH="~/folly-build/installed/gflags-KheHQBqQ3_iL3yJBFwWe5M5f8Syd-LKAX352cxkhQMc/lib"
export PYTORCH_LIB_PATH="~/pytorch/build/lib/"

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$DOUBLE_CONVERSION_LIB_PATH:$FMT_LIB_PATH:$GFLAGS_LIB_PATH:$PYTORCH_LIB_PATH"
export LIBRARY_PATH="$PYTORCH_LIB_PATH"
```

### **3. Package DLRM model**

The `PredictFactoryPackager` class in `model_packager.py` can be used to implement your own packager class. Implement
`set_extern_modules` to specify the dependencies of your predict module that should be accessed from the system and
implement `set_mocked_modules` to specify dependencies that should be mocked (necessary to import but not use). Read
more about extern and mock modules in the `torch.package` documentation: https://pytorch.org/docs/stable/package.html.

`/torchrec/examples/inference_legacy/dlrm_package.py` provides an example of packaging a module for inference (`/torchrec/examples/inference_legacy/dlrm_predict.py`).
`DLRMPredictModule` is packaged for inference in the following example.

```
git clone https://github.com/pytorch/torchrec.git

cd ~/torchrec/examples/inference_legacy/
python dlrm_packager.py --output_path /tmp/model_package.zip
```



### **4. Build inference library and example server**

Generate protobuf C++ and Python code from protobuf

```
cd ~/torchrec/inference/
mkdir -p gen/torchrec/inference

# C++ (server)
protoc -I protos/ --grpc_out=gen/torchrec/inference --plugin=protoc-gen-grpc=/home/shabab/.local/bin/grpc_cpp_plugin protos/predictor.proto

protoc -I protos/ --cpp_out=gen/torchrec/inference protos/predictor.proto


# Python (client)
python -m grpc_tools.protoc -I protos --python_out=gen/torchrec/inference --grpc_python_out=gen/torchrec/inference protos/predictor.proto
```


Build inference library and example server
```
cmake -S . -B build/ -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)');$FOLLY_CMAKE_DIR;$BOOST_CMAKE_DIR;$BOOST_CMAKE_DIR;"
-DPYTORCH_FMT_INCLUDE_PATH="$PYTORCH_FMT_INCLUDE_PATH" \
-DPYTORCH_LIB_FMT="$PYTORCH_LIB_FMT" \
-DDEPLOY_INTERPRETER_PATH="$DEPLOY_INTERPRETER_PATH" \
-DDEPLOY_SRC_PATH="$DEPLOY_SRC_PATH" \
-DGRPC_COMMON_CMAKE_PATH="$GRPC_COMMON_CMAKE_PATH" \ -DGRPC_HEADER_INCLUDE_PATH="$GRPC_HEADER_INCLUDE_PATH" \
-DFBGEMM_LIB="$FBGEMM_LIB"

cd build
make -j
```


### **5. Run server and client**

Run server. Update `CUDA_VISABLE_DEVICES` depending on the world size.
```
CUDA_VISABLE_DEVICES="0" ./server --package_path="/tmp/model_package.zip" --python_packages_path $PYTHON_PACKAGES_PATH
```

**output**

In the logs, a plan should be outputted by the Torchrec planner:

```
INFO:<torch_package_0>.torchrec.distributed.planner.stats:#                                   --- Planner Statistics ---                                    #
INFO:<torch_package_0>.torchrec.distributed.planner.stats:#                   --- Evalulated 1 proposal(s), found 1 possible plan(s) ---                    #
INFO:<torch_package_0>.torchrec.distributed.planner.stats:# ----------------------------------------------------------------------------------------------- #
INFO:<torch_package_0>.torchrec.distributed.planner.stats:#      Rank     HBM (GB)     DDR (GB)     Perf (ms)     Input (MB)     Output (MB)     Shards     #
INFO:<torch_package_0>.torchrec.distributed.planner.stats:#    ------   ----------   ----------   -----------   ------------   -------------   --------     #
INFO:<torch_package_0>.torchrec.distributed.planner.stats:#         0     0.2 (1%)     0.0 (0%)          0.08            0.1            1.02     TW: 26     #
INFO:<torch_package_0>.torchrec.distributed.planner.stats:#                                                                                                 #
INFO:<torch_package_0>.torchrec.distributed.planner.stats:# Input: MB/iteration, Output: MB/iteration, Shards: number of tables                             #
INFO:<torch_package_0>.torchrec.distributed.planner.stats:# HBM: est. peak memory usage for shards - parameter, comms, optimizer, and gradients             #
INFO:<torch_package_0>.torchrec.distributed.planner.stats:#                                                                                                 #
INFO:<torch_package_0>.torchrec.distributed.planner.stats:# Compute Kernels:                                                                                #
INFO:<torch_package_0>.torchrec.distributed.planner.stats:#   quant: 26                                                                             #
````

`nvidia-smi` output should also show allocation of the model onto the gpu:

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     86668      C   ./example                        1357MiB |
+-----------------------------------------------------------------------------+
```

Make a request to the server via the client:

```
python client.py
```

**output**

```
Response:  [0.13199582695960999, -0.1048036441206932, -0.06022112816572189, -0.08765199035406113, -0.12735335528850555, -0.1004377081990242, 0.05509107559919357, -0.10504599660634995, 0.1350800096988678, -0.09468207508325577, 0.24013587832450867, -0.09682435542345047, 0.0025023818016052246, -0.09786031395196915, -0.26396819949150085, -0.09670191258192062, 0.2691854238510132, -0.10246685892343521, -0.2019493579864502, -0.09904996305704117, 0.3894067406654358, ...]
```

<br>

## Planned work

- Provide benchmarks for torch deploy vs TorchScript and cpu, single gpu and multi-gpu inference
- In-code documentation
- Simplify installation process

<br>

## Potential issues and solutions

Skip this section if you had no issues with installation or running the example.

**Missing header files during pytorch installation**

If your environment is missing a speicfic set of header files such as `nvml.h` and `cuda_profiler_api.h`, the pytorch installation will fail with error messages similar to the code snippet below:

```
~/nvml_lib.h:13:10: fatal error: nvml.h: No such file or directory
 #include <nvml.h>
          ^~~~~~~~
compilation terminated.
[80/2643] Building CXX object third_party/ideep/mkl-dnn/third_party/oneDNN/src/cpu/CMakeFiles/dnnl_cpu.dir/cpu_convolution_list.cpp.o
ninja: build stopped: subcommand failed.
```

To get these header files, install `cudatoolkit-dev`:
```
conda install -c conda-forge cudatoolkit-dev
```

Re-run the installation after this.

**libdouble-conversion missing**
```
~/torchrec/torchrec/inference/build$ ./example
./example: error while loading shared libraries: libdouble-conversion.so.3: cannot open shared object file: No such file or directory
```

If this issue persists even after adding double-conversion's path to $LD_LIBRARY_PATH (step 2) then solve by creating a symlink to `libdouble-conversion.so.3` with folly's installation of double-conversion:

```
sudo ln -s ~/folly-build/installed/double-conversion-skGL6pOaPHjtDwdXY-agzdwT1gvTXP0bD-7P4gKJD9I/lib/libdouble-conversion.so.3.1.4 \
libdouble-conversion.so.3
```

**Two installations of glog**
```
~/torchrec/torchrec/inference/build$ ./example
ERROR: flag 'logtostderr' was defined more than once (in files '/home/shabab/glog/src/logging.cc' and
'/home/shabab/folly-build/extracted/glog-v0.4.0.tar.gz/glog-0.4.0/src/logging.cc').
```
The above issue, along with a host of others during building, can potentially occur if libinference is pointing to two different versions of glog (if one was
previously installed in your system). You can find this out by running `ldd` on your libinference shared object within the build path. The issue can be solved by using the glog version provided by folly.

To use the glog version provided by folly, add the glog install path (in your folly-build directory) to your LD_LIBRARY_PATH much like step 2.

**Undefined symbols with std::string or cxx11**

If you get undefined symbol errors and the errors mention `std::string` or `cxx11`, it's likely
that your dependencies were compiled with different ABI values. Re-compile your dependencies
and ensure they all have the same value for `_GLIBCXX_USE_CXX11_ABI` in their build.

The ABI value of pytorch can be checked via:

```
import torch
torch._C._GLIBCXX_USE_CXX11_ABI
```
