# TorchRec Inference Library (**Experimental** Release)

## Overview
TorchRec Inference is a C++ library that supports **gpu inference**. Previously, the TorchRec inference library was authored with torch.package and torch.deploy, which are old and deprecated. All the previous files live under the directory inference_legacy for reference.

TorchRec inference was reauthored with simplicity in mind, while also reflecting the current production environment for RecSys models, namely torch.fx for graph capturing/tracing and TorchScript for model inference in a C++ environment. The inference solution here is meant to serve as a simple reference and example, not a fully scaled out solution for production use cases. The current solution demonstrates converting the DLRM model in Python to TorchScript, running a C++ inference server with the model on a GPU, and sending requests to said server via a python client.

## Requirements

C++ 17 is a requirement. GCC version has to be >= 9, with initial testing done on GCC 9.

<br>

### **1. Install Dependencies**
1. [GRPC for C++][https://grpc.io/docs/languages/cpp/quickstart/] needs to be installed, with the resulting installation directory being `$HOME/.local`
2. Ensure that **the protobuf compiler (protoc) binary being used is from the GRPC installation above**. The protoc binary will live in `$HOME/.local/bin`, which may not match with the system protoc binary, can check with `which protoc`.
3. Install PyTorch, FBGEMM, and TorchRec (ideally in a virtual environment):
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics==1.0.3
pip install torchrec --index-url https://download.pytorch.org/whl/cu121
```


### **2. Set variables**

Replace these variables with the relevant paths in your system. Check `CMakeLists.txt` and `server.cpp` to see how they're used throughout the build and runtime.

```
# provide fbgemm_gpu_py.so to enable fbgemm_gpu c++ operators
find $HOME -name fbgemm_gpu_py.so

# Use path from correct virtual environment above and set environment variable $FBGEMM_LIB to it
export FBGEMM_LIB=""
```

### **3. Generate TorchScripted DLRM model**

Here, we generate the DLRM model in Torchscript and save it for model loading later on.

```
git clone https://github.com/pytorch/torchrec.git

cd ~/torchrec/torchrec/inference/
python3 dlrm_packager.py --output_path /tmp/model.pt
```


### **4. Build inference library and example server**

Generate Python code from protobuf for client and build the server.

```
# Python (client)
python -m grpc_tools.protoc -I protos --python_out=. --grpc_python_out=. protos/predictor.proto
```


Build server and C++ protobufs
```
cmake -S . -B build/ -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)');" -DFBGEMM_LIB="$FBGEMM_LIB"

cd build
make -j
```


### **5. Run server and client**

Start the server, loading in the model saved previously
```
./server /tmp/model.pt
```

**output**

In the logs, you should see:

```
Loading model...
Sanity Check with dummy inputs
 Model Forward Completed, Output: 0.489247
Server listening on 0.0.0.0:50051
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

In another terminal instance, make a request to the server via the client:

```
python client.py
```

**output**

```
Response:  [0.13199582695960999, -0.1048036441206932, -0.06022112816572189, -0.08765199035406113, -0.12735335528850555, -0.1004377081990242, 0.05509107559919357, -0.10504599660634995, 0.1350800096988678, -0.09468207508325577, 0.24013587832450867, -0.09682435542345047, 0.0025023818016052246, -0.09786031395196915, -0.26396819949150085, -0.09670191258192062, 0.2691854238510132, -0.10246685892343521, -0.2019493579864502, -0.09904996305704117, 0.3894067406654358, ...]
```
