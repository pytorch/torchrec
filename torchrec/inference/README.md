# TorchRec Inference Library (Experimental Release)

TorchRec Inference is a C++ library that supports multi-gpu inference.

## Install from source

C++ 17 is a requirement.

1. Install pytorch from source
The inference library uses torch deploy which is a library in pytorch that's only accessible if built from source. **Ensure that the pytorch version installed is compatible**
**with your CUDA toolkit and driver**. Run `nvidia-smi` to check driver version. Run `nvcc --version` or `conda list` to check toolkit version.
- Follow this link: https://github.com/pytorch/pytorch/tree/master/torch/csrc/deploy to initially install all the necessary CPython dependencies
- Follow this link: https://github.com/pytorch/pytorch/#from-source, to install pytorch from source. However, once at the installation step, run these commands:
```
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_DEPLOY=1
python setup.py develop
```

### Potential issues - Pytorch installation

If the installation in step #1 went smoothly for you, then skip to the next step. Otherwise, read this section to see if you ran into similar issues.

*Linker errors*
```
[2055/2095] Linking CXX executable bin/interactive_embedded_interpreter
FAILED: bin/interactive_embedded_interpreter
...
/usr/bin/ld: cannot find -lncursesw
/usr/bin/ld: cannot find -lpanelw
collect2: error: ld returned 1 exit status
[2091/2095] Building CXX object test_api/CMakeFiles/test_api.dir/modules.cpp.o
ninja: build stopped: subcommand failed.
```

Solve with a symlink: https://stackoverflow.com/questions/16710047/usr-bin-ld-cannot-find-lnameofthelibrary. For example:

```
ld -lncursesw --verbose
sudo find / -type f -name libncursesw.*
sudo ln -s /lib/x86_64-linux-gnu/libncursesw.so.5.9 /lib/x86_64-linux-gnu/libncursesw.so

ld -lpanelw --verbose
sudo find / -type f -name libpanelw.*
sudo ln -s usr/lib/x86_64-linux-gnu/libpanelw.so.5.9 usr/lib/x86_64-linux-gnu/libpanelw.so
```

2. Locate torch deploy library
After installing from source, you'll be able to find the torch deploy static library, `libtorch_deploy_internal.a`, in your pytorch build directory
(e.g. `pytorch/build/lib/libtorch_deploy_internal.a`). Keep note of this location as we'll need it when we use cmake to install the torchrec inference library or point it to
an environment variable.
```
export `TORCH_DEPLOY_LIB_PATH`="pytorch/build/lib/libtorch_deploy_internal.a"
```

2. Install folly from source
The inference library relies on folly for performance optimzations. Follow: https://github.com/facebook/folly#build-notes to install folly from source.
Ensure to provide a path to the `--scratch-path` option (e.g. `~/folly-build/`). Folly will be installed at the location of the scratch path. Folly will also install fmt
(along with other libraries) for you. Make note of the path to these libraries as they'll be necessary for the torchrec inference library's cmake build.

```
# Replace ~/folly-build/ with the location of your folly installation.
# Replace fmt-dGmDTkdcPS1pyvm65J7UcKzxzLonWCKaaxWmgYpScUw with the name of your fmt directory in ~/folly-build/installed

export FOLLY_INSTALL_LOCATION="~/folly-build/installed/folly/lib/cmake/folly"
export FMT_INSTALL_LOCATION="~/folly-build/installed/fmt-dGmDTkdcPS1pyvm65J7UcKzxzLonWCKaaxWmgYpScUw/lib/cmake/fmt"
```

3. Install the inference library
```
~$ git clone https://github.com/pytorch/torchrec.git
~$ cd torchrec/torchrec/inference/

~/torchrec/torchrec/inference$ mkdir build
~/torchrec/torchrec/inference$ cmake -S . -B build/ \
-DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)');$FOLLY_INSTALL_LOCATION;$FMT_INSTALL_LOCATION" \
-DTORCH_DEPLOY_LIB_PATH="$TORCH_DEPLOY_LIB_PATH"

~/torchrec/torchrec/inference$ cd build
~/torchrec/torchrec/inference/build$ make -j
```

The above commands should create the inference library, `libinference.so`, which you can use to link to executables or other libraries. An example executable was also created
to ensure that a link can be performed. Run the example in the build directory. If there are issues with glog during the build phase, see the issues section below.

```
~/torchrec/torchrec/inference$ ./example
0
```

### Potential issues - linking to libinference.so
Skip this section if you had no issues running the example.

**libdouble-conversion**
```
~/torchrec/torchrec/inference/build$ ./example
./example: error while loading shared libraries: libdouble-conversion.so.3: cannot open shared object file: No such file or directory
```

Solve by finding the installation of libdouble-conversion, provided by folly. Create a symlink to `libdouble-conversion.so.3` and add the install path to `$LD_LIBRARY_PATH`:

```
~/torchrec/torchrec/inference/build$ sudo find / -type f -name *libdouble*
~/folly-build/installed/double-conversion-skGL6pOaPHjtDwdXY-agzdwT1gvTXP0bD-7P4gKJD9I/lib/libdouble-conversion.so.3.1.4

~/torchrec/torchrec/inference/build$ sudo ln -s ~/folly-build/installed/double-conversion-skGL6pOaPHjtDwdXY-agzdwT1gvTXP0bD-7P4gKJD9I/lib/libdouble-conversion.so.3.1.4 \
libdouble-conversion.so.3

~/torchrec/torchrec/inference/build$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/folly-build/installed/double-conversion-skGL6pOaPHjtDwdXY-agzdwT1gvTXP0bD-7P4gKJD9I/lib
```


**Two installations of glog**
```
~/torchrec/torchrec/inference/build$ ./example
ERROR: flag 'logtostderr' was defined more than once (in files '/home/shabab/glog/src/logging.cc' and
'/home/shabab/folly-build/extracted/glog-v0.4.0.tar.gz/glog-0.4.0/src/logging.cc').
```
The above issue, along with other a host of others during building, can potentially occur if libinference.so is pointing to two different versions of glog (if one was
previously installed in your system). The issue can be solved by using the version provided by folly.

To use the version provided by folly, add the glog install path (in your folly-build directory) to your LD_LIBRARY_PATH much like in the libdouble-conversion issue above.
