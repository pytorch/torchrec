FROM sahanp465/multipy:latest as multipy-docker

COPY . .

# Install sys deps
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install m4
RUN apt-get install vim

# folly
RUN git clone https://github.com/facebook/folly.git && \
    cd folly && \
    git reset --hard 8656af7fe21f6abd9f14c385e6bcc3d209f1db06 && \
    cd ..
RUN python3 ./build/fbcode_builder/getdeps.py --scratch-path /workspace/folly-build/ --allow-system-packages build

# grpc
RUN mkdir $HOME/.local
ENV MY_INSTALL_DIR "$HOME/.local"
ENV PATH "$MY_INSTALL_DIR/bin:$PATH"
RUN apt install -y build-essential autoconf libtool pkg-config && \
    git clone --recurse-submodules -b v1.46.3 --depth 1 --shallow-submodules https://github.com/grpc/grpc && \
    git reset --hard 53d69cc581c5b7305708587f4f1939278477c28a && \
    cd grpc && \
    mkdir -p cmake/build && \
    pushd cmake/build && \
    cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR ../.. && \
    make -j && \
    make install && \
    popd && \
    pip install grpcio && \
    pip install grpcio-tools

RUN cd /workspace

# torchrec && fbgemm
ENV NVML_PATH "/usr/local/cuda-11.3/targets/x86_64-linux/lib/stubs/libnvidia-ml.so"
RUN pip install torchrec_nightly && \
    git clone --recursive https://github.com/pytorch/FBGEMM.git && \
    cd FBGEMM/fbgemm_gpu && \
    python setup.py install --nvml_lib_path $NVML_PATH && \
    cd /opt/conda/lib/python3.8/site-packages && \
    rm -rf fbgemm_gpu && \
    cd fbgemm_gpu-0.1.1-py3.8-linux-x86_64.egg/fbgemm_gpu && \
    cp fbgemm_gpu_py.so libfbgemm_gpu_py.so && \
    cd .. && \
    cp fbgemm_gpu/* .

RUN cd /workspace

# general
ENV CMAKE_PREFIX_PATH "/opt/conda"
ENV PATH_TO_EXTERN_PYTHON_PACKAGES "/opt/conda/lib/python3.8/site-packages"
ENV MULTIPY_DIR_LOCATION "/opt/multipy"

ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/opt/conda/lib/python3.8/site-packages/torch/lib:/opt/conda/lib"
ENV LIBRARY_PATH "$LIBRARY_PATH:/opt/conda/lib/python3.8/site-packages/torch/lib:/opt/conda/lib"
