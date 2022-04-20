FROM nvcr.io/nvidia/merlin/merlin-pytorch-training:nightly

RUN conda install -y pytorch cudatoolkit=11.3 -c pytorch-nightly \
    && pip install torchrec-nightly

WORKDIR /app
