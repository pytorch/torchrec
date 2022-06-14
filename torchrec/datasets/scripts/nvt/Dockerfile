FROM nvcr.io/nvidia/merlin/merlin-pytorch-training:nightly

RUN conda install -y pytorch cudatoolkit=11.3 -c pytorch-nightly \
    && pip install --pre torchrec_nightly -f https://download.pytorch.org/whl/nightly/torchrec_nightly/index.html

WORKDIR /app
