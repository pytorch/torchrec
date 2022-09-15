# Criteo Preprocessing

The Criteo 1TB Clock Logs dataset is a dataset comprised of 24 days of feature values and click results. More info can be found here: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/.

The data is stored in tab separated value files which we need to preprocess to use them during training.

The transformations we apply include:

* Split the last_day to 2 parts for validation and test
* Fill in missing values with 0
* Compress value range for dense feature by applying f(x) = log(x + 3)
* Categorify the sparse features by setting the max size for each feature using NUM_EMBEDDINGS_PER_FEATURE_DICT

After transformation the values are dumped into 3 binary files (train, eval and test) for storage.

Then we split each binary files to 28 binary files including: label, dense and 26 sparse features.

To preprocess the data we are using NVTabular which speeds up the precessing by utilizing the parallel compute powers of our GPUs.

This script assumes that the files of the dataset have already been downloaded and extracted.

It then processes the files in three steps:

1. Split the last day tsv file and convert tsv files into parquet format

2. Apply tranformations

3. Dump values into 3 binary files: train, eval and test

4. Split each binary files into 28 binary files for label, dense and 26 sparse features

Note that the total amount of disk space needed for this process is about 2.5 TB (without the zipped original files).

For convenience this example used a docker container.

## Run docker container

To start the docker run:

    cd torchrec/datasets/scripts/nvt

    sudo apt-get install -y nvidia-docker2

    sudo systemctl daemon-reload
    sudo systemctl restart docker

    docker build -t criteo_preprocessing:latest . && \
    docker run --runtime=nvidia -ti --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --mount type=bind,source=$(pwd),target=/app\
    --mount type=bind,source=/data/,target=/data\
    criteo_preprocessing:latest  bash

In our example the original tsv files are located in /data/criteo_tb.

## Run conversion

To convert the dataset we then execute the following command:

    bash nvt_preproc.sh /data/criteo_tb /data 8096

The first input is the path of input and the second is the path of output and the third one is the batch_size when you dump the parquet to binary, and this batch_size has NO relationship of the batch_size in data loading

The final result can be found in /data/criteo_binary/split/train/ and /data/criteo_binary/split/test and /data/criteo_binary/split/validation/
