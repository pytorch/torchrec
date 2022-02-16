# TorchRec (Beta Release)

TorchRec is a PyTorch domain library built to provide common sparsity & parallelism primitives needed for large-scale recommender systems (RecSys). It allows authors to train models with large embedding tables sharded across many GPUs.

## TorchRec contains:
- Parallelism primitives that enable easy authoring of large, performant multi-device/multi-node models using hybrid data-parallelism/model-parallelism.
- The TorchRec sharder can shard embedding tables with different sharding strategies including data-parallel, table-wise, row-wise, table-wise-row-wise, and column-wise sharding.
- The TorchRec planner can automatically generate optimized sharding plans for models.
- Pipelined training overlaps dataloading device transfer (copy to GPU), inter-device communications (input_dist), and computation (forward, backward) for increased performance.
- Optimized kernels for RecSys powered by FBGEMM.
- Quantization support for reduced precision training and inference.
- Common modules for RecSys.
- Production-proven model architectures for RecSys.
- RecSys datasets (criteo click logs and movielens)
- Examples of end-to-end training such the dlrm event prediction model trained on criteo click logs dataset.

# Installation

Torchrec requires Python >= 3.7 and CUDA >= 11.0 (CUDA is highly recommended for performance but not required). The example below shows how to install with CUDA 11.3. This setup assumes you have conda installed.

## Binaries

Experimental binary on Linux for Python 3.7, 3.8 and 3.9 can be installed via pip wheels

### CUDA
```
conda install pytorch cudatoolkit=11.3 -c pytorch-nightly
pip install torchrec-nightly
```

### CPU Only
```
conda install pytorch cpuonly -c pytorch-nightly
pip install torchrec-nightly-cpu
```

### Colab example: introduction + install
See this [colab notebook](https://colab.research.google.com/drive/1xpcfMn5aFrCueDwUbJPl4hUOo54u0xMU) for an introduction to torchrec which includes runnable installation.

## From Source

We are currently iterating on the setup experience. For now, we provide manual instructions on how to build from source. The example below shows how to install with CUDA 11.3. This setup assumes you have conda installed.

1. Install pytorch. See [pytorch documentation](https://pytorch.org/get-started/locally/)
```
conda install pytorch cudatoolkit=11.3 -c pytorch-nightly
```

2. Install Requirements
```
pip install -r requirements.txt
```

3. Next, install FBGEMM_GPU from source (included in third_party folder of torchrec) by following the directions [here](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu). Installing fbgemm GPU is optional, but using FBGEMM w/ CUDA will be much faster. For CUDA 11.3 and SM80 (Ampere) architecture, the following instructions can be used:
```
export CUB_DIR=/usr/local/cuda-11.3/include/cub
export CUDA_BIN_PATH=/usr/local/cuda-11.3/
export CUDACXX=/usr/local/cuda-11.3/bin/nvcc
python setup.py install -DTORCH_CUDA_ARCH_LIST="7.0;8.0"
```
The last line of the above code block (`python setup.py install`...) which manually installs fbgemm_gpu can be skipped if you do not need to build fbgemm_gpu with custom build-related flags. Skip to the next step if that is the case.

4. Download and install TorchRec.
```
git clone --recursive https://github.com/facebookresearch/torchrec

# cd to the directory where torchrec's setup.py is located. Then run one of the below:
cd torchrec
python setup.py install develop --skip_fbgemm  # If you manually installed fbgemm_gpu in the previous step.
python setup.py install develop                # Otherwise. This will run the fbgemm_gpu install step for you behind the scenes.
python setup.py install develop --cpu_only     # For a CPU only installation of FBGEMM
```

5. Test the installation.
```
torchx run --scheduler local_cwd test_installation.py:test_installation
```

6. If you want to run a more complex example, please take a look at the torchrec [DLRM example](torchrec/github/examples/dlrm/dlrm_main.py).

## License
TorchRec is BSD licensed, as found in the [LICENSE](LICENSE) file.
