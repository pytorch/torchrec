# Torchrec

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

We are currently iterating on the setup experience. For now, we provide manual instructions on how to build from source. The example below shows how to install with CUDA 11.1. This setup assumes you have conda installed.

1. Save the following as environment.yml on your machine.
```
name: torchrec_py386_cuda111
channels:
  - pytorch-nightly
  - iopath
  - conda-forge
dependencies:
  - python=3.8.6
  - pytorch
  - cudatoolkit=11.1
  - iopath
  - numpy
  - pip:
    - "--editable=git+https://github.com/pytorch/torchx.git@main#egg=torchx"
    - torchmetrics
    - pyre-extensions
```

2. Create a new conda environment for torchrec.
```
mkdir torchrec_oss_install
cd torchrec_oss_install
conda env create -f environment.yml
conda activate torchrec_py386_cuda111
```

3. Download the TorchRec repo.
```
cd src
git clone --recursive https://github.com/facebookresearch/torchrec
```

4. Next, install FBGEMM_GPU from source (included in third_party folder of torchrec) by following the directions [here](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu). For CUDA 11.1 and SM80 (Ampere) architecture, the following instructions can be used:
```
conda install -c conda-forge scikit-build jinja2 ninja cmake
export TORCH_CUDA_ARCH_LIST=8.0
export CUB_DIR=/usr/local/cuda-11.1/include/cub
export CUDA_BIN_PATH=/usr/local/cuda-11.1/
export CUDACXX=/usr/local/cuda-11.1/bin/nvcc
python setup.py install -Dcuda_architectures="80" -DCUDNN_LIBRARY_PATH=/usr/local/cuda-11.1/lib64/libcudnn.so -DCUDNN_INCLUDE_PATH=/usr/local/cuda-11.1/include
```
The last line of the above code block (`python setup.py install`...) which manually installs fbgemm_gpu can be skipped if you do not need to build fbgemm_gpu with custom build-related flags. Skip to the next step if that is the case.

5. Then, install TorchRec from source.
```
# cd to the directory where torchrec's setup.py is located. Then run one of the below:
python setup.py build develop --skip_fbgemm  # If you manually installed fbgemm_gpu in the previous step.
python setup.py build develop                # Otherwise. This will run the fbgemm_gpu install step for you behind the scenes.
```

6. Next, we'll need to go into the TorchRec source to update some imports related to fbgemm_gpu. Append the line `import fbgemm_gpu` to the imports in the files [torchrec/sparse/jagged_tensor.py](torchrec/sparse/jagged_tensor.py), [torchrec/distributed/comm_ops.py](torchrec/distributed/comm_ops.py), [torchrec/distributed/dist_data.py](torchrec/distributed/dist_data.py), and [torchrec/quant/embedding_modules.py](torchrec/quant/embedding_modules.py). We installed in develop mode so editing the source will reflect changes without rebuilding. We expect to remove the need for this step soon.

7. Test the installation.
```
torchx run --scheduler local_cwd test_installation.py:test_installation
```

8. If you want to run a more complex example, please take a look at the torchrec [DLRM example](torchrec/examples/dlrm/dlrm_main.py).

That's it! In the near-to-mid future, we will simplify this process considerably. Stay tuned...

## License
TorchRec is BSD licensed, as found in the [LICENSE](LICENSE) file.
