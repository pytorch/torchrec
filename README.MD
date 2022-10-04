# TorchRec (Beta Release)
[Docs](https://pytorch.org/torchrec/)

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

Torchrec requires Python >= 3.7 and CUDA >= 11.0 (CUDA is highly recommended for performance but not required). The example below shows how to install with CUDA 11.6. This setup assumes you have conda installed.

## Binaries

Experimental binary on Linux for Python 3.7, 3.8 and 3.9 can be installed via pip wheels

### Installations
```
TO use the library without cuda, use the *-cpu fbgemm installations. However, this will be much slower than the CUDA variant.

Nightly

conda install pytorch pytorch-cuda=11.7 -c pytorch-nightly -c nvidia
pip install torchrec_nightly

Stable

conda install pytorch cudatoolkit=11.6 -c pytorch -c conda-forge
pip install torchrec

If you have no CUDA device:

Nightly

pip uninstall fbgemm-gpu-nightly -y
pip install fbgemm-gpu-nightly-cpu

Stable

pip uninstall fbgemm-gpu -y
pip install fbgemm-gpu-cpu

```


### Colab example: introduction + install
See our colab notebook for an introduction to torchrec which includes runnable installation.
    - [Tutorial Source](https://github.com/pytorch/torchrec/blob/main/Torchrec_Introduction.ipynb)
    - Open in [Google Colab](https://colab.research.google.com/github/pytorch/torchrec/blob/main/Torchrec_Introduction.ipynb)

## From Source

We are currently iterating on the setup experience. For now, we provide manual instructions on how to build from source. The example below shows how to install with CUDA 11.3. This setup assumes you have conda installed.

1. Install pytorch. See [pytorch documentation](https://pytorch.org/get-started/locally/)
   ```
   conda install pytorch pytorch-cuda=11.7 -c pytorch-nightly -c nvidia
   ```

2. Install Requirements
   ```
   pip install -r requirements.txt
   ```

3. Download and install TorchRec.
   ```
   git clone --recursive https://github.com/pytorch/torchrec

   cd torchrec
   python setup.py install develop
   ```

4. Test the installation.
   ```
   GPU mode

   torchx run -s local_cwd dist.ddp -j 1x2 --gpu 2 --script test_installation.py

   CPU Mode

   torchx run -s local_cwd dist.ddp -j 1x2 --script test_installation.py -- --cpu_only
   ```
   See [TorchX](https://pytorch.org/torchx/) for more information on launching distributed and remote jobs.

5. If you want to run a more complex example, please take a look at the torchrec [DLRM example](https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/dlrm_main.py).

## License
TorchRec is BSD licensed, as found in the [LICENSE](LICENSE) file.
