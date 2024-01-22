# TorchRec (Beta Release)
[Docs](https://pytorch.org/torchrec/)

TorchRec is a PyTorch domain library built to provide common sparsity & parallelism primitives needed for large-scale recommender systems (RecSys). It allows authors to train models with large embedding tables sharded across many GPUs.

## TorchRec contains:
- Parallelism primitives that enable easy authoring of large, performant multi-device/multi-node models using hybrid data-parallelism/model-parallelism.
- The TorchRec sharder can shard embedding tables with different sharding strategies including data-parallel, table-wise, row-wise, table-wise-row-wise, column-wise, table-wise-column-wise sharding.
- The TorchRec planner can automatically generate optimized sharding plans for models.
- Pipelined training overlaps dataloading device transfer (copy to GPU), inter-device communications (input_dist), and computation (forward, backward) for increased performance.
- Optimized kernels for RecSys powered by FBGEMM.
- Quantization support for reduced precision training and inference.
- Common modules for RecSys.
- Production-proven model architectures for RecSys.
- RecSys datasets (criteo click logs and movielens)
- Examples of end-to-end training such the dlrm event prediction model trained on criteo click logs dataset.

# Installation

Torchrec requires Python >= 3.8 and CUDA >= 11.8 (CUDA is highly recommended for performance but not required). The example below shows how to install with Python 3.8 and CUDA 12.1. This setup assumes you have conda installed.

## Binaries

Experimental binary on Linux for Python 3.8, 3.9, 3.10, 3.11 and 3.12 (experimental), and CPU, CUDA 11.8 and CUDA 12.1 can be installed via pip wheels from [download.pytorch.org](download.pytorch.org) and PyPI (only for CUDA 12.1).

Below we show installations for CUDA 12.1 as an example. For CPU or CUDA 11.8, swap "cu121" for "cpu" or "cu118".

### Installations
```
Nightly

pip install torch --index-url https://download.pytorch.org/whl/nightly/cu121
pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cu121
pip install torchmetrics==1.0.3
pip install torchrec --index-url https://download.pytorch.org/whl/nightly/cu121

Stable via pytorch.org

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics==1.0.3
pip install torchrec --index-url https://download.pytorch.org/whl/cu121

Stable via PyPI (only for CUDA 12.1)

pip install torch
pip install fbgemm-gpu
pip install torchrec

```


### Colab example: introduction + install
See our colab notebook for an introduction to torchrec which includes runnable installation.
    - [Tutorial Source](https://github.com/pytorch/torchrec/blob/main/Torchrec_Introduction.ipynb)
    - Open in [Google Colab](https://colab.research.google.com/github/pytorch/torchrec/blob/main/Torchrec_Introduction.ipynb)

## From Source

We are currently iterating on the setup experience. For now, we provide manual instructions on how to build from source. The example below shows how to install with CUDA 12.1. This setup assumes you have conda installed.

1. Install pytorch. See [pytorch documentation](https://pytorch.org/get-started/locally/).
   ```
   CUDA 12.1

   pip install torch --index-url https://download.pytorch.org/whl/nightly/cu121

   CUDA 11.8

   pip install torch --index-url https://download.pytorch.org/whl/nightly/cu118

   CPU

   pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

2. Clone TorchRec.
   ```
   git clone --recursive https://github.com/pytorch/torchrec
   cd torchrec
   ```

3. Install FBGEMM.
   ```
   CUDA 12.1

   pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cu121

   CUDA 11.8

   pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cu118

   CPU

   pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

4. Install other requirements.
   ```
   pip install -r requirements.txt
   ```

4. Install TorchRec.
   ```
   python setup.py install develop
   ```

5. Test the installation (use torchx-nightly for 3.11; for 3.12, torchx currently doesn't work).
   ```
   GPU mode

   torchx run -s local_cwd dist.ddp -j 1x2 --gpu 2 --script test_installation.py

   CPU Mode

   torchx run -s local_cwd dist.ddp -j 1x2 --script test_installation.py -- --cpu_only
   ```
   See [TorchX](https://pytorch.org/torchx/) for more information on launching distributed and remote jobs.

5. If you want to run a more complex example, please take a look at the torchrec [DLRM example](https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/dlrm_main.py).

## Contributing

### Pyre and linting

Before landing, please make sure that pyre and linting look okay. To run our linters, you will need to
```
pip install pre-commit
```

, and run it.

For Pyre, you will need to
```
cat .pyre_configuration
pip install pyre-check-nightly==<VERSION FROM CONFIG>
pyre check
```

We will also check for these issues in our GitHub actions.

## License
TorchRec is BSD licensed, as found in the [LICENSE](LICENSE) file.
