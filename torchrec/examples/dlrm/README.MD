# Running

## Torchx
We recommend using [torchx](https://pytorch.org/torchx/main/quickstart.html) to run.
Here we use the [DDP builtin](https://pytorch.org/torchx/main/components/distributed.html)

1. pip install torchx
2. (optional) setup a slurm or kubernetes cluster
3.
    a. locally: torchx run dist.ddp -j 1x2 --script dlrm_main.py
    b. remotely: torchx run -s slurm dist.ddp -j 1x8 --script dlrm_main.py
