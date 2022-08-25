import os

import torch
import torch.distributed as dist
import torchrec_dynamic_embedding


__all__ = ["register_memory_io", "init_dist"]


MEMORY_IO_REGISTERED = False


def register_memory_io():
    global MEMORY_IO_REGISTERED
    if not MEMORY_IO_REGISTERED:
        torch.ops.tde.register_io(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "memory_io/memory_io.so"
            )
        )
        MEMORY_IO_REGISTERED = True


def init_dist():
    if not dist.is_initialized():
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "13579"
        dist.init_process_group("nccl")
