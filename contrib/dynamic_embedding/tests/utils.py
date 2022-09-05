import os

import torch
import torch.distributed as dist
import torchrec_dynamic_embedding


__all__ = ["register_memory_io", "init_dist"]


MEMORY_IO_REGISTERED = False


def register_memory_io():
    global MEMORY_IO_REGISTERED
    if not MEMORY_IO_REGISTERED:
        mem_io_path = os.getenv("TDE_MEMORY_IO_PATH")
        if mem_io_path is None:
            raise RuntimeError("env TDE_MEMORY_IO_PATH must set for unittest")

        torch.ops.tde.register_io(mem_io_path)
        MEMORY_IO_REGISTERED = True


def init_dist():
    if not dist.is_initialized():
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "13579"
        dist.init_process_group("nccl")
