#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import os
import random
import socket
import time
from contextlib import closing
from functools import wraps
from typing import Callable, Optional, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from pyre_extensions import ParameterSpecification

TParams = ParameterSpecification("TParams")
TReturn = TypeVar("TReturn")


def get_free_port() -> int:
    # INTERNAL
    if os.getenv("SANDCASTLE") == "1" or os.getenv("TW_JOB_USER") == "sandcastle":
        if socket.has_ipv6:
            family = socket.AF_INET6
            address = "localhost6"
        else:
            family = socket.AF_INET
            address = "localhost4"
        with socket.socket(family, socket.SOCK_STREAM) as s:
            try:
                s.bind((address, 0))
                s.listen(0)
                with closing(s):
                    return s.getsockname()[1]
            except socket.gaierror:
                if address == "localhost6":
                    address = "::1"
                else:
                    address = "127.0.0.1"
                s.bind((address, 0))
                s.listen(0)
                with closing(s):
                    return s.getsockname()[1]
            except Exception as e:
                raise Exception(
                    f"Binding failed with address {address} while getting free port {e}"
                )
    # OSS GHA: TODO remove when enable ipv6 on GHA @omkar
    else:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                s.listen(0)
                with closing(s):
                    return s.getsockname()[1]
        except Exception as e:
            raise Exception(
                f"Binding failed with address 127.0.0.1 while getting free port {e}"
            )


def is_asan() -> bool:
    """Determines if the Python interpreter is running with ASAN"""
    return hasattr(ctypes.CDLL(""), "__asan_init")


def is_tsan() -> bool:
    """Determines if the Python interpreter is running with TSAN"""
    return hasattr(ctypes.CDLL(""), "__tsan_init")


def is_asan_or_tsan() -> bool:
    return is_asan() or is_tsan()


def skip_if_asan(
    func: Callable[TParams, TReturn]
) -> Callable[TParams, Optional[TReturn]]:
    """Skip test run if we are in ASAN mode."""

    @wraps(func)
    def wrapper(*args: TParams.args, **kwargs: TParams.kwargs) -> Optional[TReturn]:
        if is_asan_or_tsan():
            print("Skipping test run since we are in ASAN mode.")
            return
        return func(*args, **kwargs)

    return wrapper


def skip_if_asan_class(cls: TReturn) -> Optional[TReturn]:
    if is_asan_or_tsan():
        print("Skipping test run since we are in ASAN mode.")
        return
    return cls


def init_distributed_single_host(
    rank: int, world_size: int, backend: str, local_size: Optional[int] = None
) -> dist.ProcessGroup:
    os.environ["LOCAL_WORLD_SIZE"] = str(local_size if local_size else world_size)
    os.environ["LOCAL_RANK"] = str(rank % local_size if local_size else rank)
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
    # pyre-fixme[7]: Expected `ProcessGroup` but got
    #  `Optional[_distributed_c10d.ProcessGroup]`.
    return dist.group.WORLD


# pyre-ignore [24]
def seed_and_log(wrapped_func: Callable) -> Callable:
    # pyre-ignore [2, 3]
    def _wrapper(*args, **kwargs):
        seed = int(time.time() * 1000) % (1 << 31)
        print(f"Using random seed: {seed}")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return wrapped_func(*args, **kwargs)

    return _wrapper
