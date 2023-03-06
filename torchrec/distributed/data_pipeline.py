#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import cast, Dict, Iterator, List, Optional, TypeVar

import torch

from torch.profiler import record_function
from torchrec.distributed.model_parallel import ShardedModule
from torchrec.distributed.pipeline_utils import (
    _rewrite_model,
    _start_data_dist,
    _to_device,
    _wait_for_batch,
    TrainPipelineContext,
)
from torchrec.streamable import Pipelineable

logger: logging.Logger = logging.getLogger(__name__)

In = TypeVar("In", bound=Pipelineable)


class DataPipeline(Iterator[In]):
    def __init__(
        self,
        dataloader_iter: Iterator[In],
    ) -> None:
        torch._C._log_api_usage_once(
            f"torchrec.distributed.data_pipeline.{self.__class__.__name__}"
        )
        self._dataloader_iter = dataloader_iter

    def __iter__(self) -> "DataPipeline[In]":
        return self

    def __next__(self) -> In:
        return next(self._dataloader_iter)


class CudaCopyingPipeline(DataPipeline[In]):
    """
    This class runs training iterations using a pipeline of two stages, each as a CUDA
    stream, namely, the current (default) stream and `self._memcpy_stream`. For each
    iteration, `self._memcpy_stream` moves the input from host (CPU) memory to GPU
    memory, and the default stream runs forward, backward, and optimization.
    Args:
        dataloader_iter: Iterator for the input data
        device: Device to put the data on
    """

    def __init__(
        self,
        dataloader_iter: Iterator[In],
        device: torch.device,
    ) -> None:
        super().__init__(dataloader_iter)
        self._device = device
        self._memcpy_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream() if device.type == "cuda" else None
        )
        self._cur_batch: Optional[In] = None
        self._connected = False

    def _connect(self) -> None:
        cur_batch = next(self._dataloader_iter)
        with torch.cuda.stream(self._memcpy_stream):
            self._cur_batch = _to_device(cur_batch, self._device, non_blocking=True)
        self._connected = True

    def __next__(self) -> In:
        if not self._connected:
            self._connect()

        # Fetch next batch
        next_batch: Optional[In] = None
        with record_function("## next_batch ##"):
            try:
                next_batch = next(self._dataloader_iter)
            except StopIteration:
                pass

        if self._cur_batch is None:
            raise StopIteration

        cur_batch = self._cur_batch

        # Wait for stream.
        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cur_batch, self._memcpy_stream)

        # Copy the next batch to GPU
        if next_batch:
            with record_function("## copy_batch_to_gpu ##"):
                with torch.cuda.stream(self._memcpy_stream):
                    self._cur_batch = _to_device(
                        next_batch, self._device, non_blocking=True
                    )
        else:
            self._cur_batch = None

        return cur_batch


class SparseDistPipeline(DataPipeline[In]):
    """
    This pipeline overlaps device transfer, and `ShardedModule.input_dist()` with
    forward and backward. This helps hide the all2all latency while preserving the
    training forward / backward ordering.
    stage 3: forward, backward - uses default CUDA stream
    stage 2: ShardedModule.input_dist() - uses data_dist CUDA stream
    stage 1: device transfer - uses memcpy CUDA stream
    `ShardedModule.input_dist()` is only done for top-level modules in the call graph.
    To be considered a top-level module, a module can only depend on 'getattr' calls on
    input.
    Input model must be symbolically traceable with the exception of `ShardedModule` and
    `DistributedDataParallel` modules.
    Args:
        dataloader_iter: Iterator for the input data
        device: Device to put the data on
        model: Model to be pipelined

    Note: This feature is experimental.
    """

    synced_pipeline_id: Dict[int, int] = {}

    def __init__(
        self,
        dataloader_iter: Iterator[In],
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(dataloader_iter)
        self._model = model
        self._device = device
        # use two data streams to support two concurrent batches
        if device.type == "cuda":
            self._memcpy_stream: Optional[
                torch.cuda.streams.Stream
            ] = torch.cuda.Stream()
            self._data_dist_stream: Optional[
                torch.cuda.streams.Stream
            ] = torch.cuda.Stream()
        else:
            self._memcpy_stream: Optional[torch.cuda.streams.Stream] = None
            self._data_dist_stream: Optional[torch.cuda.streams.Stream] = None
        self._batch_i: Optional[In] = None
        self._batch_ip1: Optional[In] = None
        self._batch_ip2: Optional[In] = None
        self._connected = False
        self._context = TrainPipelineContext()
        self._pipelined_modules: List[ShardedModule] = []
        self._event: Optional[torch.cuda.Event] = None

    def _connect(self) -> None:
        # batch 1
        with torch.cuda.stream(self._memcpy_stream):
            batch_i = next(self._dataloader_iter)
            self._batch_i = batch_i = _to_device(
                batch_i, self._device, non_blocking=True
            )
            # Try to pipeline input data dist.
            self._pipelined_modules = _rewrite_model(
                self._model, self._context, self._data_dist_stream
            )

        self._sparse_data_dist()

        # batch 2
        with torch.cuda.stream(self._memcpy_stream):
            try:
                batch_ip1 = next(self._dataloader_iter)
                self._batch_ip1 = batch_ip1 = _to_device(
                    batch_ip1, self._device, non_blocking=True
                )
            except StopIteration:
                pass
        self._connected = True
        self.__class__.synced_pipeline_id[id(self._model)] = id(self)

    def __next__(self) -> In:
        if not self._connected:
            self._connect()
        elif self.__class__.synced_pipeline_id.get(id(self._model), None) != id(self):
            self._sync_pipeline()
            self.__class__.synced_pipeline_id[id(self._model)] = id(self)

        if self._batch_i is None:
            raise StopIteration

        with record_function("## copy_batch_to_gpu ##"):
            with torch.cuda.stream(self._memcpy_stream):
                try:
                    batch_ip2 = next(self._dataloader_iter)
                    self._batch_ip2 = batch_ip2 = _to_device(
                        batch_ip2, self._device, non_blocking=True
                    )
                except StopIteration:
                    self._batch_ip2 = batch_ip2 = None

        batch_i = cast(In, self._batch_i)
        batch_ip1 = cast(In, self._batch_ip1)

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(batch_i, self._data_dist_stream)

        # Make sure that previous data distribution has finished before starting next one.
        # Otherwise, it may cause memory bloat and hurt perf.
        if self._data_dist_stream:
            event = torch.cuda.current_stream().record_event()
            # pyre-ignore
            self._data_dist_stream.wait_event(event)

        # Data distribution for batch B + 1
        self._sparse_data_dist()

        self._batch_i = batch_ip1
        self._batch_ip1 = self._batch_ip2

        return batch_i

    def _sparse_data_dist(self) -> None:
        # Data distribution for batch B + 1
        if self._batch_i:
            with record_function("## sparse_data_dist ##"):
                with torch.cuda.stream(self._data_dist_stream):
                    # pyre-ignore
                    _wait_for_batch(self._batch_i, self._memcpy_stream)
                    _start_data_dist(
                        self._pipelined_modules,
                        # pyre-ignore
                        self._batch_i,
                        self._context,
                    )

    def _sync_pipeline(self) -> None:
        """
        Syncs `PipelinedForward` for sharded modules with context and dist stream of the
        current train pipeline. Used when switching between train pipelines for the same
        model.
        """
        for module in self._pipelined_modules:
            module.forward._context = self._context
            module.forward._dist_stream = self._data_dist_stream
