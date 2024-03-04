#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NOTE: Due to an internal packaging issue, `train_pipeline.py` must be compatible with
older versions of TorchRec. Importing new modules from other files may break model
publishing flows.
"""
import abc
import logging
from typing import cast, Generic, Iterator, List, Optional, Tuple

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed.model_parallel import ShardedModule
from torchrec.distributed.train_pipeline.utils import (
    _override_input_dist_forwards,
    _rewrite_model,
    _start_data_dist,
    _to_device,
    _wait_for_batch,
    DataLoadingThread,
    In,
    Out,
    PipelineStage,
    PrefetchPipelinedForward,
    PrefetchTrainPipelineContext,
    RunnableType,
    StageOut,
    StageOutputWithEvent,
    TrainPipelineContext,
)
from torchrec.distributed.types import Awaitable
from torchrec.streamable import Multistreamable

logger: logging.Logger = logging.getLogger(__name__)


class TrainPipeline(abc.ABC, Generic[In, Out]):
    @abc.abstractmethod
    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        pass


class TrainPipelineBase(TrainPipeline[In, Out]):
    """
    This class runs training iterations using a pipeline of two stages, each as a CUDA
    stream, namely, the current (default) stream and `self._memcpy_stream`. For each
    iteration, `self._memcpy_stream` moves the input from host (CPU) memory to GPU
    memory, and the default stream runs forward, backward, and optimization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._device = device
        self._memcpy_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream() if device.type == "cuda" else None
        )
        self._cur_batch: Optional[In] = None
        self._connected = False

    def _connect(self, dataloader_iter: Iterator[In]) -> None:
        cur_batch = next(dataloader_iter)
        self._cur_batch = cur_batch
        with torch.cuda.stream(self._memcpy_stream):
            self._cur_batch = _to_device(cur_batch, self._device, non_blocking=True)
        self._connected = True

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        if not self._connected:
            self._connect(dataloader_iter)

        # Fetch next batch
        with record_function("## next_batch ##"):
            next_batch = next(dataloader_iter)
        cur_batch = self._cur_batch
        assert cur_batch is not None

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cur_batch, self._memcpy_stream)

        with record_function("## forward ##"):
            (losses, output) = self._model(cur_batch)

        if self._model.training:
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()

        # Copy the next batch to GPU
        self._cur_batch = cur_batch = next_batch
        with record_function("## copy_batch_to_gpu ##"):
            with torch.cuda.stream(self._memcpy_stream):
                self._cur_batch = _to_device(cur_batch, self._device, non_blocking=True)

        # Update
        if self._model.training:
            with record_function("## optimizer ##"):
                self._optimizer.step()

        return output


class TrainPipelineSparseDist(TrainPipeline[In, Out]):
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
        model (torch.nn.Module): model to pipeline.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device where device transfer, sparse data dist, and
            forward/backward pass will happen.
        execute_all_batches (bool): executes remaining batches in pipeline after
            exhausting dataloader iterator.
        apply_jit (bool): apply torch.jit.script to non-pipelined (unsharded) modules.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._device = device
        self._execute_all_batches = execute_all_batches
        self._apply_jit = apply_jit
        # use two data streams to support two concurrent batches
        if device.type == "cuda":
            self._memcpy_stream: Optional[torch.cuda.streams.Stream] = (
                torch.cuda.Stream(priority=-1)
            )
            self._data_dist_stream: Optional[torch.cuda.streams.Stream] = (
                torch.cuda.Stream(priority=-1)
            )
        else:
            self._memcpy_stream: Optional[torch.cuda.streams.Stream] = None
            self._data_dist_stream: Optional[torch.cuda.streams.Stream] = None
        self._batch_i: Optional[In] = None
        self._batch_ip1: Optional[In] = None
        self._batch_ip2: Optional[In] = None
        self._context = TrainPipelineContext()
        self._pipelined_modules: List[ShardedModule] = []

    def _fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        # pipeline is already filled
        if self._batch_i and self._batch_ip1:
            return
        # executes last batch in pipeline
        if self._batch_i and self._execute_all_batches:
            return

        # batch 1
        self._batch_i = self._copy_batch_to_gpu(dataloader_iter)
        if self._batch_i is None:
            raise StopIteration

        self._init_pipelined_modules(self._batch_i)
        self._start_sparse_data_dist(self._batch_i)
        self._wait_sparse_data_dist()

        # batch 2
        self._batch_ip1 = self._copy_batch_to_gpu(dataloader_iter)

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        self._fill_pipeline(dataloader_iter)

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self._batch_i), self._data_dist_stream)

        self._start_sparse_data_dist(self._batch_ip1)

        self._batch_ip2 = self._copy_batch_to_gpu(dataloader_iter)

        # forward
        with record_function("## forward ##"):
            losses, output = cast(Tuple[torch.Tensor, Out], self._model(self._batch_i))

        self._wait_sparse_data_dist()

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()

            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        self._batch_i = self._batch_ip1
        self._batch_ip1 = self._batch_ip2

        return output

    def _init_pipelined_modules(self, batch: In) -> None:
        """
        Retrieves the pipelined modules after overriding their forwards, initializes the
        modules' input dists, and overrides the input dist forwards to support fusing
        the splits collective in the input dist.
        """
        if self._pipelined_modules:
            return
        self._pipelined_modules, self._model = _rewrite_model(
            model=self._model,
            context=self._context,
            dist_stream=self._data_dist_stream,
            batch=self._batch_i,
            apply_jit=self._apply_jit,
        )
        # initializes input dist, so we can override input dist forwards
        self._start_sparse_data_dist(self._batch_i)
        _override_input_dist_forwards(self._pipelined_modules)

    def _copy_batch_to_gpu(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        """
        Retrieves batch from dataloader and moves it to the provided device.

        Raises:
            StopIteration: if the dataloader iterator is exhausted; unless
                `self._execute_all_batches=True`, then returns None.
        """
        with record_function("## copy_batch_to_gpu ##"):
            with torch.cuda.stream(self._memcpy_stream):
                batch = next(dataloader_iter, None)
                if batch is not None:
                    batch = _to_device(batch, self._device, non_blocking=True)
                elif not self._execute_all_batches:
                    raise StopIteration
                return batch

    def _start_sparse_data_dist(self, batch: Optional[In]) -> None:
        """
        Waits for batch to finish getting copied to GPU, then starts the input dist.
        """
        if batch is None:
            return
        with record_function("## start_sparse_data_dist ##"):
            with torch.cuda.stream(self._data_dist_stream):
                _wait_for_batch(batch, self._memcpy_stream)
                _start_data_dist(self._pipelined_modules, batch, self._context)

    def _wait_sparse_data_dist(self) -> None:
        """
        Waits on the input dist splits requests to get the input dist tensors requests,
        and populates the context with them.
        """
        with record_function("## wait_sparse_data_dist ##"):
            with torch.cuda.stream(self._data_dist_stream):
                self._context.module_contexts = (
                    self._context.module_contexts_next_batch.copy()
                )
                self._context.input_dist_tensors_requests.clear()
                for names, awaitable in self._context.fused_splits_awaitables:
                    for name, request in zip(names, awaitable.wait()):
                        self._context.input_dist_tensors_requests[name] = request


class PrefetchTrainPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    """
    This pipeline overlaps device transfer, `ShardedModule.input_dist()`, and cache
    prefetching with forward and backward. This helps hide the all2all latency while
    preserving the training forward / backward ordering.

    stage 4: forward, backward - uses default CUDA stream
    stage 3: prefetch - uses prefetch CUDA stream
    stage 2: ShardedModule.input_dist() - uses data_dist CUDA stream
    stage 1: device transfer - uses memcpy CUDA stream

    `ShardedModule.input_dist()` is only done for top-level modules in the call graph.
    To be considered a top-level module, a module can only depend on 'getattr' calls on
    input.

    Input model must be symbolically traceable with the exception of `ShardedModule` and
    `DistributedDataParallel` modules.

    Args:
        model (torch.nn.Module): model to pipeline.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device where device transfer, sparse data dist, prefetch,
            and forward/backward pass will happen.
        execute_all_batches (bool): executes remaining batches in pipeline after
            exhausting dataloader iterator.
        apply_jit (bool): apply torch.jit.script to non-pipelined (unsharded) modules.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
        )
        self._context = PrefetchTrainPipelineContext()
        if self._device.type == "cuda":
            self._prefetch_stream: Optional[torch.cuda.streams.Stream] = (
                torch.cuda.Stream()
            )
            self._default_stream: Optional[torch.cuda.streams.Stream] = (
                torch.cuda.current_stream()
            )
        else:
            self._prefetch_stream: Optional[torch.cuda.streams.Stream] = None
            self._default_stream: Optional[torch.cuda.streams.Stream] = None
        self._batch_ip3: Optional[In] = None

    def _fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        # pipeline is already filled
        if self._batch_i and self._batch_ip1 and self._batch_ip2:
            return
        # executes last batch in pipeline
        if self._execute_all_batches and (self._batch_i or self._batch_ip1):
            return

        # batch 1
        self._batch_i = self._copy_batch_to_gpu(dataloader_iter)
        if self._batch_i is None:
            raise StopIteration

        self._init_pipelined_modules(self._batch_i)
        self._start_sparse_data_dist(self._batch_i)
        self._wait_sparse_data_dist()
        self._prefetch(self._batch_i)

        # batch 2
        self._batch_ip1 = self._copy_batch_to_gpu(dataloader_iter)
        self._start_sparse_data_dist(self._batch_ip1)
        self._wait_sparse_data_dist()

        # batch 3
        self._batch_ip2 = self._copy_batch_to_gpu(dataloader_iter)

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        self._fill_pipeline(dataloader_iter)

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self._batch_i), self._prefetch_stream)

        self._start_sparse_data_dist(self._batch_ip2)

        self._batch_ip3 = self._copy_batch_to_gpu(dataloader_iter)

        # forward
        with record_function("## forward ##"):
            losses, output = cast(Tuple[torch.Tensor, Out], self._model(self._batch_i))

        self._prefetch(self._batch_ip1)

        self._wait_sparse_data_dist()

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()

            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        self._batch_i = self._batch_ip1
        self._batch_ip1 = self._batch_ip2
        self._batch_ip2 = self._batch_ip3

        return output

    def _init_pipelined_modules(self, batch: In) -> None:
        """
        Retrieves the pipelined modules after overriding their forwards, initializes the
        modules' input dists, and overrides the input dist forwards to support fusing
        the splits collective in the input dist.
        """
        if self._pipelined_modules:
            return
        self._pipelined_modules, self._model = _rewrite_model(
            model=self._model,
            context=self._context,
            dist_stream=self._data_dist_stream,
            batch=self._batch_i,
            apply_jit=self._apply_jit,
            pipelined_forward=PrefetchPipelinedForward,
        )

        # initializes input dist, so we can override input dist forwards
        self._start_sparse_data_dist(self._batch_i)
        _override_input_dist_forwards(self._pipelined_modules)

    def _prefetch(self, batch: Optional[In]) -> None:
        """
        Waits for input dist to finish, then prefetches data.
        """
        if batch is None:
            return
        self._context.module_input_post_prefetch.clear()
        self._context.module_contexts_post_prefetch.clear()

        with record_function("## sharded_module_prefetch ##"):
            with torch.cuda.stream(self._prefetch_stream):
                batch.record_stream(torch.cuda.current_stream())
                for sharded_module in self._pipelined_modules:
                    forward = sharded_module.forward
                    assert isinstance(forward, PrefetchPipelinedForward)

                    assert forward._name in self._context.input_dist_tensors_requests
                    request = self._context.input_dist_tensors_requests[forward._name]
                    assert isinstance(request, Awaitable)
                    with record_function("## wait_sparse_data_dist ##"):
                        # Finish waiting on the dist_stream,
                        # in case some delayed stream scheduling happens during the wait() call.
                        with torch.cuda.stream(self._data_dist_stream):
                            data = request.wait()

                    # Make sure that both result of input_dist and context
                    # are properly transferred to the current stream.
                    if self._data_dist_stream is not None:
                        torch.cuda.current_stream().wait_stream(self._data_dist_stream)
                        cur_stream = torch.cuda.current_stream()

                        assert isinstance(
                            data, (torch.Tensor, Multistreamable)
                        ), f"{type(data)} must implement Multistreamable interface"
                        data.record_stream(cur_stream)
                        data.record_stream(self._default_stream)

                        ctx = self._context.module_contexts[forward._name]
                        ctx.record_stream(cur_stream)
                        ctx.record_stream(self._default_stream)

                    sharded_module.prefetch(
                        dist_input=data, forward_stream=self._default_stream
                    )
                    self._context.module_input_post_prefetch[forward._name] = data
                    self._context.module_contexts_post_prefetch[forward._name] = (
                        self._context.module_contexts[forward._name]
                    )


class EvalPipelineSparseDist(TrainPipelineSparseDist[In, Out]):
    """
    This pipeline overlaps device transfer, and `ShardedModule.input_dist()` with
    forward. This helps hide the all2all latency. We use a background thread to
    perform device transfer to further reduce latency.

    stage 2: forward- uses default CUDA stream
    stage 1: ShardedModule.input_dist() - uses data_dist CUDA stream
    background: device transfer - uses memcpy CUDA stream

    `ShardedModule.input_dist()` is only done for top-level modules in the call graph.
    To be considered a top-level module, a module can only depend on 'getattr' calls on
    input.

    Input model must be symbolically traceable with the exception of `ShardedModule` and
    `DistributedDataParallel` modules.

    Args:
        model (torch.nn.Module): model to pipeline.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device where device transfer, sparse data dist, and
            forward/backward pass will happen.
        apply_jit (bool): apply torch.jit.script to non-pipelined (unsharded) modules.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        apply_jit: bool = False,
    ) -> None:
        super().__init__(model, optimizer, device, True, apply_jit)
        self._batch_loader: Optional[DataLoadingThread[In]] = None

    def __del__(self) -> None:
        if self._batch_loader is not None:
            self._batch_loader.stop()

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        if not self._batch_loader:
            self._batch_loader = DataLoadingThread(
                device=self._device,
                dataloader_iter=dataloader_iter,
                to_device_non_blocking=True,
                memcpy_stream_priority=-1,
            )
            self._batch_loader.start()

            batch_loader = self._batch_loader
            assert batch_loader is not None

            # batch 1
            self._batch_i = batch_loader.get_next_batch()
            assert self._batch_i is not None

            self._init_pipelined_modules(self._batch_i)
            self._start_sparse_data_dist(self._batch_i)
            self._wait_sparse_data_dist()

            # batch 2
            self._batch_ip1 = batch_loader.get_next_batch()

        if self._batch_i is None:
            raise StopIteration

        batch_loader = self._batch_loader
        assert batch_loader is not None
        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self._batch_i), self._data_dist_stream)

        self._start_sparse_data_dist(self._batch_ip1)

        # forward
        with record_function("## forward ##"):
            losses, output = cast(Tuple[torch.Tensor, Out], self._model(self._batch_i))

        self._wait_sparse_data_dist()

        self._batch_i = self._batch_ip1
        self._batch_ip1 = batch_loader.get_next_batch()

        return output


class StagedTrainPipeline(TrainPipeline[In, Optional[StageOut]]):
    """
    StagedTrainPipeline orchestrates the pipelined execution of its constitutent stages
    from inputs of `dataloader_iter`. Namely scheduling the execution of stages before
    model forward.

    NOTE: the SDD stage needs to be the final stage of the pipeline so that the
        `ShardedModule` forward can properly consume the SDD output.

    Calling progress on a `StagedTrainPipeline` provides an output that is equivalent to
    calling each of the pipeline stages in order.

    In the example below a fully synchronous will expose the `data_copy` and
    `gpu_preproc` calls. After pipelining, the `data_copy` of batch i+2 can be
    overlapped with the `gpu_preproc` of batch i+1 and the main model processing of
    batch i.

    Args:
        pipeline_stages (List[PipelineStage]): A list of stages to execute.
        debug_mode (bool): Whether to enable debug mode.

    Example::
        train_pipeline = StagedTrainPipeline(
            pipeline=[
                PipelineStage(
                    name="data_copy",
                    runnable=get_h2d_func("cuda"),
                    stream=torch.cuda.Stream(),
                ),
                PipelineStage(
                    name="gpu_preproc",
                    runnable=gpu_preproc,
                    stream=torch.cuda.Stream(),
                ),
            ]
        )

        while batch_for_forward := train_pipeline.progress(dataloader_iter):
            optimizer.zero_grad()
            loss, pred = model(batch_for_forward)
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        pipeline_stages: List[PipelineStage],
        debug_mode: bool = False,
    ) -> None:
        self._pipeline_stages = pipeline_stages
        self._debug_mode = debug_mode
        self._stage_outputs: List[Optional[StageOutputWithEvent]] = cast(
            List[Optional[StageOutputWithEvent]], [None] * len(self._pipeline_stages)
        )
        self._initialized = False
        self._num_steps = 0

    @property
    def num_stages(self) -> int:
        return len(self._pipeline_stages)

    def _advance(self) -> Optional[StageOut]:
        # left shifts all batch results.
        out = self._stage_outputs[0]
        for idx in range(self.num_stages - 1):
            self._stage_outputs[idx] = self._stage_outputs[idx + 1]
        self._stage_outputs[-1] = None
        if out is None:
            return out
        return out[0]

    def _run_with_event(
        self,
        runnable: RunnableType,
        event: Optional[torch.cuda.Event],
        inputs: Optional[In],
        stream: torch.cuda.streams.Stream,
    ) -> StageOutputWithEvent:
        if inputs is None:
            return (None, None)
        with torch.cuda.stream(stream):
            # If there is no previous event, data is entering the pipeline
            if event is not None:
                event.wait(stream)
                inputs.record_stream(stream)

            output = runnable(inputs)
            new_event = torch.cuda.Event()
            new_event.record(stream)
            return (output, new_event)

    def _run_stage(
        self,
        batch_offset: int,
        stage_idx: int,
        dataloader_iter: Iterator[In],
        fill: bool = False,
    ) -> StageOutputWithEvent:
        """
        Each stage of the pipeline MUST have an input and output. If the input is None,
        it means there is no more data to process. The stage will short circuit and NOT
        execute the runnable.
        """
        stage = self._pipeline_stages[stage_idx]

        with record_function(
            f"## Pipeline Stage {stage_idx} : {stage.name} for batch {batch_offset + self._num_steps} ##"
        ):
            if stage_idx == 0:
                batch_to_wait = next(dataloader_iter, None)
                event = None
            else:
                batch_to_wait_with_event = self._stage_outputs[batch_offset]
                assert batch_to_wait_with_event is not None
                batch_to_wait, event = batch_to_wait_with_event

            new_result = self._run_with_event(
                runnable=stage.runnable,
                event=event,
                inputs=batch_to_wait,
                stream=stage.stream,
            )

        self._stage_outputs[batch_offset] = new_result
        if self._debug_mode:
            logger.info(
                f"Running ## Pipeline Stage {stage_idx} : {stage.name} for batch {batch_offset + self._num_steps} ##",
            )

        if fill and (fill_callback := stage.fill_callback) is not None:
            if self._debug_mode:
                logger.info(f"Finished callback for {stage.name}")
            fill_callback()

        return new_result

    def _fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        """
        There should always be `self.num_stages` batches in flight. This function
        initializes the pipeline by filling it with `self.num_stages` batches.
        Intuitively, it does all the stages before the model forward.

        NOTE:
            model forward should be executed outside the pipeline in the train loop,
            using the output of `progress` as its input.

        For a 3 stage pipeline during `_fill_pipeline`:
            batch 0: stages 0, 1, 2 will be run
            batch 1: stages 0, 1 will be run
            batch 2: stage 0 will be run
            batch 3: will start in `progress()`

        In the initial `progress()`
            batch 0: model forward will be run
            batch 1: stage 2 will be run
            batch 2: stage 1 will be run
            batch 3: stage 0 will be run
        """
        for batch_offset in range(self.num_stages):
            stages_to_run = self.num_stages - batch_offset
            for stage_idx in range(stages_to_run):
                self._run_stage(
                    batch_offset=batch_offset,
                    stage_idx=stage_idx,
                    dataloader_iter=dataloader_iter,
                    fill=True,
                )

        self._initialized = True
        if self._debug_mode:
            logger.info("Finished fill pipeline")

    def progress(
        self,
        dataloader_iter: Iterator[In],
    ) -> Optional[StageOut]:
        """
        The pipeline processes data in reverse order, so stage_0 processes the
        newest data and stage_n processes the oldest.

        NOTE:
            if SDD is enabled it must be the last stage in the pipeline.

        Args:
            data_iter (Iterator[In]): An iterator that produces the inputs to
                the pipeline.

        Returns:
            Optional[StageOut]: Output of the final stage. `None` signifies that the
                dataloader iterator is depleted.
        """
        if not self._initialized:
            self._fill_pipeline(dataloader_iter)

        output = self._advance()
        self._num_steps += 1

        for stage_idx in range(self.num_stages):
            stage_output_idx = self.num_stages - 1 - stage_idx
            self._run_stage(
                batch_offset=stage_output_idx,
                stage_idx=stage_idx,
                dataloader_iter=dataloader_iter,
            )

        return output
