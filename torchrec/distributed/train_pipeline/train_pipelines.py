#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import logging
from collections import deque
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
)

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed.dist_data import KJTAllToAllTensorsAwaitable
from torchrec.distributed.model_parallel import ShardedModule
from torchrec.distributed.train_pipeline.utils import (
    _override_input_dist_forwards,
    _pipeline_detach_model,
    _rewrite_model,
    _start_data_dist,
    _start_embedding_lookup,
    _to_device,
    _wait_for_batch,
    _wait_for_event,
    DataLoadingThread,
    EmbeddingPipelinedForward,
    EmbeddingTrainPipelineContext,
    In,
    Out,
    PipelinedForward,
    PipelineStage,
    PrefetchPipelinedForward,
    PrefetchTrainPipelineContext,
    RunnableType,
    StageOut,
    StageOutputWithEvent,
    TrainPipelineContext,
)
from torchrec.distributed.types import Awaitable
from torchrec.pt2.checks import is_torchdynamo_compiling
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable


logger: logging.Logger = logging.getLogger(__name__)


class ModelDetachedException(Exception):
    pass


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
        context_type: Type[TrainPipelineContext] = TrainPipelineContext,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._device = device
        self._execute_all_batches = execute_all_batches
        self._apply_jit = apply_jit

        if device.type == "cuda":
            # use two data streams to support two concurrent batches
            # Dynamo does not support cuda stream specificaiton,
            # this freedom is left for compiler pipelining optimizations.
            assert (
                not is_torchdynamo_compiling()
            ), "Train Pipelines rely on cuda streams, which is not supported by Dynamo"

        self._memcpy_stream: Optional[torch.cuda.streams.Stream] = (
            (torch.cuda.Stream(priority=-1)) if device.type == "cuda" else None
        )
        self._data_dist_stream: Optional[torch.cuda.streams.Stream] = (
            (torch.cuda.Stream(priority=-1)) if device.type == "cuda" else None
        )

        # pyre-ignore
        self._original_forwards: List[Callable[..., Any]] = []

        self._original_kjt_dist_forwards: List[
            Callable[[KeyedJaggedTensor], Awaitable[KJTAllToAllTensorsAwaitable]]
        ] = []

        self._model_attached = True

        self._next_index: int = 0
        self.contexts: Deque[TrainPipelineContext] = deque()
        self._pipelined_modules: List[ShardedModule] = []
        self.batches: Deque[Optional[In]] = deque()
        self._dataloader_iter: Optional[Iterator[In]] = None
        self._dataloader_exhausted: bool = False
        self._context_type: Type[TrainPipelineContext] = context_type

        # DEPRECATED FIELDS
        self._batch_i: Optional[In] = None
        self._batch_ip1: Optional[In] = None
        self._batch_ip2: Optional[In] = None
        self._context: TrainPipelineContext = context_type(version=0)

    def detach(self) -> torch.nn.Module:
        """
        Detaches the model from sparse data dist (SDD) pipeline.
        To use the pipeline after detaching the model, pipeline.attach(model)
        needs to be called.
        Inflight batches are kept so pipeline.progress(data_iter) can be resumed normally.

        Returns the original model.
        """
        if self._pipelined_modules:
            _pipeline_detach_model(
                pipelined_modules=self._pipelined_modules,
                original_forwards=self._original_forwards,
                original_kjt_dist_forwards=self._original_kjt_dist_forwards,
            )

        self._model_attached = False
        return self._model

    def attach(self, model: Optional[torch.nn.Module] = None) -> None:
        if model:
            self._model = model

        self._model_attached = True
        if self.contexts:
            self._pipeline_model(
                batch=self.batches[0],
                context=self.contexts[0],
                pipelined_forward=PipelinedForward,
            )
        else:
            # attaching the model after end of train pipeline
            # model rewrite for SDD needs context but self.contexts is empty
            # reset _pipelined_modules so _fill_pipeline will rewrite model on progress()
            self._pipelined_modules = []

    def _set_module_context(self, context: TrainPipelineContext) -> None:
        for module in self._pipelined_modules:
            module.forward.set_context(context)

    def enqueue_batch(self, dataloader_iter: Iterator[In]) -> bool:
        batch, context = self.copy_batch_to_gpu(dataloader_iter)
        if batch is None:
            return False
        self.batches.append(batch)
        # pyre-ignore [6]
        self.contexts.append(context)

        return True

    def dequeue_batch(self) -> None:
        self.batches.popleft()
        self.contexts.popleft()
        # update PipelineForwards context to match next forward pass
        if len(self.batches) >= 1:
            self._set_module_context(self.contexts[0])

    def fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        # pipeline is already filled
        if len(self.batches) >= 2:
            return
        # executes last batch in pipeline
        if self.batches and self._execute_all_batches:
            return

        # batch i
        if not self.enqueue_batch(dataloader_iter):
            return

        self._init_pipelined_modules(
            # pyre-ignore [6]
            self.batches[0],
            self.contexts[0],
            PipelinedForward,
        )
        self.wait_sparse_data_dist(self.contexts[0])

        # batch i+1
        if not self.enqueue_batch(dataloader_iter):
            return

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        if not self._model_attached:
            self.attach(self._model)

        self.fill_pipeline(dataloader_iter)
        if not self.batches:
            raise StopIteration

        # TODO: Remove once Bulk Eval migrated (needed for bwd compat, this class only)
        self._set_module_context(self.contexts[0])

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self.batches[0]), self._data_dist_stream)

        if len(self.batches) >= 2:
            self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        # batch i+2
        self.enqueue_batch(dataloader_iter)

        # forward
        with record_function("## forward ##"):
            losses, output = cast(
                Tuple[torch.Tensor, Out], self._model(self.batches[0])
            )

        if len(self.batches) >= 2:
            self.wait_sparse_data_dist(self.contexts[1])

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()

            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        self.dequeue_batch()
        return output

    def _create_context(self) -> TrainPipelineContext:
        context = self._context_type(index=self._next_index, version=1)
        self._next_index += 1
        return context

    def _pipeline_model(
        self,
        batch: Optional[In],
        context: TrainPipelineContext,
        pipelined_forward: Type[PipelinedForward] = PipelinedForward,
    ) -> None:
        self._pipelined_modules, self._model, self._original_forwards = _rewrite_model(
            model=self._model,
            context=context,
            dist_stream=self._data_dist_stream,
            batch=batch,
            apply_jit=self._apply_jit,
            pipelined_forward=pipelined_forward,
        )
        # initializes input dist, so we can override input dist forwards
        self.start_sparse_data_dist(batch, context)
        self._original_kjt_dist_forwards = _override_input_dist_forwards(
            self._pipelined_modules
        )

    def _init_pipelined_modules(
        self,
        batch: In,
        context: TrainPipelineContext,
        pipelined_forward: Type[PipelinedForward] = PipelinedForward,
    ) -> None:
        """
        Retrieves the pipelined modules after overriding their forwards, initializes the
        modules' input dists, and overrides the input dist forwards to support fusing
        the splits collective in the input dist.
        """
        if self._pipelined_modules:
            self._set_module_context(context)
            self.start_sparse_data_dist(batch, context)
            return

        self._pipeline_model(batch, context, pipelined_forward)

    def copy_batch_to_gpu(
        self,
        dataloader_iter: Iterator[In],
    ) -> Tuple[Optional[In], Optional[TrainPipelineContext]]:
        """
        Retrieves batch from dataloader and moves it to the provided device.

        Raises:
            StopIteration: if the dataloader iterator is exhausted; unless
                `self._execute_all_batches=True`, then returns None.
        """
        context = None
        with record_function(f"## copy_batch_to_gpu {self._next_index} ##"):
            with torch.cuda.stream(self._memcpy_stream):
                batch = self._next_batch(dataloader_iter)
                if batch is not None:
                    batch = _to_device(batch, self._device, non_blocking=True)
                elif not self._execute_all_batches:
                    raise StopIteration
                context = self._create_context()
                return batch, context

    def _next_batch(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        """
        Retrieves next batch from dataloader and prevents calling `next` on an already
        exhausted dataloader, which can cause hanging.
        """
        if dataloader_iter is not self._dataloader_iter:
            self._dataloader_iter = dataloader_iter
            self._dataloader_exhausted = False

        if self._dataloader_exhausted:
            batch = None
        else:
            with record_function("## next_batch ##"):
                batch = next(dataloader_iter, None)
            if batch is None:
                self._dataloader_exhausted = True
        return batch

    def start_sparse_data_dist(
        self, batch: Optional[In], context: TrainPipelineContext
    ) -> None:
        """
        Waits for batch to finish getting copied to GPU, then starts the input dist.
        """
        if batch is None:
            return
        with record_function(f"## start_sparse_data_dist {context.index} ##"):
            with torch.cuda.stream(self._data_dist_stream):
                _wait_for_batch(batch, self._memcpy_stream)
                _start_data_dist(self._pipelined_modules, batch, context)

    def wait_sparse_data_dist(self, context: TrainPipelineContext) -> None:
        """
        Waits on the input dist splits requests to get the input dist tensors requests,
        and populates the context with them.
        """
        with record_function(f"## wait_sparse_data_dist {context.index} ##"):
            with torch.cuda.stream(self._data_dist_stream):
                for names, awaitable in context.fused_splits_awaitables:
                    for name, request in zip(names, awaitable.wait()):
                        context.input_dist_tensors_requests[name] = request
        context.input_dist_splits_requests.clear()
        context.fused_splits_awaitables.clear()

    def _copy_batch_to_gpu(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        """
        DEPRECATED: exists for backward compatibility on TrainPipelineContext.version 0
        """
        self._set_module_context(self._context)
        batch, _ = self.copy_batch_to_gpu(dataloader_iter)
        return batch

    def _start_sparse_data_dist(self, batch: Optional[In]) -> None:
        """
        DEPRECATED: exists for backward compatibility
        Waits for batch to finish getting copied to GPU, then starts the input dist.
        """
        self._set_module_context(self._context)
        self.start_sparse_data_dist(batch, self._context)

    def _wait_sparse_data_dist(self) -> None:
        """
        DEPRECATED: exists for backward compatibility
        Waits on the input dist splits requests to get the input dist tensors requests,
        and populates the context with them.
        """
        self._set_module_context(self._context)
        with record_function("## wait_sparse_data_dist ##"):
            with torch.cuda.stream(self._data_dist_stream):
                self._context.module_contexts = (
                    self._context.module_contexts_next_batch.copy()
                )
                self._context.input_dist_tensors_requests.clear()
                for names, awaitable in self._context.fused_splits_awaitables:
                    for name, request in zip(names, awaitable.wait()):
                        self._context.input_dist_tensors_requests[name] = request

    def _fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        """
        DEPRECATED: exists for backward compatibility
        """
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

        self._init_pipelined_modules(self._batch_i, self._context)
        self._start_sparse_data_dist(self._batch_i)
        self._wait_sparse_data_dist()

        # batch 2
        self._batch_ip1 = self._copy_batch_to_gpu(dataloader_iter)


class TrainPipelineSemiSync(TrainPipelineSparseDist[In, Out]):
    """
    Novel method for RecSys model training by leveraging "Semi-Synchronous" training,
    where the model is still synchorous but each batch prediction is calculated
    on parameters which were last updated B-2, instead of the batch prior (ie. B-1).  This
    allows the Embedding All-to-All from B to be fully overlapped with forward pass of B-1; dramatically
    improving peak training performance.


    Args:
        model (torch.nn.Module): model to pipeline.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device where device transfer, sparse data dist, and
            forward/backward pass will happen.
        execute_all_batches (bool): executes remaining batches in pipeline after
            exhausting dataloader iterator.
        apply_jit (bool): apply torch.jit.script to non-pipelined (unsharded) modules.
        start_batch (int): batch to begin semi-sync training.  Typically small period of synchorous training reduces early stage NEX.
        stash_gradients (bool): if True, will store gradients for each parameter to insure true "Semi-Sync"
            training.  If False, will update dense optimizer as soon as gradients available (naive "Semi-Sync)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        start_batch: int = 900,
        stash_gradients: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            execute_all_batches=execute_all_batches,
            apply_jit=apply_jit,
            context_type=EmbeddingTrainPipelineContext,
        )
        self._start_batch = start_batch
        self._stash_gradients = stash_gradients

        # use two data streams to support two concurrent batches
        self._embedding_odd_stream: Optional[torch.cuda.streams.Stream] = (
            (torch.cuda.Stream(priority=0)) if device.type == "cuda" else None
        )
        self._embedding_even_stream: Optional[torch.cuda.streams.Stream] = (
            (torch.cuda.Stream(priority=0)) if device.type == "cuda" else None
        )
        self._overarch_stream: Optional[torch.cuda.streams.Stream] = (
            (torch.cuda.Stream(priority=-1)) if device.type == "cuda" else None
        )
        self._gradients: Dict[str, torch.Tensor] = {}

    def _grad_swap(self) -> None:
        for name, param in self._model.named_parameters():
            grad = self._gradients.get(name, None)
            if param.grad is not None:
                self._gradients[name] = param.grad.clone()
            param.grad = grad

    def fill_pipeline(self, dataloader_iter: Iterator[In]) -> None:
        # pipeline is already filled
        if len(self.batches) >= 3:
            return
        # executes last batch in pipeline
        if self.batches and self._execute_all_batches:
            return

        # batch i
        if not self.enqueue_batch(dataloader_iter):
            return

        self._init_pipelined_modules(
            # pyre-ignore [6]
            self.batches[0],
            self.contexts[0],
            # pyre-ignore [6]
            EmbeddingPipelinedForward,
        )
        self.wait_sparse_data_dist(self.contexts[0])
        # pyre-ignore [6]
        self.start_embedding_lookup(self.batches[0], self.contexts[0])

        # batch i+1
        if not self.enqueue_batch(dataloader_iter):
            return
        self.start_sparse_data_dist(self.batches[1], self.contexts[1])
        self.wait_sparse_data_dist(self.contexts[1])

        # batch i+2
        if not self.enqueue_batch(dataloader_iter):
            return

    def is_semi_sync(self) -> bool:
        if len(self.batches) >= 1:
            # pyre-ignore [58]
            return self.contexts[0].index >= self._start_batch
        return False

    def _mlp_optimizer_step(self) -> None:
        # special case: not all optimizers support optim.step() on null gradidents
        if (
            len(self.batches) >= 1
            and self.contexts[0].index == self._start_batch
            and self._stash_gradients
        ):
            return
        self._optimizer.step()

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        self.fill_pipeline(dataloader_iter)
        if not self.batches:
            raise StopIteration

        if len(self.batches) >= 3:
            self.start_sparse_data_dist(
                self.batches[2],
                self.contexts[2],
            )

        losses, output = self._mlp_forward(cast(In, self.batches[0]), self.contexts[0])

        # batch i+3
        self.enqueue_batch(dataloader_iter)

        if len(self.batches) >= 2 and self.is_semi_sync():
            # pyre-ignore [6]
            self.start_embedding_lookup(self.batches[1], self.contexts[1])

        if len(self.batches) >= 3:
            self.wait_sparse_data_dist(self.contexts[2])

        if self._model.training:
            # backward would put an implicit sync point in stream called from, ideally
            # this would different from optimizer so it could start earilier, but currently not safe to do so.
            with torch.cuda.stream(self._overarch_stream):
                with record_function(f"## backward {self.contexts[0].index} ##"):
                    torch.sum(losses, dim=0).backward()

                with record_function(
                    f"## optimizer {cast(int, self.contexts[0].index) - 1} ##"
                ):
                    if self.is_semi_sync() and self._stash_gradients:
                        self._grad_swap()
                    self._mlp_optimizer_step()

                with record_function(
                    f"## zero_grad {cast(int, self.contexts[0].index) - 1} ##"
                ):
                    self._optimizer.zero_grad()

        if len(self.batches) >= 2 and not self.is_semi_sync():
            # pyre-ignore [6]
            self.start_embedding_lookup(self.batches[1], self.contexts[1])

        self.dequeue_batch()
        return output

    def _mlp_forward(
        self, batch: In, context: TrainPipelineContext
    ) -> Tuple[torch.Tensor, Out]:
        with record_function(f"## forward {context.index} ##"):
            with torch.cuda.stream(self._overarch_stream):
                _wait_for_event(batch, context.event)
                context.event = None
                return cast(Tuple[torch.Tensor, Out], self._model(batch))

    def copy_batch_to_gpu(
        self,
        dataloader_iter: Iterator[In],
    ) -> Tuple[Optional[In], Optional[TrainPipelineContext]]:
        context = None
        with record_function(f"## copy_batch_to_gpu {self._next_index} ##"):
            with torch.cuda.stream(self._memcpy_stream):
                batch = self._next_batch(dataloader_iter)
                if batch is not None:
                    batch = _to_device(batch, self._device, non_blocking=True)
                    context = self._create_context()
                    context.event = torch.cuda.Event()
                    context.event.record()
                return batch, context

    def start_sparse_data_dist(
        self,
        batch: Optional[In],
        context: TrainPipelineContext,
    ) -> None:
        """
        Waits for batch to finish getting copied to GPU, then starts the input dist.  This is Event based version.
        """
        if batch is None:
            return
        with record_function(f"## start_sparse_data_dist {context.index} ##"):
            with torch.cuda.stream(self._data_dist_stream):
                _wait_for_event(batch, context.event)
                _start_data_dist(self._pipelined_modules, batch, context)
                context.event = torch.cuda.Event()
                context.event.record()

    def start_embedding_lookup(
        self,
        batch: Optional[In],
        context: EmbeddingTrainPipelineContext,
    ) -> None:
        """
        Waits for batch to finish getting copied to GPU, then starts the input dist. This Event based vesrion.
        """
        if batch is None:
            return
        with record_function(f"## start_embedding_lookup {context.index} ##"):
            with torch.cuda.stream(
                self._embedding_even_stream
                if cast(int, context.index) % 2 == 0
                else self._embedding_odd_stream
            ):
                _wait_for_event(batch, context.event)
                _start_embedding_lookup(self._pipelined_modules, batch, context)
                context.event = torch.cuda.Event()
                context.event.record()


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
            context_type=PrefetchTrainPipelineContext,
        )
        self._context = PrefetchTrainPipelineContext(version=0)
        self._prefetch_stream: Optional[torch.cuda.streams.Stream] = (
            (torch.cuda.Stream()) if self._device.type == "cuda" else None
        )
        self._default_stream: Optional[torch.cuda.streams.Stream] = (
            (torch.cuda.Stream()) if self._device.type == "cuda" else None
        )
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

        self._init_pipelined_modules(
            self._batch_i,
            self._context,
            # pyre-ignore
            PrefetchPipelinedForward,
        )
        self._start_sparse_data_dist(self._batch_i)
        self._wait_sparse_data_dist()
        self._prefetch(self._batch_i)

        # batch 2
        self._batch_ip1 = self._copy_batch_to_gpu(dataloader_iter)
        self._start_sparse_data_dist(self._batch_ip1)

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        self._fill_pipeline(dataloader_iter)

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self._batch_i), self._prefetch_stream)

        self._batch_ip2 = self._copy_batch_to_gpu(dataloader_iter)

        self._wait_sparse_data_dist()
        # forward
        with record_function("## forward ##"):
            losses, output = cast(Tuple[torch.Tensor, Out], self._model(self._batch_i))

        self._prefetch(self._batch_ip1)

        if self._model.training:
            # backward
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()

            # update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        self._start_sparse_data_dist(self._batch_ip2)

        self._batch_i = self._batch_ip1
        self._batch_ip1 = self._batch_ip2

        return output

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
                    request = self._context.input_dist_tensors_requests.pop(
                        forward._name
                    )
                    assert isinstance(request, Awaitable)
                    with record_function("## wait_sparse_data_dist ##"):
                        # Finish waiting on the dist_stream,
                        # in case some delayed stream scheduling happens during the wait() call.
                        with torch.cuda.stream(self._data_dist_stream):
                            data = request.wait()

                    # Make sure that both result of input_dist and context
                    # are properly transferred to the current stream.
                    module_context = self._context.module_contexts[forward._name]
                    if self._data_dist_stream is not None:
                        torch.cuda.current_stream().wait_stream(self._data_dist_stream)
                        cur_stream = torch.cuda.current_stream()

                        assert isinstance(
                            data, (torch.Tensor, Multistreamable)
                        ), f"{type(data)} must implement Multistreamable interface"
                        data.record_stream(cur_stream)
                        data.record_stream(self._default_stream)

                        module_context.record_stream(cur_stream)
                        module_context.record_stream(self._default_stream)

                    sharded_module.prefetch(
                        ctx=module_context,
                        dist_input=data,
                        forward_stream=self._default_stream,
                    )
                    self._context.module_input_post_prefetch[forward._name] = data
                    self._context.module_contexts_post_prefetch[forward._name] = (
                        self._context.module_contexts.pop(forward._name)
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

            # batch 0
            # pyre-ignore [16]
            batch = self._batch_loader.get_next_batch()
            if batch is None:
                raise StopIteration
            self.batches.append(batch)
            self.contexts.append(self._create_context())

            self._init_pipelined_modules(
                # pyre-ignore
                self.batches[0],
                self.contexts[0],
                PipelinedForward,
            )
            self.start_sparse_data_dist(self.batches[0], self.contexts[0])
            self.wait_sparse_data_dist(self.contexts[0])

        batch = self._batch_loader.get_next_batch()
        if batch is not None:
            self.batches.append(batch)
            self.contexts.append(self._create_context())

        if len(self.batches) == 0:
            raise StopIteration

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cast(In, self.batches[0]), self._data_dist_stream)

        if len(self.batches) >= 2:
            self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        # forward
        with record_function("## forward ##"):
            losses, output = cast(
                Tuple[torch.Tensor, Out], self._model(self.batches[0])
            )

        if len(self.batches) >= 2:
            self.wait_sparse_data_dist(self.contexts[1])
        self.dequeue_batch()

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
        compute_stream (Optional[torch.cuda.Stream]): The main compute stream in which
            model forward is run, usually torch.cuda.default_stream(). Defaults to the
            current cuda stream.
        on_flush_end (Optional): Callback function that gets invoked after the pipeline
            has been flushed.

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
        compute_stream: Optional[torch.cuda.Stream] = None,
        on_flush_end: Optional[Callable[[], None]] = None,
    ) -> None:
        self._pipeline_stages = pipeline_stages
        self._debug_mode = debug_mode
        self._stage_outputs: List[Optional[StageOutputWithEvent]] = cast(
            List[Optional[StageOutputWithEvent]], [None] * len(self._pipeline_stages)
        )
        self._initialized = False
        self._num_steps = 0
        self._dataloader_iter: Optional[Iterator[In]] = None
        self._dataloader_exhausted: bool = False
        self._compute_stream: torch.cuda.streams.Stream = (
            compute_stream or torch.cuda.current_stream()
        )

        self._flushing: bool = False
        self.on_flush_end = on_flush_end

    @property
    def num_stages(self) -> int:
        return len(self._pipeline_stages)

    def _advance(self) -> Optional[StageOutputWithEvent]:
        # left shifts all batch results.
        out = self._stage_outputs[0]
        for idx in range(self.num_stages - 1):
            self._stage_outputs[idx] = self._stage_outputs[idx + 1]
        self._stage_outputs[-1] = None
        return out

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

    def _next_batch(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        """
        Retrieves next batch from dataloader and prevents calling `next` on an already
        exhausted dataloader, which can cause hanging.
        """
        if dataloader_iter is not self._dataloader_iter:
            self._dataloader_iter = dataloader_iter
            self._dataloader_exhausted = False

        if self._dataloader_exhausted or self._flushing:
            batch = None
        else:
            with record_function("## next_batch ##"):
                batch = next(dataloader_iter, None)
            if batch is None:
                self._dataloader_exhausted = True
        return batch

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
                batch_to_wait = self._next_batch(dataloader_iter)
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

    def set_flush(self, flush: bool) -> None:
        """
        Sets whether the pipeline should be flushed.

        When the pipeline is in a flushing state, it will stop getting further data from the dataloader and will continue executing the pipeline until all remaining stages are finished. Afterwards, it will invoke a callback and resume the pipeline.
        """
        self._flushing = flush

    def flush_end(self) -> None:
        self.set_flush(False)
        # Ensure pipeline gets filled again
        self._initialized = False

        if self.on_flush_end is not None:
            self.on_flush_end()

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

        output_with_event = self._advance()

        if output_with_event is None:
            # All data consumed, exit early
            return None

        self._num_steps += 1

        for stage_idx in range(self.num_stages):
            stage_output_idx = self.num_stages - 1 - stage_idx
            self._run_stage(
                batch_offset=stage_output_idx,
                stage_idx=stage_idx,
                dataloader_iter=dataloader_iter,
            )

        out, event = output_with_event
        if event is not None:
            # Since model forward() is expected to run outside the pipeline,
            # we need to explicitly wait for the last stage to finish
            event.wait(self._compute_stream)
            out.record_stream(self._compute_stream)

        if out is None and self._flushing:
            # We have exhausted all stages due to flushing
            self.flush_end()
            return self.progress(dataloader_iter)

        return out
