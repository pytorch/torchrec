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
import copy
import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    cast,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from torch import distributed as dist
from torch.autograd.profiler import record_function
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.fx.node import Node
from torchrec.distributed.dist_data import KJTAllToAll, KJTAllToAllTensorsAwaitable
from torchrec.distributed.embedding_sharding import (
    KJTListAwaitable,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.model_parallel import DistributedModelParallel, ShardedModule
from torchrec.distributed.types import Awaitable, NoWait
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable, Pipelineable

logger: logging.Logger = logging.getLogger(__name__)


In = TypeVar("In", bound=Pipelineable)
Out = TypeVar("Out")


class TrainPipeline(abc.ABC, Generic[In, Out]):
    @abc.abstractmethod
    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        pass


def _to_device(batch: In, device: torch.device, non_blocking: bool) -> In:
    assert isinstance(
        batch, (torch.Tensor, Pipelineable)
    ), f"{type(batch)} must implement Pipelineable interface"
    # print("batch in to device", batch)
    return batch
    return cast(In, batch.to(device=device, non_blocking=non_blocking))


def _wait_for_batch(batch: In, stream: Optional[torch.cuda.streams.Stream]) -> None:
    if stream is None:
        return
    torch.cuda.current_stream().wait_stream(stream)
    """
    As mentioned in
    https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html, PyTorch
    uses the "caching allocator" for memory allocation for tensors. When a tensor is
    freed, its memory is likely to be reused by newly constructed tenosrs. By default,
    this allocator traces whether a tensor is still in use by only the CUDA stream where
    it was created. When a tensor is used by additional CUDA streams, we need to call
    `record_stream` to tell the allocator about these streams. Otherwise, the allocator
    might free the underlying memory of the tensor once it is no longer used by the
    creator stream. This is a notable programming trick when we write programs using
    multiple CUDA streams.
    """

    cur_stream = torch.cuda.current_stream()
    assert isinstance(
        batch, (torch.Tensor, Multistreamable)
    ), f"{type(batch)} must implement Multistreamable interface"
    batch.record_stream(cur_stream)


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

        # if self._model.training:
        #     with record_function("## backward ##"):
        #         torch.sum(losses, dim=0).backward()

        # Copy the next batch to GPU
        self._cur_batch = cur_batch = next_batch
        with record_function("## copy_batch_to_gpu ##"):
            with torch.cuda.stream(self._memcpy_stream):
                self._cur_batch = self._cur_batch
                # _to_device(cur_batch, self._device, non_blocking=True)

        # # Update
        # if self._model.training:
        #     with record_function("## optimizer ##"):
        #         self._optimizer.step()

        return output


class Tracer(torch.fx.Tracer):
    """
    Disables proxying buffers during tracing. Ideally, proxying buffers would be
    disabled, but some models are currently mutating buffer values, which causes errors
    during tracing. If those models can be rewritten to not do that, we can likely
    remove this line.
    """

    proxy_buffer_attributes = False

    def __init__(self, leaf_modules: Optional[List[str]] = None) -> None:
        super().__init__()
        self._leaf_modules: List[str] = leaf_modules if leaf_modules is not None else []

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if (
            isinstance(m, ShardedModule)
            or module_qualified_name in self._leaf_modules
            or isinstance(m, FSDP)
        ):
            return True
        return super().is_leaf_module(m, module_qualified_name)


# TODO: remove after packaging issue is resolved.
class SplitsAllToAllAwaitable(Awaitable[List[List[int]]]):
    def __init__(
        self,
        input_tensors: List[torch.Tensor],
        pg: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.num_workers: int = pg.size()

        with record_function("## all2all_data:kjt splits ##"):
            self._output_tensor: torch.Tensor = torch.empty(
                [self.num_workers * len(input_tensors)],
                device=input_tensors[0].device,
                dtype=input_tensors[0].dtype,
            )
            input_tensor = torch.stack(input_tensors, dim=1).flatten()
            self._splits_awaitable: dist.Work = dist.all_to_all_single(
                output=self._output_tensor,
                input=input_tensor,
                group=pg,
                async_op=True,
            )

    def _wait_impl(self) -> List[List[int]]:
        self._splits_awaitable.wait()
        return self._output_tensor.view(self.num_workers, -1).T.tolist()


# TODO: remove after packaging issue is resolved.
C = TypeVar("C", bound=Multistreamable)
T = TypeVar("T")


# TODO: remove after packaging issue is resolved.
def _set_sharding_context_intra_a2a(
    tensors_awaitables: List[Awaitable[KeyedJaggedTensor]],
    ctx: C,
) -> None:
    for awaitable, sharding_context in zip(
        tensors_awaitables,
        getattr(ctx, "sharding_contexts", []),
    ):
        if isinstance(awaitable, KJTAllToAllTensorsAwaitable):
            if hasattr(sharding_context, "input_splits"):
                sharding_context.input_splits = awaitable._input_splits["values"]
            if hasattr(sharding_context, "output_splits"):
                sharding_context.output_splits = awaitable._output_splits["values"]
            if hasattr(sharding_context, "sparse_features_recat"):
                sharding_context.sparse_features_recat = awaitable._recat
            if (
                hasattr(sharding_context, "batch_size_per_rank")
                and awaitable._stride_per_rank is not None
            ):
                sharding_context.batch_size_per_rank = awaitable._stride_per_rank


# TODO: remove after packaging issue is resolved.
def _set_sharding_context_pre_a2a(
    awaitables: List[Awaitable[Awaitable[KeyedJaggedTensor]]],
    ctx: C,
) -> None:
    for awaitable, sharding_context in zip(
        awaitables,
        getattr(ctx, "sharding_contexts", []),
    ):
        kjt = (
            awaitable._obj._obj
            if isinstance(awaitable, NoWait)
            else awaitable._input  # pyre-ignore[16]: KJTAllToAllSplitsAwaitable or KJTSplitsAllToAllMeta
        )
        if hasattr(sharding_context, "batch_size_per_feature_pre_a2a"):
            sharding_context.batch_size_per_feature_pre_a2a = kjt.stride_per_key()
        if hasattr(sharding_context, "variable_batch_per_feature"):
            sharding_context.variable_batch_per_feature = kjt.variable_stride_per_key()


# TODO: remove after packaging issue is resolved.
@dataclass
class KJTSplitsAllToAllMeta:
    pg: dist.ProcessGroup
    _input: KeyedJaggedTensor
    splits: List[int]
    splits_tensors: List[torch.Tensor]
    input_splits: List[List[int]]
    input_tensors: List[torch.Tensor]
    labels: List[str]
    keys: List[str]
    device: torch.device
    stagger: int


# TODO: remove after packaging issue is resolved.
def _split(flat_list: List[T], splits: List[int]) -> List[List[T]]:
    return [
        flat_list[sum(splits[:i]) : sum(splits[:i]) + n] for i, n in enumerate(splits)
    ]


# TODO: remove after packaging issue is resolved.
class FusedKJTListSplitsAwaitable(Awaitable[List[KJTListAwaitable]]):
    def __init__(
        self,
        requests: List[KJTListSplitsAwaitable[C]],
        contexts: List[C],
        pg: Optional[dist.ProcessGroup],
    ) -> None:
        super().__init__()
        self._contexts = contexts
        self._awaitables: List[
            Union[KJTSplitsAllToAllMeta, Awaitable[Awaitable[KeyedJaggedTensor]]]
        ] = [awaitable for request in requests for awaitable in request.awaitables]
        for req, ctx in zip(requests, self._contexts):
            _set_sharding_context_pre_a2a(req.awaitables, ctx)
        self._output_lengths: List[int] = [
            len(request.awaitables) for request in requests
        ]
        self._lengths: List[int] = [
            len(awaitable.splits_tensors)
            if isinstance(awaitable, KJTSplitsAllToAllMeta)
            else 0
            for awaitable in self._awaitables
        ]
        splits_tensors = [
            splits_tensor
            for awaitable in self._awaitables
            for splits_tensor in (
                awaitable.splits_tensors
                if isinstance(awaitable, KJTSplitsAllToAllMeta)
                else []
            )
        ]
        self._splits_awaitable: Optional[SplitsAllToAllAwaitable] = (
            SplitsAllToAllAwaitable(
                input_tensors=splits_tensors,
                pg=pg,
            )
            if splits_tensors and pg is not None
            else None
        )

    def _wait_impl(self) -> List[KJTListAwaitable]:
        if self._splits_awaitable:
            splits_list = self._splits_awaitable.wait()
            splits_per_awaitable = _split(splits_list, self._lengths)
        else:
            splits_per_awaitable = [[] for _ in range(len(self._lengths))]
        tensors_awaitables = []
        for splits, awaitable in zip(splits_per_awaitable, self._awaitables):
            if not splits:  # NoWait
                assert isinstance(awaitable, Awaitable)
                tensors_awaitables.append(awaitable.wait())
                continue
            assert isinstance(awaitable, KJTSplitsAllToAllMeta)
            if awaitable._input.variable_stride_per_key():
                output_splits = splits
                stride_per_rank = None
            else:
                output_splits = splits[:-1]
                stride_per_rank = splits[-1]
            tensors_awaitables.append(
                KJTAllToAllTensorsAwaitable(
                    pg=awaitable.pg,
                    input=awaitable._input,
                    splits=awaitable.splits,
                    input_splits=awaitable.input_splits,
                    output_splits=output_splits,
                    input_tensors=awaitable.input_tensors,
                    labels=awaitable.labels,
                    keys=awaitable.keys,
                    device=awaitable.device,
                    stagger=awaitable.stagger,
                    stride_per_rank=stride_per_rank,
                )
            )
        output = []
        awaitables_per_output = _split(tensors_awaitables, self._output_lengths)
        for awaitables, ctx in zip(awaitables_per_output, self._contexts):
            _set_sharding_context_intra_a2a(awaitables, ctx)
            output.append(KJTListAwaitable(awaitables, ctx))
        return output


@dataclass
class TrainPipelineContext:
    """
    Context information for a `TrainPipelineSparseDist` instance.

    Attributes:
        input_dist_splits_requests (Dict[str, Awaitable[Any]]): Stores input dist
            requests in the splits awaitable stage, which occurs after starting the
            input dist.
        input_dist_tensors_requests (Dict[str, Awaitable[Any]]): Stores input dist
            requests in the tensors awaitable stage, which occurs after calling `wait()`
            on the splits awaitable.
        module_contexts (Dict[str, Multistreamable]): Stores module contexts from the
            input dist for the current batch.
        module_contexts_next_batch (Dict[str, Multistreamable]): Stores module contexts
            from the input dist for the next batch.
        fused_splits_awaitables (List[Tuple[List[str], FusedKJTListSplitsAwaitable]]):
            List of fused splits input dist awaitable and the corresponding module names
            of each awaitable.
    """

    # pyre-ignore [4]
    input_dist_splits_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    # pyre-ignore [4]
    input_dist_tensors_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    module_contexts: Dict[str, Multistreamable] = field(default_factory=dict)
    module_contexts_next_batch: Dict[str, Multistreamable] = field(default_factory=dict)
    fused_splits_awaitables: List[
        Tuple[List[str], FusedKJTListSplitsAwaitable]
    ] = field(default_factory=list)


@dataclass
class PrefetchTrainPipelineContext(TrainPipelineContext):
    module_input_post_prefetch: Dict[str, Multistreamable] = field(default_factory=dict)
    module_contexts_post_prefetch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )


@dataclass
class ArgInfo:
    """
    Representation of args from a node.

    Attributes:
        input_attrs (List[str]): attributes of input batch,
            e.g. `batch.attr1.attr2` will produce ["attr1", "attr2"].
        is_getitems (List[bool]): `batch[attr1].attr2` will produce [True, False].
        name (Optional[str]): name for kwarg of pipelined forward() call or None for a
            positional arg.
    """

    input_attrs: List[str]
    is_getitems: List[bool]
    name: Optional[str]


class BaseForward:
    def __init__(
        self,
        name: str,
        args: List[ArgInfo],
        module: ShardedModule,
        context: TrainPipelineContext,
        stream: Optional[torch.cuda.streams.Stream],
    ) -> None:
        self._name = name
        self._args = args
        self._module = module
        self._context = context
        self._stream = stream

    @property
    def name(self) -> str:
        return self._name

    @property
    def args(self) -> List[ArgInfo]:
        return self._args


class PipelinedForward(BaseForward):
    # pyre-ignore [2, 24]
    def __call__(self, *input, **kwargs) -> Awaitable:
        assert self._name in self._context.input_dist_tensors_requests
        request = self._context.input_dist_tensors_requests[self._name]
        assert isinstance(request, Awaitable)
        with record_function("## wait_sparse_data_dist ##"):
            # Finish waiting on the dist_stream,
            # in case some delayed stream scheduling happens during the wait() call.
            with torch.cuda.stream(self._stream):
                data = request.wait()

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
            cur_stream = torch.cuda.current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            data.record_stream(cur_stream)

            ctx = self._context.module_contexts[self._name]
            ctx.record_stream(cur_stream)

        return self._module.compute_and_output_dist(
            self._context.module_contexts[self._name], data
        )


class PrefetchPipelinedForward(BaseForward):
    def __init__(
        self,
        name: str,
        args: List[ArgInfo],
        module: ShardedModule,
        context: PrefetchTrainPipelineContext,
        prefetch_stream: Optional[torch.cuda.streams.Stream],
    ) -> None:
        super().__init__(
            name=name,
            args=args,
            module=module,
            context=context,
            stream=prefetch_stream,
        )
        self._context: PrefetchTrainPipelineContext = self._context

    # pyre-ignore [2, 24]
    def __call__(self, *input, **kwargs) -> Awaitable:
        assert self._name in self._context.module_input_post_prefetch
        data = self._context.module_input_post_prefetch[self._name]

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
            cur_stream = torch.cuda.current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            data.record_stream(cur_stream)

            ctx = self._context.module_contexts_post_prefetch[self._name]
            ctx.record_stream(cur_stream)

        return self._module.compute_and_output_dist(
            self._context.module_contexts_post_prefetch[self._name], data
        )


class KJTAllToAllForward:
    def __init__(
        self, pg: dist.ProcessGroup, splits: List[int], stagger: int = 1
    ) -> None:
        self._pg = pg
        self._splits = splits
        self._stagger = stagger
        self._splits_cumsum: List[int] = [0] + list(itertools.accumulate(splits))

    def __call__(self, input: KeyedJaggedTensor) -> KJTSplitsAllToAllMeta:
        with torch.no_grad():
            assert len(input.keys()) == sum(self._splits)
            rank = dist.get_rank(self._pg)
            local_keys = input.keys()[
                self._splits_cumsum[rank] : self._splits_cumsum[rank + 1]
            ]
            input_splits = input.dist_splits(self._splits)
            device = input.values().device
            splits_tensors = [
                torch.tensor(splits, device=device) for splits in input_splits
            ]
            if not input.variable_stride_per_key():
                splits_tensors.append(
                    torch.tensor([input.stride()] * self._pg.size(), device=device)
                )
            return KJTSplitsAllToAllMeta(
                pg=self._pg,
                _input=input,
                splits=self._splits,
                splits_tensors=splits_tensors,
                input_splits=input_splits,
                input_tensors=input.dist_tensors(),
                labels=input.dist_labels(),
                keys=local_keys,
                device=device,
                stagger=self._stagger,
            )


def _start_data_dist(
    pipelined_modules: List[ShardedModule],
    batch: In,
    context: TrainPipelineContext,
) -> None:
    context.input_dist_splits_requests.clear()
    context.module_contexts_next_batch.clear()
    context.fused_splits_awaitables.clear()
    for module in pipelined_modules:
        forward = module.forward
        assert isinstance(forward, PipelinedForward) or isinstance(
            forward, PrefetchPipelinedForward
        )

        # Retrieve argument for the input_dist of EBC
        # is_getitem True means this argument could be retrieved by a list
        # False means this argument is getting while getattr
        # and this info was done in the _rewrite_model by tracing the
        # entire model to get the arg_info_list
        args = []
        kwargs = {}
        for arg_info in forward.args:
            if arg_info.input_attrs:
                arg = batch
                for attr, is_getitem in zip(arg_info.input_attrs, arg_info.is_getitems):
                    if is_getitem:
                        arg = arg[attr]
                    else:
                        arg = getattr(arg, attr)
                if arg_info.name:
                    kwargs[arg_info.name] = arg
                else:
                    args.append(arg)
            else:
                args.append(None)
        # Start input distribution.
        module_ctx = module.create_context()
        context.module_contexts_next_batch[forward.name] = module_ctx
        context.input_dist_splits_requests[forward.name] = module.input_dist(
            module_ctx, *args, **kwargs
        )
    _fuse_input_dist_splits(context)


def _fuse_input_dist_splits(context: TrainPipelineContext) -> None:
    names_per_pg = defaultdict(list)
    for name, request in context.input_dist_splits_requests.items():
        pg = None
        if isinstance(request, KJTListSplitsAwaitable):
            for awaitable in request.awaitables:
                if isinstance(awaitable, KJTSplitsAllToAllMeta):
                    pg = awaitable.pg
                    break
        names_per_pg[pg].append(name)

    for pg, names in names_per_pg.items():
        context.fused_splits_awaitables.append(
            (
                names,
                FusedKJTListSplitsAwaitable(
                    # pyre-ignore[6]
                    requests=[
                        context.input_dist_splits_requests[name] for name in names
                    ],
                    contexts=[
                        context.module_contexts_next_batch[name] for name in names
                    ],
                    pg=pg,
                ),
            )
        )


def _get_node_args_helper(
    # pyre-ignore
    arguments,
    num_found: int,
) -> Tuple[List[ArgInfo], int]:
    """
    Goes through the args/kwargs of a node and arranges them into a list of `ArgInfo`s.
    It also counts the number of (args + kwargs) found.
    """

    arg_info_list = [ArgInfo([], [], None) for _ in range(len(arguments))]
    for arg, arg_info in zip(arguments, arg_info_list):
        if arg is None:
            num_found += 1
            continue
        while True:
            if not isinstance(arg, torch.fx.Node):
                break
            child_node = arg

            if child_node.op == "placeholder":
                if hasattr(child_node, "ph_key"):
                    # pyre-ignore[16]
                    arg_info.input_attrs.insert(0, child_node.ph_key)
                    arg_info.is_getitems.insert(0, False)
                num_found += 1
                break
            elif (
                child_node.op == "call_function"
                and child_node.target.__module__ == "builtins"
                # pyre-ignore[16]
                and child_node.target.__name__ == "getattr"
            ):
                # pyre-fixme[6]: For 2nd argument expected `str` but got
                #  `Union[None, Dict[str, typing.Any], List[typing.Any], Node, bool,
                #  complex, float, int, range, slice, str, device, dtype, layout,
                #  memory_format, Tensor, typing.Tuple[typing.Any, ...]]`.
                arg_info.input_attrs.insert(0, child_node.args[1])
                arg_info.is_getitems.insert(0, False)
                arg = child_node.args[0]
            elif (
                child_node.op == "call_function"
                and child_node.target.__module__ == "_operator"
                # pyre-ignore[16]
                and child_node.target.__name__ == "getitem"
            ):
                # pyre-fixme[6]: For 2nd argument expected `str` but got
                #  `Union[None, Dict[str, typing.Any], List[typing.Any], Node, bool,
                #  complex, float, int, range, slice, str, device, dtype, layout,
                #  memory_format, Tensor, typing.Tuple[typing.Any, ...]]`.
                arg_info.input_attrs.insert(0, child_node.args[1])
                arg_info.is_getitems.insert(0, True)
                arg = child_node.args[0]
            elif (
                child_node.op == "call_function"
                and child_node.target.__module__ == "torch.utils._pytree"
                # pyre-ignore[16]
                and child_node.target.__name__ == "tree_unflatten"
            ):
                """
                This is for the PT2 export path where we unflatten the input to reconstruct
                the structure with the recorded tree spec.
                """
                assert arg_info.is_getitems[0]
                # pyre-fixme[16]
                arg = child_node.args[0][arg_info.input_attrs[0]]
            elif (
                child_node.op == "call_function"
                and child_node.target.__module__ == "torchrec.sparse.jagged_tensor"
                # pyre-fixme[16]
                and child_node.target.__name__ == "KeyedJaggedTensor"
            ):
                if "values" in child_node.kwargs:
                    arg = child_node.kwargs["values"]
                else:
                    arg = child_node.args[1]
            else:
                break
    return arg_info_list, num_found


def _get_node_args(
    node: Node,
) -> Tuple[List[ArgInfo], int]:
    num_found = 0
    pos_arg_info_list, num_found = _get_node_args_helper(node.args, num_found)
    kwargs_arg_info_list, num_found = _get_node_args_helper(
        node.kwargs.values(), num_found
    )

    # Replace with proper names for kwargs
    for name, arg_info_list in zip(node.kwargs, kwargs_arg_info_list):
        arg_info_list.name = name

    arg_info_list = pos_arg_info_list + kwargs_arg_info_list
    return arg_info_list, num_found


def _get_leaf_module_names_helper(
    model: torch.nn.Module,
    path: str,
    leaf_module_names: Set[str],
) -> bool:
    sharded_children = set()
    for name, child in model.named_children():
        curr_path = path + name
        if isinstance(child, ShardedModule):
            sharded_children.add(name)
        else:
            child_sharded = _get_leaf_module_names_helper(
                child,
                curr_path + ".",
                leaf_module_names,
            )
            if child_sharded:
                sharded_children.add(name)

    if len(sharded_children) > 0:
        for name, child in model.named_children():
            if name in sharded_children:
                continue
            # assume module is leaf node unless annotated otherwise
            if not getattr(child, "_is_pytorch_fx_traceable", False):
                leaf_module_names.add(path + name)
    return len(sharded_children) > 0


def _get_leaf_module_names(model: torch.nn.Module) -> List[str]:
    """
    Returns a list of top level modules to be used as leaf modules for FX tracing.
    This is a shallow FX trace that only goes the minimum depth required to pipeline
    the model unless child modules are explicitly tagged as `_is_pytorch_fx_traceable`.
    """

    leaf_module_names: Set[str] = set()
    _get_leaf_module_names_helper(
        model,
        "",
        leaf_module_names,
    )
    return list(leaf_module_names)


def _jit_modules(module: torch.nn.Module, path: str, optional: bool = True) -> bool:
    sharded_children = set()
    for name, child in module.named_children():
        curr_path = path + name
        if isinstance(child, ShardedModule):
            sharded_children.add(name)
        else:
            child_sharded = _jit_modules(child, curr_path + ".", optional)
            if child_sharded:
                sharded_children.add(name)

    if len(sharded_children) > 0:
        for name, child in module.named_children():
            if name not in sharded_children:
                try:
                    jit_child = torch.jit.script(child)
                    setattr(module, name, jit_child)
                    logger.info(f"jit.script applied to {path + name}.")
                except Exception as error:
                    if not optional:
                        raise
                    else:
                        logger.info(
                            f"Warning: failed to jit.script {path + name}: {error}."
                        )

    return len(sharded_children) > 0


def _rewrite_model(  # noqa C901
    model: torch.nn.Module,
    context: TrainPipelineContext,
    dist_stream: Optional[torch.cuda.streams.Stream],
    batch: Optional[In] = None,
    apply_jit: bool = False,
    pipelined_forward: Type[BaseForward] = PipelinedForward,
) -> Tuple[List[ShardedModule], torch.nn.Module]:
    input_model = model
    # Get underlying nn.Module
    if isinstance(model, DistributedModelParallel):
        model = model.module

    # Collect a list of sharded modules.
    sharded_modules = {}
    for name, m in model.named_modules():
        if isinstance(m, ShardedModule):
            sharded_modules[name] = m

    # Trace a model.
    concrete_args = {}
    if batch:
        if hasattr(batch, "to_proxy"):
            # for some special models, it requires using "input"
            # as the key for input
            # pyre-ignore[16]: Variable[In (bound to Pipelineable)] has no attribute to_proxy.
            concrete_args["inputs"] = copy.copy(batch).to_proxy()
        elif hasattr(batch, "to_proxy_tuple"):
            # when the model is pre-fx traced or dynamo exported, the
            # inputs are already flattened, and therefore we use
            # tuple as concrete args that fx.trace will automatically
            # match with the argument names.
            # We pass in the model for the caller side to customize
            # the batch
            # pyre-ignore[16]: Variable[In (bound to Pipelineable)] has no attribute to_proxy_tuple.
            concrete_args = batch.to_proxy_tuple(model)

    tracer = Tracer(leaf_modules=_get_leaf_module_names(model))
    graph = tracer.trace(model, concrete_args=concrete_args)

    # Select sharded modules, which are top-level in the forward call graph,
    # i.e. don't have input transformations, i.e. rely only on 'builtins.getattr'.
    pipelined_forwards = []
    for node in graph.nodes:
        if node.op == "call_module" and node.target in sharded_modules:
            total_num_args = len(node.args) + len(node.kwargs)
            if total_num_args == 0:
                continue
            arg_info_list, num_found = _get_node_args(node)

            if num_found == total_num_args:
                logger.info(f"Module '{node.target}'' will be pipelined")
                child = sharded_modules[node.target]
                child.forward = pipelined_forward(
                    node.target,
                    arg_info_list,
                    child,
                    context,
                    dist_stream,
                )
                pipelined_forwards.append(child)

    # JIT script unsharded modules if applicable.
    if apply_jit:
        graph_model = torch.fx.GraphModule(model, graph)
        _jit_modules(graph_model, "")
        if isinstance(input_model, DistributedModelParallel):
            input_model.module = graph_model

    return pipelined_forwards, input_model


def _override_input_dist_forwards(pipelined_modules: List[ShardedModule]) -> None:
    """
    Overrides each input dist forward to support fusing the splits collective.
    NOTE: this can only be called after the input dists are initialized.
    """
    for module in pipelined_modules:
        for child_fqn, child_module in module.named_modules():
            if hasattr(child_module, "_has_uninitialized_input_dist"):
                assert (
                    not child_module._has_uninitialized_input_dist
                ), f"{child_fqn} has uninitialized input dist"

            if not hasattr(child_module, "_input_dists"):
                continue

            for input_dist in child_module._input_dists:
                if hasattr(input_dist, "_dist"):
                    assert isinstance(input_dist._dist, KJTAllToAll)
                    input_dist._dist.forward = KJTAllToAllForward(
                        pg=input_dist._dist._pg,
                        splits=input_dist._dist._splits,
                        stagger=input_dist._dist._stagger,
                    )


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
            self._memcpy_stream: Optional[
                torch.cuda.streams.Stream
            ] = torch.cuda.Stream(priority=-1)
            self._data_dist_stream: Optional[
                torch.cuda.streams.Stream
            ] = torch.cuda.Stream(priority=-1)
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
                # print("batch from data_iter", batch)
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
            self._prefetch_stream: Optional[
                torch.cuda.streams.Stream
            ] = torch.cuda.Stream()
            self._default_stream: Optional[
                torch.cuda.streams.Stream
            ] = torch.cuda.current_stream()
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
                    self._context.module_contexts_post_prefetch[
                        forward._name
                    ] = self._context.module_contexts[forward._name]
