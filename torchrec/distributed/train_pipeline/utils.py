#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from itertools import chain
from threading import Event, Thread
from typing import (
    Any,
    Callable,
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

from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.fx.immutable_collections import immutable_dict as fx_immutable_dict
from torch.fx.node import Node
from torch.profiler import record_function
from torchrec.distributed.dist_data import KJTAllToAll, KJTAllToAllTensorsAwaitable
from torchrec.distributed.embedding_sharding import (
    FusedKJTListSplitsAwaitable,
    KJTListSplitsAwaitable,
    KJTSplitsAllToAllMeta,
)
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.model_parallel import DistributedModelParallel, ShardedModule

from torchrec.distributed.types import Awaitable, LazyNoWait

from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
from torchrec.streamable import Multistreamable, Pipelineable

logger: logging.Logger = logging.getLogger(__name__)

import torch

In = TypeVar("In", bound=Pipelineable)
StageOut = TypeVar("StageOut", bound=Pipelineable)
Out = TypeVar("Out")

RunnableType = Callable[..., StageOut]
StageOutputWithEvent = Tuple[Optional[StageOut], Optional[torch.Event]]


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
            from the input dist for the next batch. (only for version 0)
        fused_splits_awaitables (List[Tuple[List[str], FusedKJTListSplitsAwaitable]]):
            List of fused splits input dist awaitable and the corresponding module names
            of each awaitable.
        event: Optional[torch.cuda.Event]: Event to record the completion of this stage
        index: Optional[int]: Index of the current batch.
        version: int = 0; support for backward compatiblity
    """

    # pyre-ignore [4]
    input_dist_splits_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    # pyre-ignore [4]
    input_dist_tensors_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    module_contexts: Dict[str, Multistreamable] = field(default_factory=dict)
    module_contexts_next_batch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )  # deprecated: to support legacy code
    fused_splits_awaitables: List[Tuple[List[str], FusedKJTListSplitsAwaitable]] = (
        field(default_factory=list)
    )
    events: List[torch.Event] = field(default_factory=list)
    preproc_fwd_results: Dict[str, Any] = field(default_factory=dict)
    index: Optional[int] = None
    version: int = (
        0  # 1 is current version, 0 is deprecated but supported for backward compatibility
    )


@dataclass
class PrefetchTrainPipelineContext(TrainPipelineContext):
    module_input_post_prefetch: Dict[str, Multistreamable] = field(default_factory=dict)
    module_contexts_post_prefetch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )
    module_input_post_prefetch_next_batch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )
    module_contexts_post_prefetch_next_batch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )


@dataclass
class EmbeddingTrainPipelineContext(TrainPipelineContext):
    embedding_a2a_requests: Dict[str, Multistreamable] = field(default_factory=dict)
    embedding_tensors: List[List[torch.Tensor]] = field(default_factory=list)
    detached_embedding_tensors: List[List[torch.Tensor]] = field(default_factory=list)


@dataclass
class PipelineStage:
    """
    A pipeline stage represents a transform to an input that is independent of the
    backwards() of the model. Examples include batch H2D transfer, GPU preproc, or
    gradient-less model processing.

    Args:
        name (str): Name of the stage.
        runnable (Callable[In, Out]): Function that performs a gradient-less
            transform.
        stream (torch.cuda.streams.Stream): Stream to run on. Often each stage has a
            unique stream, but having different pipelines share a stream provides more
            synchronization semantics.
    """

    name: str
    runnable: RunnableType
    stream: torch.Stream
    fill_callback: Optional[Callable[[], None]] = None


@dataclass
class ArgInfo:
    """
    Representation of args from a node.

    Attributes:
        input_attrs (List[str]): attributes of input batch,
            e.g. `batch.attr1.attr2` will produce ["attr1", "attr2"].
        is_getitems (List[bool]): `batch[attr1].attr2` will produce [True, False].
        preproc_modules (List[Optional[PipelinedPreproc]]): list of torch.nn.Modules that
            transform the input batch.
        constants: constant arguments that are passed to preproc modules.
        name (Optional[str]): name for kwarg of pipelined forward() call or None for a
            positional arg.
    """

    input_attrs: List[str]
    is_getitems: List[bool]
    # recursive dataclass as preproc_modules.args -> arginfo.preproc_modules -> so on
    preproc_modules: List[Optional["PipelinedPreproc"]]
    constants: List[Optional[object]]
    name: Optional[str]


# pyre-ignore
def _build_args_kwargs(
    # pyre-ignore
    initial_input: Any,
    fwd_args: List[ArgInfo],
) -> Tuple[List[Any], Dict[str, Any]]:
    args = []
    kwargs = {}
    for arg_info in fwd_args:
        if arg_info.input_attrs:
            arg = initial_input
            for attr, is_getitem, preproc_mod, obj in zip(
                arg_info.input_attrs,
                arg_info.is_getitems,
                arg_info.preproc_modules,
                arg_info.constants,
            ):
                if obj is not None:
                    arg = obj
                    break
                elif preproc_mod is not None:
                    # preproc will internally run the same logic recursively
                    # if its args are derived from other preproc modules
                    # we can get all inputs to preproc mod based on its recorded args_info + arg passed to it
                    arg = preproc_mod(arg)
                else:
                    if is_getitem:
                        arg = arg[attr]
                    elif attr != "":
                        arg = getattr(arg, attr)
                    else:
                        # neither is_getitem nor valid attr, no-op
                        arg = arg
            if arg_info.name:
                kwargs[arg_info.name] = arg
            else:
                args.append(arg)
        else:
            args.append(None)
    return args, kwargs


class PipelinedPreproc(torch.nn.Module):
    """
    Wrapper around preproc module found during model graph traversal for sparse data dist
    pipelining. In addition to the original module, it encapsulates information needed for
    execution such as list of ArgInfo and the current training pipeline context.

    Args:
        preproc_module (torch.nn.Module): preproc module to run
        fqn (str): fqn of the preproc module in the model being pipelined
        args (List[ArgInfo]): list of ArgInfo for the preproc module
        context (TrainPipelineContext): Training context for the next iteration / batch

    Returns:
        Any

    Example:
        preproc = PipelinedPreproc(preproc_module, fqn, args, context)
        # module-swap with pipeliend preproc
        setattr(model, fqn, preproc)
    """

    def __init__(
        self,
        preproc_module: torch.nn.Module,
        fqn: str,
        args: List[ArgInfo],
        context: TrainPipelineContext,
    ) -> None:
        super().__init__()
        self._preproc_module = preproc_module
        self._fqn = fqn
        self._args_list: List[List[ArgInfo]] = [args]
        self._context = context
        self._call_idx = 0

    @property
    def preproc_module(self) -> torch.nn.Module:
        return self._preproc_module

    @property
    def fqn(self) -> str:
        return self._fqn

    def register_args(self, args: List[ArgInfo]) -> None:
        self._args_list.append(args)

    # pyre-ignore
    def forward(self, *input, **kwargs) -> Any:
        """
        Args:
            Any args and kwargs during model fwd
            During _start_data_dist, input[0] contains the current data
        Returns:
            Any
        """
        cache_key = self._fqn + str(self._call_idx)
        if cache_key in self._context.preproc_fwd_results:
            # This should only be hit in two cases:
            # 1) During model forward
            # During model forward, avoid duplicate work
            # by returning the cached result from previous
            # iteration's _start_data_dist
            # 2) During _start_data_dist when preproc module is
            # shared by more than one args. e.g. if we have
            # preproc_out_a = preproc_a(input)
            # preproc_out_b = preproc_b(preproc_out_a) <- preproc_a shared
            # preproc_out_c = preproc_c(preproc_out_a) <-^
            # When processing preproc_b, we cache value of preproc_a(input)
            # so when processing preproc_c, we can reuse preproc_a(input)
            res = self._context.preproc_fwd_results[cache_key]
            self._call_idx += 1
            return res

        # Everything below should only be called during _start_data_dist stage

        # Build up arg and kwargs from recursive call to pass to preproc module
        # Arguments to preproc module can be also be a derived product
        # of another preproc module call, as long as module is pipelineable

        # Use input[0] as _start_data_dist only passes 1 arg
        args, kwargs = _build_args_kwargs(input[0], self._args_list[self._call_idx])

        with record_function(f"## sdd_input_preproc {self._context.index} ##"):
            res = self._preproc_module(*args, **kwargs)
            # Cache results, only during _start_data_dist
            self._context.preproc_fwd_results[cache_key] = res
            self._call_idx += 1
            return res

    @property
    def args(self) -> List[ArgInfo]:
        return self._args

    def set_context(self, context: TrainPipelineContext) -> None:
        self._context = context
        # reset call index if multiple calls to this module
        self._call_idx = 0

    def get_context(self) -> TrainPipelineContext:
        return self._context


class BaseForward:
    def __init__(
        self,
        name: str,
        args: List[ArgInfo],
        module: ShardedModule,
        context: TrainPipelineContext,
        stream: Optional[torch.Stream] = None,
    ) -> None:
        self._name = name
        self._args = args
        self._module = module
        self._context = context
        self._stream = stream
        self._device: torch.device = stream.device if stream else torch.device("cuda")

    @property
    def name(self) -> str:
        return self._name

    @property
    def args(self) -> List[ArgInfo]:
        return self._args

    def set_context(self, context: TrainPipelineContext) -> None:
        self._context = context

    def get_context(self) -> TrainPipelineContext:
        return self._context


class PipelinedForward(BaseForward):
    # pyre-ignore [2, 24]
    def __call__(self, *input, **kwargs) -> Awaitable:
        assert (
            self._name in self._context.input_dist_tensors_requests
        ), "Invalid PipelinedForward usage, please do not directly call model.forward()"
        request = self._context.input_dist_tensors_requests.pop(self._name)
        assert isinstance(request, Awaitable)
        with record_function("## wait_sparse_data_dist ##"):
            # Finish waiting on the dist_stream,
            # in case some delayed stream scheduling happens during the wait() call.
            with torch.get_device_module(self._device).stream(self._stream):
                data = request.wait()

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        ctx = self._context.module_contexts.pop(self._name)

        if self._stream is not None:
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._stream
            )
            cur_stream = torch.get_device_module(self._device).current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            data.record_stream(cur_stream)
            ctx.record_stream(cur_stream)

        return self._module.compute_and_output_dist(ctx, data)


class EmbeddingPipelinedForward(BaseForward):
    # pyre-ignore [2, 24]
    def __call__(self, *input, **kwargs) -> Awaitable:
        assert (
            self._name
            # pyre-ignore [16]
            in self._context.embedding_a2a_requests
        ), "Invalid EmbeddingPipelinedForward usage, please do not directly call model.forward()"

        ctx = self._context.module_contexts.pop(self._name)
        if self._stream is not None:
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._stream
            )
            cur_stream = torch.get_device_module(self._device).current_stream()
            ctx.record_stream(cur_stream)
        awaitable = self._context.embedding_a2a_requests.pop(self._name)
        embeddings = awaitable.wait()  # trigger awaitable manually for type checking
        tensors = []
        detached_tensors = []
        if isinstance(embeddings, Dict):
            for jt in embeddings.values():
                assert isinstance(jt, JaggedTensor)
                tensor = jt.values()
                detached_tensor = tensor.detach().requires_grad_()
                detached_tensor.retain_grad()
                jt._values = detached_tensor
                tensors.append(tensor)
                detached_tensors.append(detached_tensor)
            # pyre-ignore [16]
            self._context.embedding_tensors.append(tensors)
            # pyre-ignore [16]
            self._context.detached_embedding_tensors.append(detached_tensors)
        else:
            assert isinstance(embeddings, KeyedTensor)
            tensor = embeddings.values()
            detached_tensor = tensor.detach().requires_grad_()
            detached_tensor.retain_grad()
            embeddings._values = detached_tensor
            tensors.append(tensor)
            detached_tensors.append(detached_tensor)
            self._context.embedding_tensors.append(tensors)
            self._context.detached_embedding_tensors.append(detached_tensors)

        return LazyNoWait(embeddings)


class PrefetchPipelinedForward(BaseForward):
    def __init__(
        self,
        name: str,
        args: List[ArgInfo],
        module: ShardedModule,
        context: PrefetchTrainPipelineContext,
        prefetch_stream: Optional[torch.Stream] = None,
    ) -> None:
        super().__init__(
            name=name,
            args=args,
            module=module,
            context=context,
            stream=prefetch_stream,
        )

    # pyre-ignore [2, 24]
    def __call__(self, *input, **kwargs) -> Awaitable:
        assert (
            self._name
            # pyre-ignore [16]
            in self._context.module_input_post_prefetch
        ), "Invalid PrefetchPipelinedForward usage, please do not directly call model.forward()"
        data = self._context.module_input_post_prefetch.pop(self._name)
        # pyre-ignore [16]
        ctx = self._context.module_contexts_post_prefetch.pop(self._name)

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        if self._stream is not None:
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._stream
            )
            cur_stream = torch.get_device_module(self._device).current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            data.record_stream(cur_stream)

            ctx.record_stream(cur_stream)

        return self._module.compute_and_output_dist(ctx, data)


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


def _to_device(batch: In, device: torch.device, non_blocking: bool) -> In:
    assert isinstance(
        batch, (torch.Tensor, Pipelineable)
    ), f"{type(batch)} must implement Pipelineable interface"
    return cast(In, batch.to(device=device, non_blocking=non_blocking))


def _wait_for_batch(batch: In, stream: Optional[torch.Stream]) -> None:
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
    if stream is None:
        return

    device = stream.device
    torch.get_device_module(device).current_stream().wait_stream(stream)
    cur_stream = torch.get_device_module(device).current_stream()
    assert isinstance(
        batch, (torch.Tensor, Multistreamable)
    ), f"{type(batch)} must implement Multistreamable interface"
    batch.record_stream(cur_stream)


def _wait_for_events(
    batch: In,
    context: TrainPipelineContext,
    stream: Optional[torch.Stream],
) -> None:
    """
    Wait for any outstanding events for a given context
    """

    for event in context.events:
        event.wait()
    context.events.clear()
    if stream:
        assert isinstance(
            batch, (torch.Tensor, Multistreamable)
        ), f"{type(batch)} must implement Multistreamable interface"
        batch.record_stream(stream)


def _start_data_dist(
    pipelined_modules: List[ShardedModule],
    batch: In,
    context: TrainPipelineContext,
) -> None:
    if context.version == 0:
        context.input_dist_splits_requests.clear()
        context.module_contexts_next_batch.clear()
        context.fused_splits_awaitables.clear()

    for module in pipelined_modules:
        forward = module.forward
        assert (
            isinstance(forward, PipelinedForward)
            or isinstance(forward, PrefetchPipelinedForward)
            or isinstance(forward, EmbeddingPipelinedForward)
        )

        # Retrieve argument for the input_dist of EBC
        # is_getitem True means this argument could be retrieved by a list
        # False means this argument is getting while getattr
        # and this info was done in the _rewrite_model by tracing the
        # entire model to get the arg_info_list
        args, kwargs = _build_args_kwargs(batch, forward.args)

        # Start input distribution.
        module_ctx = module.create_context()
        if context.version == 0:
            context.module_contexts_next_batch[forward.name] = module_ctx
        else:
            context.module_contexts[forward.name] = module_ctx
        context.input_dist_splits_requests[forward.name] = module.input_dist(
            module_ctx, *args, **kwargs
        )
    _fuse_input_dist_splits(context)


def _start_embedding_lookup(
    module: ShardedModule,
    batch: In,  # not used in this function
    context: EmbeddingTrainPipelineContext,
    stream: Optional[torch.Stream],
) -> None:
    kjt = context.input_dist_tensors_requests[module.forward.name].wait()
    module_context = context.module_contexts[module.forward.name]
    if stream:
        kjt.record_stream(stream)
        module_context.record_stream(stream)
    a2a_awaitable = module.compute_and_output_dist(module_context, kjt)
    # pyre-ignore[6]
    context.embedding_a2a_requests[module.forward.name] = a2a_awaitable


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
                        (
                            context.module_contexts_next_batch[name]
                            if context.version == 0
                            else context.module_contexts[name]
                        )
                        for name in names
                    ],
                    pg=pg,
                ),
            )
        )


def _check_args_for_call_module(
    node: torch.fx.Node,
) -> bool:
    """
    Recursively checks if args to a node is the result of a call_module.
    """
    if node.op == "call_module":
        return True

    for arg in node.args:
        if isinstance(arg, torch.fx.Node) and _check_args_for_call_module(arg):
            return True

    return False


def _check_preproc_pipelineable(
    module: torch.nn.Module,
) -> bool:
    for _, _ in module.named_parameters(recurse=True):
        # Cannot have any trainable params for it to be pipelined
        logger.warning(
            f"Module {module} cannot be pipelined as it has trainable parameters"
        )
        return False
    return True


def _find_preproc_module_recursive(
    module: torch.nn.Module,
    preproc_module_fqn: str,
) -> Optional[torch.nn.Module]:
    """
    Finds the preproc module in the model.
    """
    for name, child in module.named_modules():
        if name == preproc_module_fqn:
            return child
    return None


def _swap_preproc_module_recursive(
    module: torch.nn.Module,
    to_swap_module: torch.nn.Module,
    preproc_module_fqn: str,
    path: str = "",
) -> torch.nn.Module:
    """
    Swaps the preproc module in the model.
    """
    if isinstance(module, PipelinedPreproc):
        return module

    if path == preproc_module_fqn:
        return to_swap_module

    for name, child in module.named_children():
        child = _swap_preproc_module_recursive(
            child,
            to_swap_module,
            preproc_module_fqn,
            path + "." + name if path else name,
        )
        setattr(module, name, child)

    return module


def _get_node_args_helper(
    model: torch.nn.Module,
    # pyre-ignore
    arguments,
    num_found: int,
    pipelined_preprocs: Set[PipelinedPreproc],
    context: TrainPipelineContext,
    pipeline_preproc: bool,
) -> Tuple[List[ArgInfo], int]:
    """
    Goes through the args/kwargs of a node and arranges them into a list of `ArgInfo`s.
    It also counts the number of (args + kwargs) found.
    """
    arg_info_list = [ArgInfo([], [], [], [], None) for _ in range(len(arguments))]
    for arg, arg_info in zip(arguments, arg_info_list):
        if arg is None:
            num_found += 1
            continue
        while True:
            if not isinstance(arg, torch.fx.Node):
                if pipeline_preproc:
                    if isinstance(arg, fx_immutable_dict):
                        arg_info.input_attrs.insert(0, "")
                        arg_info.is_getitems.insert(0, False)
                        arg_info.preproc_modules.insert(0, None)
                        arg_info.constants.insert(0, arg.copy())
                        num_found += 1
                break

            child_node = arg

            if child_node.op == "placeholder":
                if hasattr(child_node, "ph_key"):
                    # pyre-ignore[16]
                    ph_key: str = child_node.ph_key
                    # example: ph_key = 'event_id_list_features_seqs[marketplace]'
                    ph_keys = ph_key.split("[")
                    for key in ph_keys:
                        if "]" in key:
                            arg_info.input_attrs.append(key[:-1])
                            arg_info.is_getitems.append(True)
                        else:
                            arg_info.input_attrs.append(key)
                            arg_info.is_getitems.append(False)
                        arg_info.preproc_modules.append(None)
                        arg_info.constants.append(None)
                else:
                    # no-op
                    arg_info.input_attrs.insert(0, "")
                    arg_info.is_getitems.insert(0, False)
                    arg_info.preproc_modules.insert(0, None)
                    arg_info.constants.insert(0, None)

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
                arg_info.preproc_modules.insert(0, None)
                arg_info.constants.insert(0, None)
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
                arg_info.preproc_modules.insert(0, None)
                arg_info.constants.insert(0, None)
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
                call_module_found = False

                for arg_node in chain(child_node.args, child_node.kwargs.values()):
                    if isinstance(
                        arg_node, torch.fx.Node
                    ) and _check_args_for_call_module(arg_node):
                        call_module_found = True
                        break

                if call_module_found:
                    break

                if "values" in child_node.kwargs:
                    arg = child_node.kwargs["values"]
                else:
                    arg = child_node.args[1]
            elif child_node.op == "call_method" and child_node.target == "get":
                # pyre-ignore[6]
                arg_info.input_attrs.insert(0, child_node.args[1])
                arg_info.is_getitems.insert(0, True)
                arg_info.preproc_modules.insert(0, None)
                arg = child_node.args[0]
            elif child_node.op == "call_module":
                preproc_module_fqn = str(child_node.target)
                preproc_module = _find_preproc_module_recursive(
                    model, preproc_module_fqn
                )

                if not pipeline_preproc:
                    logger.warning(
                        f"Found module {preproc_module} that potentially modifies KJ. Train pipeline initialized with `pipeline_preproc=False` (default), so we assume KJT input modification. To allow torchrec to check if this module can be safely pipelined, please set `pipeline_preproc=True`"
                    )
                    break

                if not preproc_module:
                    # Could not find such module, should not happen
                    break

                if not isinstance(preproc_module, torch.nn.Module):
                    logger.warning(
                        f"Expected preproc_module to be nn.Module but was {type(preproc_module)}"
                    )
                    break

                # check if module is safe to pipeline i.e.no trainable param
                if not _check_preproc_pipelineable(preproc_module):
                    break

                # For module calls, `self` isn't counted
                total_num_args = len(child_node.args) + len(child_node.kwargs)
                if total_num_args == 0:
                    # module call without any args, assume KJT modified
                    break

                # recursive call to check that all inputs to this preproc module
                # is either made of preproc module or non-modifying train batch input
                # transformations
                preproc_args, num_found_safe_preproc_args = _get_node_args(
                    model, child_node, pipelined_preprocs, context, pipeline_preproc
                )
                if num_found_safe_preproc_args == total_num_args:
                    logger.info(
                        f"""Module {preproc_module} is a valid preproc module (no
                        trainable params and inputs can be derived from train batch input
                         via a series of either valid preproc modules or non-modifying
                         transformations) and will be applied during sparse data dist 
                         stage"""
                    )

                    if isinstance(preproc_module, PipelinedPreproc):
                        # Already did module swap, registe args for current call idx
                        preproc_module.register_args(preproc_args)

                        arg_info.input_attrs.insert(0, "")  # dummy value
                        arg_info.is_getitems.insert(0, False)
                        pipelined_preprocs.add(preproc_module)
                        arg_info.preproc_modules.insert(0, preproc_module)
                        arg_info.constants.insert(0, None)
                        num_found += 1
                        break

                    pipelined_preproc_module = PipelinedPreproc(
                        preproc_module,
                        preproc_module_fqn,
                        preproc_args,
                        context,
                    )

                    # module swap
                    _swap_preproc_module_recursive(
                        model, pipelined_preproc_module, preproc_module_fqn
                    )

                    arg_info.input_attrs.insert(0, "")  # dummy value
                    arg_info.is_getitems.insert(0, False)
                    pipelined_preprocs.add(pipelined_preproc_module)
                    arg_info.preproc_modules.insert(0, pipelined_preproc_module)
                    arg_info.constants.insert(0, None)

                    num_found += 1

                # we cannot set any other `arg` value here
                # break to avoid infinite loop
                break
            else:
                break
    return arg_info_list, num_found


def _get_node_args(
    model: torch.nn.Module,
    node: Node,
    pipelined_preprocs: Set[PipelinedPreproc],
    context: TrainPipelineContext,
    pipeline_preproc: bool,
) -> Tuple[List[ArgInfo], int]:
    num_found = 0

    pos_arg_info_list, num_found = _get_node_args_helper(
        model,
        node.args,
        num_found,
        pipelined_preprocs,
        context,
        pipeline_preproc,
    )
    kwargs_arg_info_list, num_found = _get_node_args_helper(
        model,
        node.kwargs.values(),
        num_found,
        pipelined_preprocs,
        context,
        pipeline_preproc,
    )

    # Replace with proper names for kwargs
    for name, arg_info_list in zip(node.kwargs, kwargs_arg_info_list):
        arg_info_list.name = name

    arg_info_list = pos_arg_info_list + kwargs_arg_info_list

    return (arg_info_list, num_found)


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


def _pipeline_detach_model(
    pipelined_modules: List[ShardedModule],
    # pyre-ignore[2]
    original_forwards: List[Callable[..., Any]],
    original_kjt_dist_forwards: List[
        Callable[[KeyedJaggedTensor], Awaitable[KJTAllToAllTensorsAwaitable]]
    ],
) -> None:
    kjt_dists = []
    for mod, original_fwd in zip(pipelined_modules, original_forwards):
        # pyre-ignore
        mod.forward = original_fwd

        for _, child_module in mod.named_modules():
            if not hasattr(child_module, "_input_dists"):
                continue
            for input_dist in child_module._input_dists:
                if hasattr(input_dist, "_dist"):
                    kjt_dists.append(input_dist._dist)
    assert len(kjt_dists) == len(
        original_kjt_dist_forwards
    ), f"Number of KJT dists ({len(kjt_dists)}) does not match number of kjt dist forwards provided ({len(original_kjt_dist_forwards)})"

    for kjt_dist, original_kjt_dist_fwd in zip(
        kjt_dists,
        original_kjt_dist_forwards,
    ):
        kjt_dist.forward = original_kjt_dist_fwd


# pyre-ignore[3]
def _rewrite_model(  # noqa C901
    model: torch.nn.Module,
    context: TrainPipelineContext,
    dist_stream: Optional[torch.Stream],
    batch: Optional[In] = None,
    apply_jit: bool = False,
    pipelined_forward: Type[BaseForward] = PipelinedForward,
    pipeline_preproc: bool = False,
) -> Tuple[
    List[ShardedModule],
    torch.nn.Module,
    List[Callable[..., Any]],
    List[PipelinedPreproc],
]:
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
    original_forwards = []

    pipelined_preprocs: Set[PipelinedPreproc] = set()

    for node in graph.nodes:
        if node.op == "call_module" and node.target in sharded_modules:
            total_num_args = len(node.args) + len(node.kwargs)
            if total_num_args == 0:
                continue
            arg_info_list, num_found = _get_node_args(
                model,
                node,
                pipelined_preprocs,
                context,
                pipeline_preproc,
            )

            if num_found == total_num_args:
                logger.info(f"Module '{node.target}' will be pipelined")
                child = sharded_modules[node.target]
                original_forwards.append(child.forward)
                child.forward = pipelined_forward(
                    node.target,
                    arg_info_list,
                    child,
                    context,
                    dist_stream,
                )
                pipelined_forwards.append(child)
            else:
                logger.warning(
                    f"Module '{node.target}'' will not be pipelined, due to input modifications"
                )

    # JIT script unsharded modules if applicable.
    if apply_jit:
        graph_model = torch.fx.GraphModule(model, graph)
        _jit_modules(graph_model, "")
        if isinstance(input_model, DistributedModelParallel):
            input_model.module = graph_model

    return pipelined_forwards, input_model, original_forwards, list(pipelined_preprocs)


def _override_input_dist_forwards(
    pipelined_modules: List[ShardedModule],
) -> List[Callable[[KeyedJaggedTensor], Awaitable[KJTAllToAllTensorsAwaitable]]]:
    """
    Overrides each input dist forward to support fusing the splits collective.
    NOTE: this can only be called after the input dists are initialized.
    """
    original_kjt_dist_forwards = []
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
                    original_kjt_dist_forwards.append(input_dist._dist.forward)
                    input_dist._dist.forward = KJTAllToAllForward(
                        pg=input_dist._dist._pg,
                        splits=input_dist._dist._splits,
                        stagger=input_dist._dist._stagger,
                    )
    return original_kjt_dist_forwards


def get_h2d_func(batch: In, device: torch.device) -> Pipelineable:
    return batch.to(device, non_blocking=True)


class DataLoadingThread(Thread, Generic[In]):
    def __init__(
        self,
        device: torch.device,
        dataloader_iter: Iterator[In],
        to_device_non_blocking: bool,
        memcpy_stream_priority: int = 0,
    ) -> None:
        super().__init__()
        self._stop: bool = False
        self._dataloader_iter = dataloader_iter
        self._buffer_empty_event: Event = Event()
        self._buffer_filled_event: Event = Event()
        self._memcpy_stream: Optional[torch.Stream] = (
            torch.get_device_module(device).Stream(priority=memcpy_stream_priority)
            if device.type in ["cuda", "mtia"]
            else None
        )
        self._device = device
        self._to_device_non_blocking = to_device_non_blocking
        self._buffered: Optional[In] = None
        self._buffer_empty_event.set()

    def run(self) -> None:
        if self._device.type == "cuda" and torch.cuda.is_available():
            # set the current device the same as the one used in the main thread
            torch.cuda.set_device(self._device)
        elif self._device.type == "mtia" and torch.mtia.is_available():
            # set the current device the same as the one used in the main thread
            torch.mtia.set_device(self._device)

        while not self._stop:
            self._buffer_empty_event.wait()
            # Set the filled event to unblock progress() and return.
            if self._stop:
                self._buffer_filled_event.set()
                return
            with record_function("## load_batch ##"):
                try:
                    batch = next(self._dataloader_iter)
                except StopIteration:
                    self._stop = True
                    self._buffer_filled_event.set()
                    return
            with record_function("## copy_batch_to_gpu ##"):
                with torch.get_device_module(self._device).stream(self._memcpy_stream):
                    self._buffered = cast(
                        In,
                        batch.to(
                            self._device, non_blocking=self._to_device_non_blocking
                        ),
                    )
                self._buffer_empty_event.clear()
                self._buffer_filled_event.set()

    def stop(self) -> None:
        logger.info("Stopping data loading thread...")
        self._stop = True
        # Unblock any thread that are waiting for these events.
        self._buffer_filled_event.set()
        self._buffer_empty_event.set()
        logger.info("Data loading thread stopped.")

    def get_next_batch(self, none_throws: bool = False) -> Optional[In]:
        """
        Get the next batch from the buffer if threading is enabled, otherwise
        call load_next_batch directly.

        This function is not thread safe. We assume this is only invoked from
        the main thread in the training loop.
        """
        self._buffer_filled_event.wait()
        batch = self._buffered
        if batch is None:
            if none_throws:
                raise StopIteration
            return None
        self._buffered = None
        self._buffer_filled_event.clear()
        self._buffer_empty_event.set()
        return batch


def _prefetch_embeddings(
    batch: In,
    context: PrefetchTrainPipelineContext,
    pipelined_modules: List[ShardedModule],
    device: torch.device,
    stream_context: torch.Stream,
    data_dist_stream: Optional[torch.Stream],
    default_stream: Optional[torch.Stream],
) -> Dict[str, KJTList]:
    data_per_sharded_module = {}
    for sharded_module in pipelined_modules:
        forward = sharded_module.forward
        assert isinstance(forward, PrefetchPipelinedForward)

        assert forward._name in context.input_dist_tensors_requests
        request = context.input_dist_tensors_requests.pop(forward._name)
        assert isinstance(request, Awaitable)
        with record_function("## wait_sparse_data_dist ##"):
            # Finish waiting on the dist_stream,
            # in case some delayed stream scheduling happens during the wait() call.
            with stream_context(data_dist_stream):
                data = request.wait()

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        module_context = context.module_contexts[forward._name]
        if data_dist_stream is not None:
            torch.get_device_module(device).current_stream().wait_stream(
                data_dist_stream
            )
            cur_stream = torch.get_device_module(device).current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            data.record_stream(cur_stream)
            data.record_stream(default_stream)

            module_context.record_stream(cur_stream)
            module_context.record_stream(default_stream)

        sharded_module.prefetch(
            ctx=module_context,
            dist_input=data,
            forward_stream=default_stream,
        )
        data_per_sharded_module[forward._name] = data
    return data_per_sharded_module


class SparseDataDistUtil(Generic[In]):
    """
    Helper class exposing methods for sparse data dist and prefetch pipelining.
    Currently used for `StagedTrainPipeline` pipeline stages

    Args:
        model (torch.nn.Module): Model to pipeline
        data_dist_stream (torch.cuda.Stream): Stream on which to run sparse data dist.
        apply_jit (bool): apply torch.jit.script to non-pipelined (unsharded) modules.
        prefetch_stream (Optional[torch.cuda.Stream]): Stream on which model prefetch runs
            Defaults to `None`. This needs to be passed in to enable prefetch pipelining.

    Example::
        sdd = SparseDataDistUtil(
            model=model,
            data_dist_stream=torch.cuda.Stream(),
            prefetch_stream=torch.cuda.Stream(), <-- required to enable prefetch pipeline
        )
        pipeline = [
            PipelineStage(
                name="data_copy",
                runnable=lambda batch, context: batch.to(
                    self._device, non_blocking=True
                ),
                stream=torch.cuda.Stream(),
            ),
            PipelineStage(
                name="start_sparse_data_dist",
                runnable=sdd.start_sparse_data_dist,
                stream=sdd.data_dist_stream,
                fill_callback=sdd.wait_sparse_data_dist,
            ),
            PipelineStage(
                name="prefetch",
                runnable=sdd.prefetch,
                stream=sdd.prefetch_stream,
                fill_callback=sdd.load_prefetch,
            ),
        ]

        return StagedTrainPipeline(pipeline_stages=pipeline)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data_dist_stream: torch.Stream,
        apply_jit: bool = False,
        prefetch_stream: Optional[torch.Stream] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.data_dist_stream = data_dist_stream
        self.prefetch_stream = prefetch_stream
        self.apply_jit = apply_jit
        self.context = (
            PrefetchTrainPipelineContext(version=0)
            if prefetch_stream
            else TrainPipelineContext(version=0)
        )
        self.initialized = False
        self._pipelined_modules: List[ShardedModule] = []
        # pyre-ignore
        self.fwd_hook = None
        self._device: torch.device = data_dist_stream.device

        # pyre-ignore
        self._original_forwards: List[Callable[..., Any]] = []
        self._original_kjt_dist_forwards: List[
            Callable[[KeyedJaggedTensor], Awaitable[KJTAllToAllTensorsAwaitable]]
        ] = []

        self._pipelined_forward = (
            PrefetchPipelinedForward if prefetch_stream else PipelinedForward
        )

        self._default_stream: Optional[torch.Stream] = (
            (torch.get_device_module(self._device).Stream())
            if self._device.type in ["cuda", "mtia"]
            else None
        )

    def detach(self) -> torch.nn.Module:
        """
        Removes sparse data dist (SDD) pipelining from model forward and input dist.
        Modifies existing model in place and returns the model.

        detach() can be called at any point, and inflight batches do not need to be
        flushed before calling it. Calling pipeline.progress() will re-attach the model
        to the pipeline and the pipeline will progress normally from the point it was detached (i.e. inflight batches will be kept when calling detach).

        While the model is detached, it is equivalent to the model before passing to
        the pipeline, so forward and backward passes, and optimizer updates can be
        carried out normally.
        """
        if self.initialized:
            assert self.fwd_hook is not None
            self.fwd_hook.remove()

            _pipeline_detach_model(
                pipelined_modules=self._pipelined_modules,
                original_forwards=self._original_forwards,
                original_kjt_dist_forwards=self._original_kjt_dist_forwards,
            )

        self.initialized = False
        return self.model

    def start_sparse_data_dist(self, batch: In) -> In:
        if not self.initialized:
            # Step 1: Pipeline input dist in trec sharded modules
            # TODO (yhshin): support preproc modules for `StagedTrainPipeline`
            self._pipelined_modules, self.model, self._original_forwards, _ = (
                _rewrite_model(
                    model=self.model,
                    context=self.context,
                    dist_stream=self.data_dist_stream,
                    batch=batch,
                    apply_jit=self.apply_jit,
                    pipelined_forward=self._pipelined_forward,
                )
            )
            # initializes input dist, so we can override input dist forwards
            _start_data_dist(self._pipelined_modules, batch, self.context)
            self._original_kjt_dist_forwards = _override_input_dist_forwards(
                self._pipelined_modules
            )

            # Step 2: Register post-forward hook to wait SDD
            def forward_hook(
                module: torch.nn.Module,
                input: Union[torch.Tensor, Tuple[torch.Tensor]],
                output: Union[torch.Tensor, Tuple[torch.Tensor]],
            ) -> None:
                if self.prefetch_stream is not None:
                    # Need to load prefetch before wait_sparse_data_dist
                    self.load_prefetch()
                self.wait_sparse_data_dist()

            self.fwd_hook = self.model.register_forward_hook(forward_hook)

            self.initialized = True

        _start_data_dist(self._pipelined_modules, batch, self.context)

        return batch

    def wait_sparse_data_dist(self) -> None:
        with record_function("## wait_sparse_data_dist ##"):
            with torch.get_device_module(self._device).stream(self.data_dist_stream):
                self.context.module_contexts = (
                    self.context.module_contexts_next_batch.copy()
                )
                self.context.input_dist_tensors_requests.clear()
                for names, awaitable in self.context.fused_splits_awaitables:
                    for name, request in zip(names, awaitable.wait()):
                        self.context.input_dist_tensors_requests[name] = request

    def prefetch(self, batch: In) -> In:
        """
        Waits for input dist to finish, then prefetches data.
        """
        assert isinstance(
            self.context, PrefetchTrainPipelineContext
        ), "Pass prefetch_stream into SparseDataDistUtil to use prefetch() as a stage"
        self.context.module_input_post_prefetch_next_batch.clear()
        # pyre-ignore
        self.context.module_contexts_post_prefetch_next_batch.clear()

        data_per_pipelined_module = _prefetch_embeddings(
            batch,
            # pyre-ignore
            self.context,
            self._pipelined_modules,
            self._device,
            torch.get_device_module(self._device).stream,
            self.data_dist_stream,
            self._default_stream,
        )
        for sharded_module in self._pipelined_modules:
            forward = sharded_module.forward
            data = data_per_pipelined_module[forward._name]
            # pyre-ignore [16]
            self.context.module_input_post_prefetch_next_batch[forward._name] = data
            self.context.module_contexts_post_prefetch_next_batch[forward._name] = (
                self.context.module_contexts.pop(forward._name)
            )
        return batch

    def load_prefetch(self) -> None:
        assert isinstance(
            self.context, PrefetchTrainPipelineContext
        ), "Pass prefetch_stream into SparseDataDistUtil to use load_prefetch()"
        self.context.module_input_post_prefetch.clear()
        # pyre-ignore
        self.context.module_contexts_post_prefetch.clear()

        with record_function("## load_sharded_module_prefetch ##"):
            with torch.get_device_module(self._device).stream(self.prefetch_stream):
                for sharded_module in self._pipelined_modules:
                    forward = sharded_module.forward
                    assert isinstance(forward, PrefetchPipelinedForward)
                    self.context.module_input_post_prefetch[forward._name] = (
                        self.context.module_input_post_prefetch_next_batch[
                            forward._name
                        ]
                    )
                    self.context.module_contexts_post_prefetch[forward._name] = (
                        self.context.module_contexts_post_prefetch_next_batch[
                            forward._name
                        ]
                    )
