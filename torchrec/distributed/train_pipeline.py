#!/usr/bin/env python3

import abc
import logging
from dataclasses import dataclass, field
from typing import (
    Iterator,
    Tuple,
    Optional,
    TypeVar,
    Generic,
    cast,
    Any,
    Dict,
    List,
)

import torch
from torch.autograd.profiler import record_function
from torch.fx.node import Node
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.model_parallel import DistributedModelParallel, ShardedModule
from torchrec.distributed.types import (
    Awaitable,
    ShardedModuleContext,
)
from torchrec.distributed.utils import get_unsharded_module_names

logger: logging.Logger = logging.getLogger(__name__)


class PipelinedInput(abc.ABC):
    """
    This interface contains two methods, one for moving an input to GPU, the other
    one for synchronizing the moving in a CUDA stream with other CUDA streams.

    torch.Tensor implements this interface so we can used it in most applications.
    """

    @abc.abstractmethod
    def to(self, device: torch.device, non_blocking: bool) -> "PipelinedInput":
        ...

    @abc.abstractmethod
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        ...


In = TypeVar("In", bound=PipelinedInput)
Out = TypeVar("Out")
DistIn = TypeVar("DistIn")
DistOut = TypeVar("DistOut")


class TrainPipeline(abc.ABC, Generic[In, Out]):
    @abc.abstractmethod
    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        pass


def _to_device(batch: In, device: torch.device, non_blocking: bool) -> In:
    return cast(In, batch.to(device=device, non_blocking=non_blocking))


def _wait_for_batch(batch: In, stream: Optional[torch.cuda.streams.Stream]) -> None:
    if stream is None:
        return
    torch.cuda.current_stream().wait_stream(stream)
    # since these tensors were allocated on a different stream,
    # and are going to be used
    # on the current, let caching allocator know
    cur_stream = torch.cuda.current_stream()

    batch.record_stream(cur_stream)


class TrainPipelineBase(TrainPipeline[In, Out]):
    """
    Performs a single step of a training loop
    while doing CPU -> GPU transfer of inputs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self._model: torch.nn.Module = model
        self._optimizer = optimizer
        self._device = device
        if str(self._device) == "cpu":
            self._memcpy_stream: Optional[torch.cuda.streams.Stream] = None
        else:
            self._memcpy_stream: Optional[
                torch.cuda.streams.Stream
            ] = torch.cuda.Stream()
        self._cur_batch: Optional[In] = None
        self._connected = False

    def _connect(self, dataloader_iter: Iterator[In]) -> None:
        cur_batch = next(dataloader_iter)
        self._cur_batch = cur_batch
        with torch.cuda.stream(self._memcpy_stream):
            self._cur_batch = _to_device(cur_batch, self._device, non_blocking=True)

            with torch.no_grad():
                # Init lazy modules if any.
                model = self._model
                model(self._cur_batch)

                # Make sure we init data parallel modules if not done yet.
                if isinstance(model, DistributedModelParallel):
                    model.init_data_parallel()

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

        # Forward
        with record_function("## forward ##"):
            (losses, output) = self._model(cur_batch)

        if self._model.training:
            # Backward
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


class Tracer(torch.fx.Tracer):
    def __init__(self, unsharded_module_names: List[str]) -> None:
        super().__init__()
        self._unsharded_module_names = unsharded_module_names

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if (
            isinstance(m, ShardedModule)
            or module_qualified_name in self._unsharded_module_names
        ):
            return True
        return super().is_leaf_module(m, module_qualified_name)


@dataclass
class TrainPipelineContext:
    # pyre-ignore [4]
    input_dist_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    module_contexts: Dict[str, ShardedModuleContext] = field(default_factory=dict)


@dataclass
class ArgInfo:
    # attributes of input batch, e.g. batch.attr1.attr2 call
    # will produce ["attr1", "attr2"]
    input_attrs: List[str]
    # name for kwarg of pipelined forward() call or None
    # for a positional arg
    name: Optional[str]


class PipelinedForward(Generic[DistIn, DistOut, Out]):
    def __init__(
        self,
        name: str,
        args: List[ArgInfo],
        module: ShardedModule[DistIn, DistOut, Out],
        context: TrainPipelineContext,
        dist_stream: Optional[torch.cuda.streams.Stream],
    ) -> None:
        self._name = name
        self._args = args
        self._module = module
        self._context = context
        self._dist_stream = dist_stream

    # pyre-ignore [2]
    def __call__(self, *input, **kwargs) -> Awaitable[Out]:
        assert self._name in self._context.input_dist_requests
        request = self._context.input_dist_requests[self._name]
        assert isinstance(request, Awaitable)
        with record_function("## wait_sparse_data_dist ##"):
            data = request.wait()

        if self._dist_stream is not None:
            torch.cuda.current_stream().wait_stream(self._dist_stream)
            cur_stream = torch.cuda.current_stream()
            if isinstance(data, list):
                for d in data:
                    d.record_stream(cur_stream)
            else:
                data.record_stream(cur_stream)
            # self._context.module_contexts[self._name].record_stream(cur_stream)

        return self._module.compute_and_output_dist(
            self._context.module_contexts[self._name], data
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def args(self) -> List[ArgInfo]:
        return self._args


def _start_data_dist(
    pipelined_modules: List[ShardedModule],
    batch: In,
    context: TrainPipelineContext,
) -> None:
    context.input_dist_requests.clear()
    context.module_contexts.clear()
    for module in pipelined_modules:
        forward = module.forward
        assert isinstance(forward, PipelinedForward)

        # Retrieve argument.
        args = []
        kwargs = {}
        for arg_info in forward.args:
            if arg_info.input_attrs:
                arg = batch
                for attr in arg_info.input_attrs:
                    arg = getattr(arg, attr)
                if arg_info.name:
                    kwargs[arg_info.name] = arg
                else:
                    args.append(arg)
            else:
                args.append(None)

        # Start input distribution.
        module_ctx = module.create_context()
        context.module_contexts[forward.name] = module_ctx
        context.input_dist_requests[forward.name] = module.input_dist(
            module_ctx, *args, **kwargs
        )


# pyre-ignore
def _get_node_args_helper(arguments, num_found: int) -> Tuple[List[ArgInfo], int]:
    """A helper funtion that goes through the args/kwargs of a
    Node and arranges them into a list of ArgInfos. It also counts the number
    of (args + kwargs) found.
    """
    arg_info_list = [ArgInfo([], None) for _ in range(len(arguments))]
    for arg, arg_info in zip(arguments, arg_info_list):
        if arg is None:
            num_found += 1
            continue
        while True:
            if not isinstance(arg, torch.fx.Node):
                break
            child_node = arg

            if child_node.op == "placeholder":
                num_found += 1
                break
            elif (
                child_node.op == "call_function"
                and child_node.target.__module__ == "builtins"
                # pyre-ignore [16]
                and child_node.target.__name__ == "getattr"
            ):
                arg_info.input_attrs.insert(0, child_node.args[1])
                arg = child_node.args[0]
            else:
                break
    return arg_info_list, num_found


def _get_node_args(node: Node) -> Tuple[List[ArgInfo], int]:
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


def _rewrite_model(  # noqa C901
    model: torch.nn.Module,
    context: TrainPipelineContext,
    dist_stream: Optional[torch.cuda.streams.Stream],
) -> List[ShardedModule]:

    # Get underlying nn.Module
    while isinstance(model, DistributedModelParallel) or isinstance(
        model, DistributedDataParallel
    ):
        model = model.module

    # Collect a list of sharded modules.
    sharded_modules = {}
    for name, m in model.named_modules():
        if isinstance(m, ShardedModule):
            sharded_modules[name] = m

    # Trace a model.
    tracer = Tracer(get_unsharded_module_names(model))
    graph = tracer.trace(model)

    # Select sharded modules, which are top-level in the forward call graph,
    # i.e. which don't have input transformations, i.e.
    # rely only on 'builtins.getattr'.
    ret = []
    for node in graph.nodes:
        if node.op == "call_module" and node.target in sharded_modules:
            total_num_args = len(node.args) + len(node.kwargs)
            if total_num_args == 0:
                continue
            arg_info_list, num_found = _get_node_args(node)
            if num_found == total_num_args:
                logger.info(f"Module '{node.target}'' will be pipelined")
                child = sharded_modules[node.target]
                child.forward = PipelinedForward(
                    node.target, arg_info_list, child, context, dist_stream
                )
                ret.append(child)
    return ret


class TrainPipelineSparseDist(TrainPipeline[In, Out]):
    """
    This pipeline overlaps device transfer, and ShardedModule.input_dist() with
    forward and backward. This helps hiding all2all latency while preserving
    the training forward / backward ordering.
    stage 3: forward, backward
    stage 2: ShardedModule.input_dist()
    stage 1: device transfer

    ShardedModule.input_dist() is only done for top-level modules in the call graph.
    To be considered a top-level module,
    a module can only depend on 'getattr' calls on input.

    Input model must be symbolically traceable
    with the exception of ShardedModule and DistributedDataParallel modules.
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
        # use two data streams to support two concurrent batches
        if str(self._device) == "cpu":
            self._memcpy_stream: Optional[torch.cuda.streams.Stream] = None
            self._data_stream: Optional[torch.cuda.streams.Stream] = None
        else:
            self._memcpy_stream: Optional[
                torch.cuda.streams.Stream
            ] = torch.cuda.Stream()
            self._data_stream: Optional[torch.cuda.streams.Stream] = torch.cuda.Stream()
        self._batch_i: Optional[In] = None
        self._batch_ip1: Optional[In] = None
        self._batch_ip2: Optional[In] = None
        self._connected = False
        self._context = TrainPipelineContext()
        self._pipelined_modules: List[ShardedModule] = []

    def _connect(self, dataloader_iter: Iterator[In]) -> None:
        # batch 1
        with torch.cuda.stream(self._memcpy_stream):
            batch_i = next(dataloader_iter)
            self._batch_i = batch_i = _to_device(
                batch_i, self._device, non_blocking=True
            )
            with torch.no_grad():
                # Init lazy modules if any.
                model = self._model
                model(self._batch_i)

                if isinstance(model, DistributedModelParallel):
                    model.init_data_parallel()

                # Try to pipeline input data dist.
                self._pipelined_modules = _rewrite_model(
                    model, self._context, self._data_stream
                )

        with torch.cuda.stream(self._data_stream):
            _wait_for_batch(batch_i, self._memcpy_stream)
            _start_data_dist(self._pipelined_modules, batch_i, self._context)

        # batch 2
        with torch.cuda.stream(self._memcpy_stream):
            batch_ip1 = next(dataloader_iter)
            self._batch_ip1 = batch_ip1 = _to_device(
                batch_ip1, self._device, non_blocking=True
            )
        self._connected = True

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        if not self._connected:
            self._connect(dataloader_iter)

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        with record_function("## copy_batch_to_gpu ##"):
            with torch.cuda.stream(self._memcpy_stream):
                batch_ip2 = next(dataloader_iter)
                self._batch_ip2 = batch_ip2 = _to_device(
                    batch_ip2, self._device, non_blocking=True
                )
        batch_i = cast(In, self._batch_i)
        batch_ip1 = cast(In, self._batch_ip1)

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(batch_i, self._data_stream)

        # Forward
        with record_function("## forward ##"):
            (losses, output) = cast(Tuple[torch.Tensor, Out], self._model(batch_i))

        with record_function("## sparse_data_dist ##"):
            with torch.cuda.stream(self._data_stream):
                _wait_for_batch(batch_ip1, self._memcpy_stream)
                _start_data_dist(self._pipelined_modules, batch_ip1, self._context)

        if self._model.training:
            # Backward
            with record_function("## backward ##"):
                torch.sum(losses, dim=0).backward()

            # Update
            with record_function("## optimizer ##"):
                self._optimizer.step()

        self._batch_i = batch_ip1
        self._batch_ip1 = batch_ip2

        return output
