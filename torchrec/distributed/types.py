#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import operator
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar

from torch.autograd.profiler import record_function
from torchrec.types import ModuleNoCopyMixin

try:
    # For python 3.6 and below, GenericMeta will be used by
    # other metaclasses (i.e. AwaitableMeta) for customized
    # behaviors, as Generic is non-trival metaclass in
    # python 3.6 and below
    from typing import GenericMeta  # pyre-ignore: python 3.6
except ImportError:
    # In python 3.7+, GenericMeta doesn't exist as it's no
    # longer a non-trival metaclass,
    #   see https://www.python.org/dev/peps/pep-0560/
    # So we make a fake type here in order to share the same
    # code with python 3.6 or below, it will just be used as
    # a placeholder for customized metaclass behaviors
    # (i.e. Awaitable)
    class GenericMeta(type):
        pass


import torch
import torch.distributed as dist
import torch.fx
from torch import nn

# @manual
from torch.distributed._shard.sharded_tensor import (  # noqa
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)

# @manual
from torch.distributed._shard.sharding_spec import (  # noqa
    EnumerableShardingSpec,
    ShardingSpec,
    ShardMetadata,
)
from torch.nn.modules.module import _addindent
from torchrec.streamable import Multistreamable


class ShardingType(Enum):
    """
    Well-known sharding types, used by inter-module optimizations.
    """

    # Replicated on all ranks
    DATA_PARALLEL = "data_parallel"
    # Placed on a single rank
    TABLE_WISE = "table_wise"
    # Placed on multiple ranks as different sharded tables
    COLUMN_WISE = "column_wise"
    # Range-split on the first dimension across all ranks
    ROW_WISE = "row_wise"
    # Row-wise on the same node and table-wise across nodes
    # Useful when having multiple ranks per node
    # and comms within a single node are more efficient than across nodes.
    TABLE_ROW_WISE = "table_row_wise"
    # Column-wise on the same node and table-wise across nodes
    TABLE_COLUMN_WISE = "table_column_wise"


class ParameterStorage(Enum):
    """
    Well-known physical resources, which can be used as constraints by ShardingPlanner.
    """

    # GPU-attached memory
    HBM = "hbm"
    # CPU-attached memory
    DDR = "ddr"


@unique
class ComputeKernel(Enum):
    DEFAULT = "default"


QuantizationContext = TypeVar("QuantizationContext")


# Once we only support python3.8+ use
# from typing import protocol.
# We can't use from typing_extensions import Protocol due to torch.deploy restrictions.
class QuantizedCommCodec(Generic[QuantizationContext]):
    """
    Provide an implementation to quantized, or apply mixed precision, to the tensors used in collective calls (pooled_all_to_all, reduce_scatter, etc).
    The dtype is the dtype of the tensor called from encode.

    This makes the assumption that the input tensor has type torch.float32

    >>>
        quantized_tensor = quantized_comm_codec.encode(input_tensor)
        quantized_tensor.dtype == quantized_comm_codec.quantized_dtype
        collective_call(output_tensors, input_tensors=tensor)
        output_tensor = decode(output_tensors)

        torch.assert_close(input_tensors, output_tensor)

    """

    def encode(
        self, input_tensor: torch.Tensor, ctx: Optional[QuantizationContext] = None
    ) -> torch.Tensor:
        ...

    def decode(
        self, input_grad: torch.Tensor, ctx: Optional[QuantizationContext] = None
    ) -> torch.Tensor:
        ...

    def quantized_dtype(self) -> torch.dtype:
        """
        tensor.dtype of the resultant encode(input_tensor)
        """
        ...

    def calc_quantized_size(
        self,
        input_len: int,
        ctx: Optional[QuantizationContext] = None,
    ) -> int:
        """
        Given the length of input tensor, returns the length of tensor after
        quantization. Used by INT8 codecs where the quantized tensor have
        some additional parameters. For other cases, the quantized tensor should
        have the same length with input.
        """
        ...

    def create_context(self) -> Optional[QuantizationContext]:
        """
        Create a context object that can be used to carry session-based
        parameters between encoder and decoder.
        """
        ...


class NoOpQuantizedCommCodec(Generic[QuantizationContext]):
    """
    Default No-Op implementation of QuantizedCommCodec
    """

    def encode(
        self,
        input_tensor: torch.Tensor,
        ctx: Optional[QuantizationContext] = None,
    ) -> torch.Tensor:
        return input_tensor

    def decode(
        self,
        input_grad: torch.Tensor,
        ctx: Optional[QuantizationContext] = None,
    ) -> torch.Tensor:
        return input_grad

    def quantized_dtype(self) -> torch.dtype:
        return torch.float

    def calc_quantized_size(
        self,
        input_len: int,
        ctx: Optional[QuantizationContext] = None,
    ) -> int:
        return input_len

    def create_context(self) -> Optional[QuantizationContext]:
        return None


@dataclass
class QuantizedCommCodecs:
    """
    The quantization codecs to use for the forward and backward pass respectively of a comm op (e.g. pooled_all_to_all, reduce_scatter, sequence_all_to_all).
    """

    # pyre-ignore
    forward: QuantizedCommCodec = NoOpQuantizedCommCodec()
    # pyre-ignore
    backward: QuantizedCommCodec = NoOpQuantizedCommCodec()


class CommOp(Enum):
    # For detailed descriptions of each of these, see their doc strings in dist_data.
    # These are commonly used inside of a QuantizedCommsRegistry
    POOLED_EMBEDDINGS_ALL_TO_ALL = "pooled_embeddings_all_to_all"
    POOLED_EMBEDDINGS_REDUCE_SCATTER = "pooled_embeddings_reduce_scatter"
    SEQUENCE_EMBEDDINGS_ALL_TO_ALL = "sequence_embeddings_all_to_all"


W = TypeVar("W")
M = TypeVar("M", bound=nn.Module)


class Awaitable(abc.ABC, Generic[W]):
    def __init__(self) -> None:
        self._callbacks: List[Callable[[W], W]] = []

    @abc.abstractmethod
    def _wait_impl(self) -> W:
        pass

    def wait(self) -> W:
        with record_function(f"## {self.__class__.__name__} wait() ##"):
            ret: W = self._wait_impl()
            for callback in self.callbacks:
                ret = callback(ret)
        return ret

    @property
    def callbacks(self) -> List[Callable[[W], W]]:
        return self._callbacks


class NoWait(Awaitable[W]):
    def __init__(self, obj: W) -> None:
        super().__init__()
        self._obj = obj

    def _wait_impl(self) -> W:
        return self._obj


# pyre-fixme[11]: Annotation `ProxyableClassMeta` is not defined as a type.
class _LazyAwaitableMeta(GenericMeta, abc.ABCMeta, torch.fx.ProxyableClassMeta):
    """
    The _LazyAwaitableMeta class that inherits both ABCMeta and ProxyableClassMeta
    This is because ABCMeta/ProxyableClassMeta are both non-trival metaclasses
    Declaring this separately to ensure the init order of metaclasses

    XXX: Generics are non-trival metaclass before python 3.7 but are removed
    afterwards. we add GenericsMeta here to support version before 3.7.
    """

    pass


class LazyAwaitable(Awaitable[W], metaclass=_LazyAwaitableMeta):
    """
    The LazyAwaitable type which exposes a `wait()` API, concrete types
    can control how to initialize and how the `wait()` behavior should
    be in order to achieve specific async operation.

    This base LazyAwaitable type is a "lazy" async type, which means it will
    delay `wait()` as late as possible, see details in `__torch_function__`
    below. This could help the model automatically enable computation and
    communication overlap, model author doesn't need to manually call
    `wait()` if the results is used by a pytorch function, or by other python
    operations (NOTE: need to implement corresponding magic methods
    like __getattr__ below)

    Some caveats:

    * This works with Pytorch functions, but not any generic method, if
      you would like to do arbitary python operations, you need to
      implement the corresponding magic methods

    * In the case that one function have two or more arguments are LazyAwaitable,
      the lazy wait mechanism can't ensure perfect computation/communication
      overlap (i.e. quickly waited the first one but long wait on the second)
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()
        # _result is used to cache the results after the wait() is called.
        self._result: Optional[W] = None

    @staticmethod
    # pyre-ignore [2, 3]
    def _wait_async(obj: Any) -> Any:
        """
        This method is used internally to automatically wait when necessary
        and cache the results of the `LazyAwaitable.wait()` call
        """
        if isinstance(obj, LazyAwaitable):
            if obj._result is None:
                obj._result = obj.wait()
            return obj._result
        else:
            return obj

    # pyre-ignore [2, 3]
    def __torch_function__(self, func, types, args=(), kwargs=None):
        """
        The LazyAwaitable type has a `__torch_function__` implementation.
        This means when this type is seens as an argument to a PyTorch
        function in a position where it expects a W, the PyTorch's
        dispatcher will call into this function for special handling

        Our `__torch_function__` implementation goes through all of the
        args and kwargs and checks if any of them are `LazyAwaitable`.
        If it is, it will call `wait()` on it and replace the LazyAwaitable
        type object with the result of wait. In this way, async values
        are waited on when the concrete value is first needed and without
        the user having to write an explicit `wait()` call.
        """
        kwargs = kwargs or {}

        # wait() on all LazyAwaitable args/kwargs and replace
        # them with the resulting value.
        new_args = torch.fx.node.map_aggregate(args, LazyAwaitable._wait_async)
        new_kwargs = torch.fx.node.map_aggregate(kwargs, LazyAwaitable._wait_async)

        return func(*new_args, **new_kwargs)

    # pyre-ignore [2, 3]
    def __getattr__(self, name):
        """
        Overrides __getattr__ to allow LazyAwaitable to first wait and then call getattr
        on the wait results.
        """
        if name == "_result":
            raise RuntimeError(
                f"LazyAwaitable type {type(self)} has not been initialized properly, "
                f"did you forget to call 'super()'?"
            )

        res = LazyAwaitable._wait_async(self)
        return getattr(res, name)


class LazyNoWait(LazyAwaitable[W]):
    def __init__(self, obj: W) -> None:
        super().__init__()
        self._obj = obj

    def _wait_impl(self) -> W:
        return self._obj


# install magic methods
for orig_method_name in torch.fx.graph.magic_methods:
    as_magic = f"__{orig_method_name}__"

    def scope(method):
        def impl(*args, **kwargs):
            lhs = args[0]
            op_fn = getattr(operator, method)
            if len(args) == 1:
                return op_fn(LazyAwaitable._wait_async(lhs))
            elif len(args) == 2:
                rhs = args[1]
                return op_fn(
                    LazyAwaitable._wait_async(lhs), LazyAwaitable._wait_async(rhs)
                )
            else:
                raise RuntimeError(f"magic method {as_magic} not supported!")

        impl.__name__ = as_magic
        setattr(LazyAwaitable, as_magic, impl)

    # pyre-ignore [16]
    scope(orig_method_name)

# install reflective magic methods
for orig_method_name in torch.fx.graph.reflectable_magic_methods:
    as_magic = f"__r{orig_method_name}__"
    # pyre-ignore [2, 3]
    def scope(method):
        # pyre-ignore [2, 3, 53]
        def impl(self, rhs):
            op_fn = getattr(operator, method)
            return op_fn(
                LazyAwaitable._wait_async(rhs), LazyAwaitable._wait_async(self)
            )

        impl.__name__ = as_magic
        impl.__qualname__ = as_magic
        setattr(LazyAwaitable, as_magic, impl)

    # pyre-ignore [16]
    scope(orig_method_name)


@dataclass
class ParameterSharding:
    """
        Describes the sharding of the parameter.

        sharding_type (str): how this parameter is sharded. See ShardingType for well-known
            types.
        compute_kernel (str): compute kernel to be used by this parameter.
        ranks (Optional[List[int]]): rank of each shard.
        sharding_spec (Optional[ShardingSpec]): list of ShardMetadata for each shard.

    NOTE:
      ShardingType.TABLE_WISE - rank where this embedding is placed
      ShardingType.COLUMN_WISE - rank where the embedding shards are placed, seen as
      individual tables
      ShardingType.TABLE_ROW_WISE  - first rank when this embedding is placed
      ShardingType.ROW_WISE, ShardingType.DATA_PARALLEL - unused

    """

    sharding_type: str
    compute_kernel: str
    ranks: Optional[List[int]] = None
    sharding_spec: Optional[ShardingSpec] = None


ModuleShardingPlan = Dict[str, ParameterSharding]
"""
Map of ParameterSharding per parameter (usually a table). This describes the sharding plan for a torchrec module (e.g. `EmbeddingBagCollection`)
"""


@dataclass
class ShardingPlan:
    """
    Representation of sharding plan. This uses the FQN of the larger wrapped model (i.e the model that is wrapped using `DistributedModelParallel`)
    ModuleShardingPlan should be used when TorchRec composability is desired.

    Attributes:
        plan (Dict[str, ModuleShardingPlan]): dict keyed by module path of
            dict of parameter sharding specs keyed by parameter name.
    """

    plan: Dict[str, ModuleShardingPlan]

    def get_plan_for_module(self, module_path: str) -> Optional[ModuleShardingPlan]:
        """
        Args:
            module_path (str):

        Returns:
            Optional[ModuleShardingPlan]: dict of parameter sharding specs keyed by parameter name. None if sharding specs do not exist for given module_path.
        """
        return self.plan.get(module_path, None)

    def __str__(self) -> str:
        return str(self.plan)


ShardedModuleContext = Multistreamable


class NullShardedModuleContext(Multistreamable):
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        pass

    # pyre-ignore [2]
    def __setattr__(self, key: str, value: Any) -> None:
        raise NotImplementedError()


class ShardingEnv:
    """
    Provides an abstraction over `torch.distributed.ProcessGroup`, which practically
    enables `DistributedModelParallel` to be used during inference.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.process_group: Optional[dist.ProcessGroup] = pg

    @classmethod
    def from_process_group(cls, pg: dist.ProcessGroup) -> "ShardingEnv":
        """
        Creates ProcessGroup-based sharding environment.

        NOTE:
            Typically used during training.
        """
        return cls(dist.get_world_size(pg), dist.get_rank(pg), pg)

    @classmethod
    def from_local(cls, world_size: int, rank: int) -> "ShardingEnv":
        """
        Creates a local host-based sharding environment.

        NOTE:
            Typically used during single host inference.
        """
        return cls(world_size, rank, None)


class FeatureShardingMixIn:
    """
    Feature Sharding Interface to provide sharding-aware feature metadata.
    """

    def id_list_feature_names(self) -> List[str]:
        raise NotImplementedError

    def id_score_list_feature_names(self) -> List[str]:
        raise NotImplementedError

    def id_list_feature_names_per_rank(self) -> List[List[str]]:
        raise NotImplementedError

    def id_score_list_feature_names_per_rank(self) -> List[List[str]]:
        raise NotImplementedError

    def id_list_features_per_rank(self) -> List[int]:
        raise NotImplementedError

    def id_score_list_features_per_rank(self) -> List[int]:
        raise NotImplementedError


class ModuleShardingMixIn:
    """
    The interface to access a sharded module's sharding scheme.
    """

    @property
    def shardings(self) -> Dict[str, FeatureShardingMixIn]:
        raise NotImplementedError


Out = TypeVar("Out")
CompIn = TypeVar("CompIn", bound=Multistreamable)
DistOut = TypeVar("DistOut")
ShrdCtx = TypeVar("ShrdCtx", bound=Multistreamable)


class ShardedModule(
    abc.ABC,
    nn.Module,
    Generic[CompIn, DistOut, Out, ShrdCtx],
    ModuleNoCopyMixin,
    ModuleShardingMixIn,
):
    """
    All model-parallel modules implement this interface.
    Inputs and outputs are data-parallel.

    Args::
        qcomm_codecs_registry (Optional[Dict[str, QuantizedCommCodecs]]) : Mapping of CommOp name to QuantizedCommCodecs

    NOTE:
        'input_dist' / 'output_dist' are responsible of transforming inputs / outputs
        from data-parallel to model parallel and vise-versa.
    """

    @abc.abstractmethod
    def __init__(
        self, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None
    ) -> None:

        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")

        if qcomm_codecs_registry is None:
            qcomm_codecs_registry = {}
        self._qcomm_codecs_registry = qcomm_codecs_registry

        self._input_dists: List[nn.Module] = []
        self._lookups: List[nn.Module] = []
        self._output_dists: List[nn.Module] = []

    @abc.abstractmethod
    def create_context(self) -> ShrdCtx:
        pass

    @property
    def qcomm_codecs_registry(self) -> Optional[Dict[str, QuantizedCommCodecs]]:
        return self._qcomm_codecs_registry

    @abc.abstractmethod
    def input_dist(
        self,
        ctx: ShrdCtx,
        # pyre-ignore[2]
        *input,
        # pyre-ignore[2]
        **kwargs,
    ) -> Awaitable[CompIn]:
        pass

    @abc.abstractmethod
    def compute(self, ctx: ShrdCtx, dist_input: CompIn) -> DistOut:
        pass

    @abc.abstractmethod
    def output_dist(self, ctx: ShrdCtx, output: DistOut) -> LazyAwaitable[Out]:
        pass

    def compute_and_output_dist(
        self, ctx: ShrdCtx, input: CompIn
    ) -> LazyAwaitable[Out]:
        """
        In case of multiple output distributions it makes sense to override this method
        and initiate the output distibution as soon as the corresponding compute
        completes.
        """
        output = self.compute(ctx, input)
        return self.output_dist(ctx, output)

    # pyre-ignore[2]
    def forward(self, *input, **kwargs) -> LazyAwaitable[Out]:
        """
        Executes the input dist, compute, and output dist steps.

        Args:
            *input: input.
            **kwargs: keyword arguments.

        Returns:
            LazyAwaitable[Out]: awaitable of output from output dist.
        """
        ctx = self.create_context()
        dist_input = self.input_dist(ctx, *input, **kwargs).wait()
        return self.compute_and_output_dist(ctx, dist_input)

    def sparse_grad_parameter_names(
        self,
        destination: Optional[List[str]] = None,
        prefix: str = "",
    ) -> List[str]:
        destination = [] if destination is None else destination
        return destination

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        for key, _ in self.named_parameters(prefix):
            yield key

    def extra_repr(self) -> str:
        """
        Pretty prints representation of the module's lookup modules, input_dists and output_dists
        """

        def loop(key: str, modules: List[nn.Module]) -> List[str]:
            child_lines = []
            if len(modules) > 0:
                child_lines.append("(" + key + "): ")
            for module in modules:
                mod_str = repr(module)
                mod_str = _addindent(mod_str, 2)
                child_lines.append(mod_str)
            return child_lines

        rep = []
        rep.extend(loop("lookups", self._lookups))
        rep.extend(loop("_input_dists", self._input_dists))
        rep.extend(loop("_output_dists", self._output_dists))

        return "\n ".join(rep)


class ModuleSharder(abc.ABC, Generic[M]):
    """
    `ModuleSharder` is per each module, which supports sharding,
    e.g. `EmbeddingBagCollection`.

    Args::
        qcomm_codecs_registry (Optional[Dict[str, QuantizedCommCodecs]]) : Mapping of CommOp name to QuantizedCommCodecs
    """

    def __init__(
        self, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None
    ) -> None:
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._qcomm_codecs_registry = qcomm_codecs_registry

    @abc.abstractclassmethod
    # pyre-ignore [3]
    def shard(
        self,
        module: M,
        params: ModuleShardingPlan,
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedModule[Any, Any, Any, Any]:
        """
        Does the actual sharding. It will allocate parameters on the requested locations
        as specified by corresponding ParameterSharding.

        Default implementation is data-parallel replication.

        Args:
            module (M): module to shard.
            params (ModuleShardingPlan): dict of fully qualified parameter names
                (module path + parameter name, '.'-separated) to its sharding spec.
            env (ShardingEnv): sharding environment that has the process group.
            device (torch.device): compute device.

        Returns:
            ShardedModule[Any, Any, Any]: sharded module implementation.
        """
        ...

    @property
    @abc.abstractmethod
    def module_type(self) -> Type[M]:
        ...

    @property
    def qcomm_codecs_registry(self) -> Optional[Dict[str, QuantizedCommCodecs]]:
        return self._qcomm_codecs_registry

    def shardable_parameters(self, module: M) -> Dict[str, nn.Parameter]:
        """
        List of parameters that can be sharded.
        """
        return dict(module.named_parameters())

    def sharding_types(self, compute_device_type: str) -> List[str]:
        """
        List of supported sharding types. See `ShardingType` for well-known examples.
        """
        return [ShardingType.DATA_PARALLEL.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        """
        List of supported compute kernels for a given sharding type and compute device.
        """

        return [ComputeKernel.DEFAULT.value]

    def storage_usage(
        self, tensor: torch.Tensor, compute_device_type: str, compute_kernel: str
    ) -> Dict[str, int]:
        """
        List of system resources and corresponding usage given a compute device and
        compute kernel.
        """

        assert compute_device_type in {"cuda", "cpu"}
        storage_map = {"cuda": ParameterStorage.HBM, "cpu": ParameterStorage.DDR}
        return {
            storage_map[compute_device_type].value: tensor.element_size()
            * tensor.nelement()
        }


class ShardingPlanner(abc.ABC):
    """
    Plans sharding.
    This plan can be saved and re-used to ensure sharding stability.
    """

    @abc.abstractmethod
    def plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        """
        Plans sharding for provided module and given sharders.

        Args:
            module (nn.Module): module that sharding is planned for.
            sharders (List[ModuleSharder[nn.Module]]): provided sharders for module.

        Returns:
            ShardingPlan: the computed sharding plan.
        """
        ...

    @abc.abstractmethod
    def collective_plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        """
        Calls self.plan(...) on rank 0 and broadcasts.

        Args:
            module (nn.Module): module that sharding is planned for.
            sharders (List[ModuleSharder[nn.Module]]): provided sharders for module.

        Returns:
            ShardingPlan: the computed sharding plan.
        """
        ...
