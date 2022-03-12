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
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    TypeVar,
    List,
    Type,
    Iterator,
)

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
    ShardingSpec,
    ShardMetadata,
    EnumerableShardingSpec,
)
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


W = TypeVar("W")
M = TypeVar("M", bound=nn.Module)
Out = TypeVar("Out")
CompIn = TypeVar("CompIn", bound=Multistreamable)
DistOut = TypeVar("DistOut")


class Awaitable(abc.ABC, Generic[W]):
    def __init__(self) -> None:
        self._callbacks: List[Callable[[W], W]] = []

    @abc.abstractmethod
    def _wait_impl(self) -> W:
        pass

    def wait(self) -> W:
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


@dataclass
class ShardedModuleContext(Multistreamable):
    pass


class EmptyContext(ShardedModuleContext):
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
        # pyre-fixme[11]: Annotation `ProcessGroup` is not defined as a type.
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


class ShardedModule(abc.ABC, nn.Module, Generic[CompIn, DistOut, Out]):
    """
    All model-parallel modules implement this interface.
    Inputs and outputs are data-parallel.

    NOTE:
        'input_dist' / 'output_dist' are responsible of transforming inputs / outputs
        from data-parallel to model parallel and vise-versa.
    """

    @abc.abstractmethod
    def __init__(self) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")

    def create_context(self) -> ShardedModuleContext:
        return EmptyContext()

    @abc.abstractmethod
    def input_dist(
        self,
        ctx: ShardedModuleContext,
        # pyre-ignore[2]
        *input,
        # pyre-ignore[2]
        **kwargs,
    ) -> Awaitable[CompIn]:
        pass

    @abc.abstractmethod
    def compute(self, ctx: ShardedModuleContext, dist_input: CompIn) -> DistOut:
        pass

    @abc.abstractmethod
    def output_dist(
        self, ctx: ShardedModuleContext, output: DistOut
    ) -> LazyAwaitable[Out]:
        pass

    def compute_and_output_dist(
        self, ctx: ShardedModuleContext, input: CompIn
    ) -> LazyAwaitable[Out]:
        """
        In case of multiple output distributions it makes sense to override this method
        and initiate output distibution as soon as the corresponding compute completes.
        """
        output = self.compute(ctx, input)
        return self.output_dist(ctx, output)

    # pyre-ignore[2]
    def forward(self, *input, **kwargs) -> LazyAwaitable[Out]:
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

    def copy(self, device: torch.device) -> nn.Module:
        return self.to(device)


class ModuleSharder(abc.ABC, Generic[M]):
    """
    `ModuleSharder` is per each module, which supports sharding,
    e.g. `EmbeddingBagCollection`.
    """

    def __init__(self) -> None:
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")

    @abc.abstractclassmethod
    # pyre-ignore [3]
    def shard(
        self,
        module: M,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedModule[Any, Any, Any]:
        """
        Does the actual sharding. It will allocate parameters on the requested locations
        as specified by corresponding ParameterSharding.

        Default implementation is data-parallel replication.

        Args:
            module (M): module to shard.
            params (Dict[str, ParameterSharding]): dict of fully qualified parameter names
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

    def shardable_parameters(self, module: M) -> Dict[str, nn.Parameter]:
        """
        List of parameters that can be sharded.
        """
        return dict(module.named_parameters())

    def sharding_types(self, compute_device_type: str) -> List[str]:
        """
        List of supported sharding types. See ShardingType for well-known examples.
        """
        return [ShardingType.DATA_PARALLEL.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        """
        List of supported compute kernels for a given sharding_type and compute device.
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


@dataclass
class ShardingPlan:
    """
    Representation of sharding plan.
    Attributes:

        plan (Dict[str, Dict[str, ParameterSharding]]): dict keyed by module path of
            dict of parameter sharding specs keyed by parameter name.
    """

    plan: Dict[str, Dict[str, ParameterSharding]]

    def get_plan_for_module(
        self, module_path: str
    ) -> Optional[Dict[str, ParameterSharding]]:
        """
        Args:
            module_path (str):

        Returns:
            Optional[Dict[str, ParameterSharding]]: dict of parameter sharding specs keyed by parameter name. None if sharding specs do not exist for given module_path.
        """
        return self.plan.get(module_path, None)

    def __str__(self) -> str:
        return str(self.plan)


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
