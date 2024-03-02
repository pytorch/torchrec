#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging

from collections import OrderedDict
from itertools import accumulate
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import nn
from torchrec import optim as trec_optim
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.types import (
    DataType,
    ParameterSharding,
    ShardedModule,
    ShardingType,
)
from torchrec.modules.embedding_configs import data_type_to_sparse_type
from torchrec.types import CopyMixIn

logger: logging.Logger = logging.getLogger(__name__)
_T = TypeVar("_T")

try:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_cpu"
    )
    try:
        if torch.version.hip:
            torch.ops.load_library(
                "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_gpu_hip"
            )
        else:
            torch.ops.load_library(
                "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_gpu_cuda"
            )
    except OSError:
        # For backward compatibility
        torch.ops.load_library(
            "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_gpu"
        )
except OSError:
    pass

# OSS
try:
    pass
except ImportError:
    pass

"""
torch.package safe functions from pyre_extensions. However, pyre_extensions is
not safe to use in code that will be torch.packaged, as it requires sys for
version checks
"""


def none_throws(optional: Optional[_T], message: str = "Unexpected `None`") -> _T:
    """Convert an optional to its value. Raises an `AssertionError` if the
    value is `None`"""
    if optional is None:
        raise AssertionError(message)
    return optional


def append_prefix(prefix: str, name: str) -> str:
    """
    Appends provided prefix to provided name.
    """

    if prefix != "" and name != "":
        return prefix + "." + name
    else:
        return prefix + name


def filter_state_dict(
    state_dict: "OrderedDict[str, torch.Tensor]", name: str
) -> "OrderedDict[str, torch.Tensor]":
    """
    Filters state dict for keys that start with provided name.
    Strips provided name from beginning of key in the resulting state dict.

    Args:
        state_dict (OrderedDict[str, torch.Tensor]): input state dict to filter.
        name (str): name to filter from state dict keys.

    Returns:
        OrderedDict[str, torch.Tensor]: filtered state dict.
    """

    filtered_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(name + "."):
            # + 1 to length is to remove the '.' after the key
            filtered_state_dict[key[len(name) + 1 :]] = value
    return filtered_state_dict


def add_prefix_to_state_dict(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Adds prefix to all keys in state dict, in place.

    Args:
        state_dict (Dict[str, Any]): input state dict to update.
        prefix (str): name to filter from state dict keys.

    Returns:
        None.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        state_dict[prefix + key] = state_dict.pop(key)

    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            if len(key) == 0:
                continue
            metadata[prefix + key] = metadata.pop(key)


def _get_unsharded_module_names_helper(
    model: torch.nn.Module,
    path: str,
    unsharded_module_names: Set[str],
) -> bool:
    sharded_children = set()
    for name, child in model.named_children():
        curr_path = path + name
        if isinstance(child, ShardedModule):
            sharded_children.add(name)
        else:
            child_sharded = _get_unsharded_module_names_helper(
                child,
                curr_path + ".",
                unsharded_module_names,
            )
            if child_sharded:
                sharded_children.add(name)

    if len(sharded_children) > 0:
        for name, _ in model.named_children():
            if name not in sharded_children:
                unsharded_module_names.add(path + name)

    return len(sharded_children) > 0


def get_unsharded_module_names(model: torch.nn.Module) -> List[str]:
    """
    Retrieves names of top level modules that do not contain any sharded sub-modules.

    Args:
        model (torch.nn.Module): model to retrieve unsharded module names from.

    Returns:
        List[str]: list of names of modules that don't have sharded sub-modules.
    """

    unsharded_module_names: Set[str] = set()
    _get_unsharded_module_names_helper(
        model,
        "",
        unsharded_module_names,
    )
    return list(unsharded_module_names)


class sharded_model_copy:
    """
    Allows copying of DistributedModelParallel module to a target device.

    Example::

        # Copying model to CPU.

        m = DistributedModelParallel(m)
        with sharded_model_copy("cpu"):
            m_cpu = copy.deepcopy(m)
    """

    def __init__(self, device: Optional[Union[str, int, torch.device]]) -> None:
        self.device = device

    def __enter__(self) -> None:
        # pyre-ignore [16]
        self.t_copy_save_ = torch.Tensor.__deepcopy__
        # pyre-ignore [16]
        self.p_copy_save_ = torch.nn.Parameter.__deepcopy__

        device = self.device

        # pyre-ignore [2, 3, 53]
        def _tensor_copy(tensor, memo):
            if tensor.device != device:
                return tensor.detach().to(device)
            else:
                return tensor.detach().clone()

        # pyre-ignore [2, 3]
        def _no_copy(obj, memo):
            return obj

        _copy_or_not = _tensor_copy if self.device is not None else _no_copy

        # pyre-ignore [2, 3, 53]
        def _param_copy(param, memo):
            return torch.nn.Parameter(
                _copy_or_not(param, memo), requires_grad=param.requires_grad
            )

        torch.Tensor.__deepcopy__ = _copy_or_not
        torch.nn.Parameter.__deepcopy__ = _param_copy
        # pyre-fixme[16]: `Type` has no attribute `__deepcopy__`.
        torch._C._distributed_c10d.ProcessGroupNCCL.__deepcopy__ = _no_copy
        # pyre-fixme[16]: `Type` has no attribute `__deepcopy__`.
        torch._C._distributed_c10d.ProcessGroupGloo.__deepcopy__ = _no_copy
        # pyre-fixme[16]: `Type` has no attribute `__deepcopy__`.
        torch._C._distributed_c10d.Work.__deepcopy__ = _no_copy
        # pyre-ignore [16]
        torch.cuda.streams.Stream.__deepcopy__ = _no_copy

    # pyre-ignore [2]
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # pyre-ignore [16]
        torch.Tensor.__deepcopy__ = self.t_copy_save_
        # pyre-ignore [16]
        torch.nn.Parameter.__deepcopy__ = self.p_copy_save_
        # pyre-fixme[16]: `Type` has no attribute `__deepcopy__`.
        torch._C._distributed_c10d.ProcessGroupNCCL.__deepcopy__ = None
        # pyre-fixme[16]: `Type` has no attribute `__deepcopy__`.
        torch._C._distributed_c10d.ProcessGroupGloo.__deepcopy__ = None
        # pyre-fixme[16]: `Type` has no attribute `__deepcopy__`.
        torch._C._distributed_c10d.Work.__deepcopy__ = None
        # pyre-ignore [16]
        torch.cuda.streams.Stream.__deepcopy__ = None


def copy_to_device(
    module: nn.Module,
    current_device: torch.device,
    to_device: torch.device,
) -> nn.Module:

    with sharded_model_copy(device=None):
        copy_module = copy.deepcopy(module)

    # Copy only weights with matching device.
    def _copy_if_device_match(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == current_device:
            return tensor.to(to_device)
        return tensor

    # if this is a sharded module, customize the copy
    if isinstance(copy_module, CopyMixIn):
        return copy_module.copy(to_device)
    copied_param = {
        name: torch.nn.Parameter(
            _copy_if_device_match(param.data), requires_grad=param.requires_grad
        )
        for name, param in copy_module.named_parameters(recurse=False)
    }
    copied_buffer = {
        name: _copy_if_device_match(buffer)
        for name, buffer in copy_module.named_buffers(recurse=False)
    }
    for name, param in copied_param.items():
        copy_module.register_parameter(name, param)
    for name, buffer in copied_buffer.items():
        copy_module.register_buffer(name, buffer)
    for child_name, child in copy_module.named_children():
        if not any([isinstance(submodule, CopyMixIn) for submodule in child.modules()]):
            child_copy = child._apply(_copy_if_device_match)
        else:
            child_copy = copy_to_device(child, current_device, to_device)
        copy_module.register_module(child_name, child_copy)
    return copy_module


class CopyableMixin(nn.Module):
    """
    Allows copying of module to a target device.

    Example::

        class MyModule(CopyableMixin):
            ...

    Args:
        device : torch.device to copy to

    Returns
        nn.Module on new device
    """

    def copy(
        self,
        device: torch.device,
    ) -> nn.Module:
        return copy_to_device(
            self,
            current_device=torch.device("cpu"),
            to_device=device,
        )


def optimizer_type_to_emb_opt_type(
    optimizer_class: Type[torch.optim.Optimizer],
) -> Optional[EmbOptimType]:
    # TODO add more optimizers to be in parity with ones provided by FBGEMM
    # TODO kwargs accepted by fbgemm and and canonical optimizers are different
    # may need to add special handling for them
    lookup = {
        torch.optim.SGD: EmbOptimType.EXACT_SGD,
        torch.optim.Adagrad: EmbOptimType.EXACT_ADAGRAD,
        torch.optim.Adam: EmbOptimType.ADAM,
        # below are torchrec wrappers over these optims.
        # they accept an **unused kwargs portion, that let us set FBGEMM specific args such as
        # max gradient, etc
        trec_optim.SGD: EmbOptimType.EXACT_SGD,
        trec_optim.LarsSGD: EmbOptimType.LARS_SGD,
        trec_optim.LAMB: EmbOptimType.LAMB,
        trec_optim.PartialRowWiseLAMB: EmbOptimType.PARTIAL_ROWWISE_LAMB,
        trec_optim.Adam: EmbOptimType.ADAM,
        trec_optim.PartialRowWiseAdam: EmbOptimType.PARTIAL_ROWWISE_ADAM,
        trec_optim.Adagrad: EmbOptimType.EXACT_ADAGRAD,
        trec_optim.RowWiseAdagrad: EmbOptimType.EXACT_ROWWISE_ADAGRAD,
    }
    if optimizer_class not in lookup:
        raise ValueError(f"Cannot cast {optimizer_class} to an EmbOptimType")
    return lookup[optimizer_class]


def merge_fused_params(
    fused_params: Optional[Dict[str, Any]] = None,
    param_fused_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    """
    Configure the fused_params including cache_precision if the value is not preset.

    Values set in table_level_fused_params take precidence over the global fused_params

    Args:
        fused_params (Optional[Dict[str, Any]]): the original fused_params
        grouped_fused_params

    Returns:
        [Dict[str, Any]]: a non-null configured fused_params dictionary to be
        used to configure the embedding lookup kernel
    """

    if fused_params is None:
        fused_params = {}
    if param_fused_params is None:
        param_fused_params = {}
    if "lr" in param_fused_params:
        param_fused_params["learning_rate"] = param_fused_params.pop("lr")

    _fused_params = copy.deepcopy(fused_params)
    _fused_params.update(param_fused_params)
    return _fused_params


def add_params_from_parameter_sharding(
    fused_params: Optional[Dict[str, Any]],
    parameter_sharding: ParameterSharding,
) -> Dict[str, Any]:
    """
    Extract params from parameter sharding and then add them to fused_params.

    Params from parameter sharding will override the ones in fused_params if they
    exist already.

    Args:
        fused_params (Optional[Dict[str, Any]]): the existing fused_params
        parameter_sharding (ParameterSharding): the parameter sharding to use

    Returns:
        [Dict[str, Any]]: the fused_params dictionary with params from parameter
        sharding added.

    """
    if fused_params is None:
        fused_params = {}

    # update fused_params using params from parameter_sharding
    # this will take precidence over the fused_params provided from sharders
    if parameter_sharding.cache_params is not None:
        cache_params = parameter_sharding.cache_params
        if cache_params.algorithm is not None:
            fused_params["cache_algorithm"] = cache_params.algorithm
        if cache_params.load_factor is not None:
            fused_params["cache_load_factor"] = cache_params.load_factor
        if cache_params.reserved_memory is not None:
            fused_params["cache_reserved_memory"] = cache_params.reserved_memory
        if cache_params.precision is not None:
            fused_params["cache_precision"] = cache_params.precision
        if cache_params.prefetch_pipeline is not None:
            fused_params["prefetch_pipeline"] = cache_params.prefetch_pipeline

    if parameter_sharding.enforce_hbm is not None:
        fused_params["enforce_hbm"] = parameter_sharding.enforce_hbm

    if parameter_sharding.stochastic_rounding is not None:
        fused_params["stochastic_rounding"] = parameter_sharding.stochastic_rounding

    if parameter_sharding.bounds_check_mode is not None:
        fused_params["bounds_check_mode"] = parameter_sharding.bounds_check_mode

    # print warning if sharding_type is data_parallel or kernel is dense
    if parameter_sharding.sharding_type == ShardingType.DATA_PARALLEL.value:
        logger.warning(
            f"Sharding Type is {parameter_sharding.sharding_type}, "
            "caching params will be ignored"
        )
    elif parameter_sharding.compute_kernel == EmbeddingComputeKernel.DENSE.value:
        logger.warning(
            f"Compute Kernel is {parameter_sharding.compute_kernel}, "
            "caching params will be ignored"
        )

    return fused_params


def convert_to_fbgemm_types(fused_params: Dict[str, Any]) -> Dict[str, Any]:
    if "cache_precision" in fused_params:
        if isinstance(fused_params["cache_precision"], DataType):
            fused_params["cache_precision"] = data_type_to_sparse_type(
                fused_params["cache_precision"]
            )

    if "weights_precision" in fused_params:
        if isinstance(fused_params["weights_precision"], DataType):
            fused_params["weights_precision"] = data_type_to_sparse_type(
                fused_params["weights_precision"]
            )

    if "output_dtype" in fused_params:
        if isinstance(fused_params["output_dtype"], DataType):
            fused_params["output_dtype"] = data_type_to_sparse_type(
                fused_params["output_dtype"]
            )

    return fused_params


def init_parameters(module: nn.Module, device: torch.device) -> None:
    with torch.no_grad():
        has_meta_param = any(t.is_meta for t in module.parameters())
        if has_meta_param:
            module.to_empty(device=device)

            def maybe_reset_parameters(m: nn.Module) -> None:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

            module.apply(maybe_reset_parameters)


# TODO remove and use FBGEMM version once changes are in the package
class PermutePooledEmbeddings:
    def __init__(
        self,
        embs_dims: List[int],
        permute: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        logging.info("Using Permute Pooled Embeddings")
        self._offset_dim_list: torch.Tensor = torch.tensor(
            [0] + list(accumulate(embs_dims)), device=device, dtype=torch.int64
        )

        self._permute: torch.Tensor = torch.tensor(
            permute, device=device, dtype=torch.int64
        )

        inv_permute: List[int] = [0] * len(permute)
        for i, p in enumerate(permute):
            inv_permute[p] = i

        self._inv_permute: torch.Tensor = torch.tensor(
            inv_permute, device=device, dtype=torch.int64
        )

        inv_embs_dims = [embs_dims[i] for i in permute]

        self._inv_offset_dim_list: torch.Tensor = torch.tensor(
            [0] + list(accumulate(inv_embs_dims)), device=device, dtype=torch.int64
        )

    def __call__(self, pooled_embs: torch.Tensor) -> torch.Tensor:
        result = torch.ops.fbgemm.permute_pooled_embs_auto_grad(
            pooled_embs,
            self._offset_dim_list.to(device=pooled_embs.device),
            self._permute.to(device=pooled_embs.device),
            self._inv_offset_dim_list.to(device=pooled_embs.device),
            self._inv_permute.to(device=pooled_embs.device),
        )
        return result
