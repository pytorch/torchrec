#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import itertools
from collections import defaultdict, OrderedDict
from typing import Any, cast, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    ComputeDevice,
    EmbeddingLocation,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from torch.nn.modules.module import _IncompatibleKeys
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    pooling_type_to_pooling_mode,
    PoolingType,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
    EmbeddingCollection,
    get_embedding_names_by_table,
)

from torchrec.optim.fused import FusedOptimizer, FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class EmbeddingFusedOptimizer(FusedOptimizer):
    """
    EmbeddingFusedOptimizer exposes the internal SplitTableBatchedEmbeddingBagsCodeGen optimizer state.
    It can be used like a normal optimizer to perform functionalities such as loading/saving optimizers
    and updating learning rates.

    Args:
        tables (List[EmbeddingBagConfig]): list of embedding tables.
        emb_module (SplitTableBatchedEmbeddingBagsCodeGen): Fbgemm module whose optimizer state we want to expose
    Example:
        See usage in _BatchedFusedEmbeddingBag
    """

    def __init__(  # noqa C901
        self,
        tables: List[EmbeddingBagConfig],
        emb_module: SplitTableBatchedEmbeddingBagsCodegen,
    ) -> None:
        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = emb_module

        # pyre-ignore [33]
        state: Dict[Any, Any] = {}
        param_group: Dict[str, Any] = {
            "params": [],
            "lr": emb_module.optimizer_args.learning_rate,
        }

        params: Dict[str, torch.Tensor] = {}

        # Fused optimizers use buffers (they don't use autograd) and we want to make sure
        # that state_dict look identical to no-fused version.
        split_embedding_weights = emb_module.split_embedding_weights()
        split_optimizer_states = emb_module.split_optimizer_states()

        for embedding_weight, optimizer_states, table in zip(
            split_embedding_weights, split_optimizer_states, tables
        ):
            weight = embedding_weight

            state[weight] = {}
            param_group["params"].append(weight)
            param_key = table.name + ".weight"
            params[param_key] = weight

            if len(optimizer_states) >= 1:
                state[weight][f"{param_key}.momentum_1"] = optimizer_states[0]
            if len(optimizer_states) >= 2:
                state[weight][f"{param_key}.momentum_2"] = optimizer_states[1]

        super().__init__(params, state, [param_group])

    def zero_grad(self, set_to_none: bool = False) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        # pyre-ignore [16]
        self._emb_module.set_learning_rate(self.param_groups[0]["lr"])


class _BatchedFusedEmbeddingBag(nn.Module, FusedOptimizerModule):
    """
    _BatchedFusedEmbeddingBag is a thin wrapper we have around SplitTableBatchedEmbeddingBagsCodegen.
    This is not meant to be directly used. Instead use FusedEmbeddingBagCollection (which in turn utilizes this).

    Example
    -------
    >>> See usage in FusedEmbeddingBagcCollection

    """

    def __init__(
        self,
        embedding_tables: List[EmbeddingBagConfig],
        data_type: DataType,
        pooling: PoolingType,
        optimizer_type: EmbOptimType,
        optimizer_kwargs: Dict[str, Any],
        device: torch.device,
        embedding_location: EmbeddingLocation,
    ) -> None:
        super().__init__()

        self._rows: List[int] = []
        self._weight_init_mins: List[float] = []
        self._weight_init_maxs: List[float] = []
        self._num_embeddings: List[int] = []
        self._cols: List[int] = []
        self._feature_table_map: List[int] = []
        self._emb_names: List[str] = []
        self._embedding_tables = embedding_tables

        for idx, table in enumerate(embedding_tables):
            self._rows.append(table.num_embeddings)
            self._weight_init_mins.append(table.get_weight_init_min())
            self._weight_init_maxs.append(table.get_weight_init_max())
            self._num_embeddings.append(table.num_embeddings)
            self._cols.append(table.embedding_dim)
            self._feature_table_map.extend([idx] * table.num_features())

        compute_device = ComputeDevice.CPU
        if device.type == "cuda":
            compute_device = ComputeDevice.CUDA

        self._emb_module: SplitTableBatchedEmbeddingBagsCodegen = (
            SplitTableBatchedEmbeddingBagsCodegen(
                list(
                    zip(
                        self._num_embeddings,
                        self._cols,
                        [embedding_location] * len(embedding_tables),
                        [compute_device] * len(embedding_tables),
                    )
                ),
                feature_table_map=self._feature_table_map,
                pooling_mode=pooling_type_to_pooling_mode(pooling),
                device=device,
                optimizer=optimizer_type,
                **optimizer_kwargs,
            )
        )
        self._optim: EmbeddingFusedOptimizer = EmbeddingFusedOptimizer(
            embedding_tables,
            self._emb_module,
        )

        self._init_parameters()

    def forward(
        self,
        values: torch.Tensor,
        offsets: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            values (torch.Tensor): Tensor containing bags of indicies into the embedding matrix.
            offsets (torch.Tensor): Starting index position of each sequence
            weights (Optional[torch.Tensor]): weights to use to calculate weighted pooling
        Returns:
            Tensor output of `(B , table_1_embedding_dim + table_2_embedding_dim + ...)`
        """

        return self._emb_module(
            indices=values,
            offsets=offsets,
            per_sample_weights=weights,
        )

    def _init_parameters(self) -> None:
        assert len(self._num_embeddings) == len(
            self._emb_module.split_embedding_weights()
        )
        for (rows, emb_dim, weight_init_min, weight_init_max, param) in zip(
            self._rows,
            self._cols,
            self._weight_init_mins,
            self._weight_init_maxs,
            self._emb_module.split_embedding_weights(),
        ):
            assert param.shape == (rows, emb_dim)
            param.data.uniform_(
                weight_init_min,
                weight_init_max,
            )

    def split_embedding_weights(self) -> List[torch.Tensor]:
        return self._emb_module.split_embedding_weights()

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        return self._optim

    def parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[nn.Parameter]:
        yield cast(nn.Parameter, self._emb_module.weights)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        combined_key = "/".join([table.name for table in self._embedding_tables])
        name = f"{combined_key}.weight"
        key = f"{prefix}.{name}" if (prefix and name) else (prefix + name)
        yield key, cast(nn.Parameter, self._emb_module.weights)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for table, param in zip(self._embedding_tables, self.split_embedding_weights()):
            name = f"{table.name}.weight"
            key = f"{prefix}.{name}" if (prefix and name) else (prefix + name)
            yield key, param

    def buffers(self, prefix: str = "", recurse: bool = True) -> Iterator[torch.Tensor]:
        yield from self.split_embedding_weights()

    def flush(self) -> None:
        self._emb_module.flush()


VALID_OPTIMIZER_KEYS: Set[str] = {
    "gradient_clipping",
    "max_gradient",
    "stochastic_rounding",
    "learning_rate",
    "eps",
    "momentum",
    "weight_decay",
    "weight_decay_mode",
    "eta",
    "beta1",
    "beta2",
}


def convert_optimizer_type_and_kwargs(
    optimizer_type: Type[torch.optim.Optimizer],
    optimizer_kwargs: Dict[str, Any],
    device: Optional[torch.device],
) -> Optional[Tuple[EmbOptimType, Dict[str, Any]]]:
    device_type = device.type if device is not None else "cpu"
    if "lr" in optimizer_kwargs:
        optimizer_kwargs["learning_rate"] = optimizer_kwargs["lr"]
        optimizer_kwargs.pop("lr")

    invalid_optimizer_kwargs = set(optimizer_kwargs.keys()).difference(
        VALID_OPTIMIZER_KEYS
    )

    if invalid_optimizer_kwargs:
        raise ValueError(f"Cannot use {invalid_optimizer_kwargs}")

    if isinstance(optimizer_type, EmbOptimType):
        return (optimizer_type, optimizer_kwargs)
    if optimizer_type == torch.optim.SGD:
        if device_type == "cuda":
            return (
                EmbOptimType.EXACT_SGD,
                optimizer_kwargs,
            )
        else:
            return (
                EmbOptimType.SGD,
                optimizer_kwargs,
            )
    # TODO the below might not be perfect, will clean up
    # if optimizer_type == torch.optim.Adam:
    # return (
    # EmbOptimType.ADAM
    # if optimizer_kwargs.get("partial_row_wise", False)
    # else EmbOptimType.PARTIAL_ROWWISE_ADAM, {}
    # )
    # if optimizer_type == torch.optim.Adagrad:
    # if optimizer_kwargs.get("exact", False):
    # if optimizer_kwargs.get("row_wise", False):
    # if optimizer_kwargs.get("weighted", False):
    # return (EmbOptimType.EXACT_ROWWISE_WEIGHTED_ADAGRAD, {})
    # return (EmbOptimType.EXACT_ROWWISE_ADAGRAD, {})
    # return EmbOptimType.EXACT_ADAGRAD
    return None


class FusedEmbeddingBagCollection(
    EmbeddingBagCollectionInterface, FusedOptimizerModule
):
    """
    FusedEmbeddingBagCollection represents a collection of pooled embeddings (`EmbeddingBags`).
    It utilizes a technique called Optimizer fusion (register the optimizer with model). The semantics
    of this is that during the backwards pass, the registered optimizer will be called.

    It processes sparse data in the form of `KeyedJaggedTensor` with values of the form
    [F X B X L] where:

    * F: features (keys)
    * B: batch size
    * L: length of sparse features (jagged)

    and outputs a `KeyedTensor` with values of the form [B x F x D] where:

    * F: features (keys)
    * D: each feature's (key's) embedding dimension
    * B: batch size

    Args:
        tables (List[EmbeddingBagConfig]): list of embedding tables.
        is_weighted (bool): whether input `KeyedJaggedTensor` is weighted.
        optimizer (Type[torch.optim.Optimizer]): fusion optimizer type
        optimizer_kwargs: Dict[str, Any]: fusion optimizer kwargs
        device (Optional[torch.device]): compute device.

    Example::

        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        table_1 = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = FusedEmeddingBagCollection(tables=[table_0, table_1], optimizer=torch.optim.SGD, optimizer_kwargs={"lr": .01})

        #        0       1        2  <-- batch
        # "f1"   [0,1] None    [2]
        # "f2"   [3]    [4]    [5,6,7]
        #  ^
        # feature

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        pooled_embeddings = ebc(features)
        print(pooled_embeddings.values())
        tensor([[-0.6149,  0.0000, -0.3176],
        [-0.8876,  0.0000, -1.5606],
        [ 1.6805,  0.0000,  0.6810],
        [-1.4206, -1.0409,  0.2249],
        [ 0.1823, -0.4697,  1.3823],
        [-0.2767, -0.9965, -0.1797],
        [ 0.8864,  0.1315, -2.0724]], grad_fn=<TransposeBackward0>)
        print(pooled_embeddings.keys())
        ['f1', 'f2']
        print(pooled_embeddings.offset_per_key())
        tensor([0, 3, 7])
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        optimizer_type: Type[torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
        location: Optional[EmbeddingLocation] = None,
    ) -> None:
        super().__init__()

        self._optimizer_type = optimizer_type
        self._optimizer_kwargs = optimizer_kwargs

        emb_optim_and_kwargs = convert_optimizer_type_and_kwargs(
            optimizer_type, optimizer_kwargs, device
        )
        if emb_optim_and_kwargs is None:
            raise ValueError(
                f"Cannot fuse optimizer_type={optimizer_type} with kwargs {optimizer_kwargs}"
            )
        (emb_optim_type, emb_opt_kwargs) = emb_optim_and_kwargs

        if location in [
            EmbeddingLocation.DEVICE,
            EmbeddingLocation.MANAGED,
            EmbeddingLocation.MANAGED_CACHING,
        ]:
            assert device is not None and device.type in [
                "cuda",
                "meta",
            ], f"Using location={location} requires device=cuda or meta"

        if device is None:
            device = torch.device("cpu")

        if location is None:
            if device.type == "cpu":
                location = EmbeddingLocation.HOST
            elif device.type == "cuda":
                location = EmbeddingLocation.DEVICE
            else:
                raise ValueError("EmbeddingLocation could not be set")

        self._is_weighted = is_weighted
        self._embedding_bag_configs = tables
        self._emb_modules: nn.ModuleList = nn.ModuleList()

        self._key_to_tables: Dict[
            Tuple[PoolingType, DataType], List[EmbeddingBagConfig]
        ] = defaultdict(list)

        self._embedding_names: List[str] = []

        self._length_per_key: List[int] = []

        for table in tables:
            self._length_per_key.extend(
                [table.embedding_dim] * len(table.feature_names)
            )

            key = (table.pooling, table.data_type)
            self._key_to_tables[key].append(table)

        optims = []
        for key, tables in self._key_to_tables.items():
            (pooling, data_type) = key
            emb_module = _BatchedFusedEmbeddingBag(
                tables,
                data_type=data_type,
                pooling=pooling,
                optimizer_type=emb_optim_type,
                optimizer_kwargs=emb_opt_kwargs,
                device=device,
                embedding_location=location,
            )
            self._emb_modules.append(emb_module)
            params: Dict[str, torch.Tensor] = {}
            for param_key, weight in emb_module.fused_optimizer.params.items():
                # pyre-ignore
                params[f"embedding_bags.{param_key}"] = weight
                optims.append(("", emb_module.fused_optimizer))

        self._optim: CombinedOptimizer = CombinedOptimizer(optims)
        self._embedding_names = list(
            itertools.chain(*get_embedding_names_by_table(tables))
        )

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            KeyedTensor
        """
        assert features is not None
        feature_dict = features.to_dict()
        embeddings = []

        for emb_op, (_key, tables) in zip(
            self._emb_modules, self._key_to_tables.items()
        ):
            indicies = []
            lengths = []
            offsets = []
            weights = []

            for table in tables:
                for feature in table.feature_names:
                    f = feature_dict[feature]
                    indicies.append(f.values())
                    lengths.append(f.lengths())
                    if self._is_weighted:
                        weights.append(f.weights())

            indicies = torch.cat(indicies)
            lengths = torch.cat(lengths)

            offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
            if self._is_weighted:
                weights = torch.cat(weights)

            embeddings.append(
                emb_op(
                    indicies.int(),
                    offsets.int(),
                    weights if self._is_weighted else None,
                )
            )

        embeddings = torch.cat(embeddings, dim=1)
        return KeyedTensor(
            keys=self._embedding_names,
            values=embeddings,
            length_per_key=self._length_per_key,
        )

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()
        for emb_op, (_key, tables) in zip(
            self._emb_modules, self._key_to_tables.items()
        ):
            for table, weight in zip(tables, emb_op.split_embedding_weights()):
                destination[prefix + f"embedding_bags.{table.name}.weight"] = weight
        return destination

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module` inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        strict: bool = True,
    ) -> _IncompatibleKeys:

        missing_keys = []
        unexpected_keys = []

        current_state_dict = self.state_dict()
        for key in current_state_dict.keys():
            if key not in state_dict:
                missing_keys.append(key)
        for key in state_dict.keys():
            if key not in current_state_dict.keys():
                unexpected_keys.append(key)

        if missing_keys or unexpected_keys:
            return _IncompatibleKeys(
                missing_keys=missing_keys, unexpected_keys=unexpected_keys
            )

        for (_key, tables) in self._key_to_tables.items():
            for table in tables:
                current_state_dict[
                    f"embedding_bags.{table.name}.weight"
                ].detach().copy_(state_dict[f"embedding_bags.{table.name}.weight"])

        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    def _get_name(self) -> str:
        return "FusedEmeddingBagCollection"

    @property
    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    @property
    def is_weighted(self) -> bool:
        return self._is_weighted

    @property
    def optimizer_type(self) -> Type[torch.optim.Optimizer]:
        return self._optimizer_type

    @property
    def optimizer_kwargs(self) -> Dict[str, Any]:
        return self._optimizer_kwargs

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim


class FusedEmbeddingCollection(EmbeddingCollection, FusedOptimizerModule):
    pass


def fuse_optimizer(
    embedding_module: Union[EmbeddingBagCollection, EmbeddingCollection],
    optimizer_type: Type[torch.optim.Optimizer],
    optimizer_kwargs: Dict[str, Any],
    device: torch.device,
    location: Optional[EmbeddingLocation] = None,
) -> Union[FusedEmbeddingBagCollection, FusedEmbeddingCollection]:

    if isinstance(embedding_module, EmbeddingBagCollection):
        return FusedEmbeddingBagCollection(
            embedding_module.embedding_bag_configs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
            location=location,
        )
    elif isinstance(embedding_module, EmbeddingCollection):
        raise NotImplementedError()
    raise ValueError(
        "Only EmbeddingBagCollections and EmbeddingCollections can have operators fused to them"
    )


def fuse_embedding_optimizer(
    model: nn.Module,
    optimizer_type: Type[torch.optim.Optimizer],
    optimizer_kwargs: Dict[str, Any],
    device: torch.device,
) -> None:
    # TODO
    # This module will replace all EBCs and ECs with a corresponding FusedEmbeddingModule. The passed in module can be anything that contains an EBC/EC
    raise NotImplementedError()
