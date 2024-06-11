#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast, List, OrderedDict, Union

import torch
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import given, settings, strategies as st, Verbosity
from torchrec.distributed.batched_embedding_kernel import (
    KeyValueEmbedding,
    KeyValueEmbeddingBag,
)
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    ShardedEmbeddingTable,
)
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.test_utils.test_model_parallel_base import (
    ModelParallelSingleRankBase,
)
from torchrec.distributed.test_utils.test_sharding import (
    copy_state_dict,
    create_test_sharder,
    SharderType,
)
from torchrec.distributed.tests.test_sequence_model import (
    TestEmbeddingCollectionSharder,
    TestSequenceSparseNN,
)
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
)


def _load_split_embedding_weights(
    emb_module: Union[KeyValueEmbedding, KeyValueEmbeddingBag],
    weights: List[torch.Tensor],
) -> None:
    """
    Util function to set the weights of SSD TBE.
    """
    embedding_tables: List[ShardedEmbeddingTable] = emb_module.config.embedding_tables

    assert len(weights) == len(
        embedding_tables
    ), "Expect length of weights to be equal to number of embedding tables. "

    cum_sum = 0
    for table_id, (table, weight) in enumerate(zip(embedding_tables, weights)):
        # load weights for SSD TBE
        height = weight.shape[0]
        shard_shape = table.local_rows, table.local_cols
        assert shard_shape == weight.shape, "Expect shard shape to match tensor shape."
        assert weight.device == torch.device("cpu"), "Weight has to be on CPU."
        emb_module.emb_module.ssd_db.set_cuda(
            torch.arange(cum_sum, cum_sum + height, dtype=torch.int64),
            weight,
            torch.as_tensor([height]),
            table_id,
        )
        cum_sum += height


class KeyValueModelParallelTest(ModelParallelSingleRankBase):
    def _create_tables(self) -> None:
        num_features = 4
        self.tables += [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 1000,
                embedding_dim=256,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

    @staticmethod
    def _copy_ssd_emb_modules(
        m1: DistributedModelParallel, m2: DistributedModelParallel
    ) -> None:
        """
        Util function to copy and set the SSD TBE modules of two models. It
        requires both DMP modules to have the same sharding plan.
        """
        for lookup1, lookup2 in zip(
            m1.module.sparse.ebc._lookups, m2.module.sparse.ebc._lookups
        ):
            for emb_module1, emb_module2 in zip(
                lookup1._emb_modules, lookup2._emb_modules
            ):
                ssd_emb_modules = {KeyValueEmbeddingBag, KeyValueEmbedding}
                if type(emb_module1) in ssd_emb_modules:
                    assert type(emb_module1) is type(emb_module2), (
                        "Expect two emb_modules to be of the same type, either both "
                        "SSDEmbeddingBag or SSDEmbeddingBag."
                    )

                    weights = emb_module1.emb_module.debug_split_embedding_weights()
                    # need to set emb_module1 as well, since otherwise emb_module1 would
                    # produce a random debug_split_embedding_weights everytime
                    _load_split_embedding_weights(emb_module1, weights)
                    _load_split_embedding_weights(emb_module2, weights)

                    # purge after loading. This is needed, since we pass a batch
                    # through dmp when instantiating them.
                    emb_module1.purge()
                    emb_module2.purge()

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    # pyre-ignore[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.KEY_VALUE.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        is_training=st.booleans(),
        stochastic_rounding=st.booleans(),
        dtype=st.sampled_from([DataType.FP32, DataType.FP16]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_ssd_load_state_dict(
        self,
        sharder_type: str,
        kernel_type: str,
        sharding_type: str,
        is_training: bool,
        stochastic_rounding: bool,
        dtype: DataType,
    ) -> None:
        """
        This test checks that if SSD TBE is deterministic. That is, if two SSD
        TBEs start with the same state, they would produce the same output.
        """
        self._set_table_weights_precision(dtype)

        fused_params = {
            "learning_rate": 0.1,
            "stochastic_rounding": stochastic_rounding,
        }
        is_deterministic = dtype == DataType.FP32 or not stochastic_rounding
        constraints = {
            table.name: ParameterConstraints(
                sharding_types=[sharding_type],
                compute_kernels=[kernel_type],
            )
            for i, table in enumerate(self.tables)
        }
        sharders = [
            create_test_sharder(
                sharder_type,
                sharding_type,
                kernel_type,
                fused_params=fused_params,
            ),
        ]

        # pyre-ignore
        models, batch = self._generate_dmps_and_batch(sharders, constraints=constraints)
        m1, m2 = models

        # load state dict for dense modules
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))
        self._copy_ssd_emb_modules(m1, m2)

        if is_training:
            self._train_models(m1, m2, batch)
        self._eval_models(m1, m2, batch, is_deterministic=is_deterministic)
        self._compare_models(m1, m2, is_deterministic=is_deterministic)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    # pyre-ignore[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.KEY_VALUE.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        is_training=st.booleans(),
        stochastic_rounding=st.booleans(),
        dtype=st.sampled_from([DataType.FP32, DataType.FP16]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_ssd_tbe_numerical_accuracy(
        self,
        sharder_type: str,
        kernel_type: str,
        sharding_type: str,
        is_training: bool,
        stochastic_rounding: bool,
        dtype: DataType,
    ) -> None:
        """
        Make sure it produces same numbers as normal TBE.
        """
        self._set_table_weights_precision(dtype)

        base_kernel_type = EmbeddingComputeKernel.FUSED.value
        learning_rate = 0.1
        fused_params = {
            "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            "learning_rate": learning_rate,
            "stochastic_rounding": stochastic_rounding,
        }
        is_deterministic = dtype == DataType.FP32 or not stochastic_rounding
        fused_sharders = [
            cast(
                ModuleSharder[nn.Module],
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    base_kernel_type,  # base kernel type
                    fused_params=fused_params,
                ),
            ),
        ]
        ssd_sharders = [
            cast(
                ModuleSharder[nn.Module],
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    fused_params=fused_params,
                ),
            ),
        ]
        ssd_constraints = {
            table.name: ParameterConstraints(
                sharding_types=[sharding_type],
                compute_kernels=[kernel_type],
            )
            for i, table in enumerate(self.tables)
        }
        (fused_model, _), _ = self._generate_dmps_and_batch(fused_sharders)
        (ssd_model, _), batch = self._generate_dmps_and_batch(
            ssd_sharders, constraints=ssd_constraints
        )

        # load state dict for dense modules
        copy_state_dict(
            ssd_model.state_dict(), fused_model.state_dict(), exclude_predfix="sparse"
        )

        # for this to work, we expect the order of lookups to be the same
        assert len(fused_model.module.sparse.ebc._lookups) == len(
            ssd_model.module.sparse.ebc._lookups
        ), "Expect same number of lookups"

        for fused_lookup, ssd_lookup in zip(
            fused_model.module.sparse.ebc._lookups, ssd_model.module.sparse.ebc._lookups
        ):
            assert len(fused_lookup._emb_modules) == len(
                ssd_lookup._emb_modules
            ), "Expect same number of emb modules"
            for fused_emb_module, ssd_emb_module in zip(
                fused_lookup._emb_modules, ssd_lookup._emb_modules
            ):
                weights = fused_emb_module.split_embedding_weights()
                weights = [weight.to("cpu") for weight in weights]
                _load_split_embedding_weights(ssd_emb_module, weights)

                # purge after loading. This is needed, since we pass a batch
                # through dmp when instantiating them.
                ssd_emb_module.purge()

        if is_training:
            self._train_models(fused_model, ssd_model, batch)
        self._eval_models(
            fused_model, ssd_model, batch, is_deterministic=is_deterministic
        )

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    # pyre-ignore[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.KEY_VALUE.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        is_training=st.booleans(),
        stochastic_rounding=st.booleans(),
        dtype=st.sampled_from([DataType.FP32, DataType.FP16]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_ssd_fused_optimizer(
        self,
        sharder_type: str,
        kernel_type: str,
        sharding_type: str,
        is_training: bool,
        stochastic_rounding: bool,
        dtype: DataType,
    ) -> None:
        """
        Purpose of this test is to make sure it works with warm up policy.
        """
        self._set_table_weights_precision(dtype)

        is_deterministic = dtype == DataType.FP32 or not stochastic_rounding

        constraints = {
            table.name: ParameterConstraints(
                sharding_types=[sharding_type],
                compute_kernels=[kernel_type],
            )
            for i, table in enumerate(self.tables)
        }

        base_sharders = [
            create_test_sharder(
                sharder_type,
                sharding_type,
                kernel_type,
                fused_params={
                    "learning_rate": 0.2,
                    "stochastic_rounding": stochastic_rounding,
                },
            ),
        ]
        models, batch = self._generate_dmps_and_batch(
            base_sharders,  # pyre-ignore
            constraints=constraints,
        )
        base_model, _ = models

        test_sharders = [
            create_test_sharder(
                sharder_type,
                sharding_type,
                kernel_type,
                fused_params={
                    "learning_rate": 0.1,
                    "stochastic_rounding": stochastic_rounding,
                },
            ),
        ]
        models, _ = self._generate_dmps_and_batch(
            test_sharders,  # pyre-ignore
            constraints=constraints,
        )
        test_model, _ = models

        # load state dict for dense modules
        test_model.load_state_dict(
            cast("OrderedDict[str, torch.Tensor]", base_model.state_dict())
        )
        self._copy_ssd_emb_modules(base_model, test_model)

        self._eval_models(
            base_model, test_model, batch, is_deterministic=is_deterministic
        )

        # change learning rate for test_model
        fused_opt = test_model.fused_optimizer
        # pyre-ignore
        fused_opt.param_groups[0]["lr"] = 0.2
        fused_opt.zero_grad()

        if is_training:
            self._train_models(base_model, test_model, batch)
        self._eval_models(
            base_model, test_model, batch, is_deterministic=is_deterministic
        )
        self._compare_models(base_model, test_model, is_deterministic=is_deterministic)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    # pyre-ignore[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.KEY_VALUE.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        is_training=st.booleans(),
        stochastic_rounding=st.booleans(),
        dtype=st.sampled_from([DataType.FP32, DataType.FP16]),
        fused_first=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_ssd_mixed_kernels(
        self,
        sharder_type: str,
        kernel_type: str,
        sharding_type: str,
        is_training: bool,
        stochastic_rounding: bool,
        dtype: DataType,
        fused_first: bool,
    ) -> None:
        """
        Purpose of this test is to make sure it works with warm up policy.
        """
        self._set_table_weights_precision(dtype)

        base_kernel_type = EmbeddingComputeKernel.FUSED.value

        is_deterministic = dtype == DataType.FP32 or not stochastic_rounding

        constraints = {
            table.name: ParameterConstraints(
                sharding_types=[sharding_type],
                compute_kernels=(
                    [base_kernel_type] if i % 2 == fused_first else [kernel_type]
                ),
            )
            for i, table in enumerate(self.tables)
        }

        fused_params = {
            "learning_rate": 0.1,
            "stochastic_rounding": stochastic_rounding,
        }
        sharders = [
            EmbeddingBagCollectionSharder(fused_params=fused_params),
        ]

        # pyre-ignore
        models, batch = self._generate_dmps_and_batch(sharders, constraints=constraints)
        m1, m2 = models

        # load state dict for dense modules
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))
        self._copy_ssd_emb_modules(m1, m2)

        if is_training:
            self._train_models(m1, m2, batch)
        self._eval_models(m1, m2, batch, is_deterministic=is_deterministic)
        self._compare_models(m1, m2, is_deterministic=is_deterministic)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    # pyre-ignore[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.KEY_VALUE.value,
            ]
        ),
        is_training=st.booleans(),
        stochastic_rounding=st.booleans(),
        dtype=st.sampled_from([DataType.FP32, DataType.FP16]),
        table_wise_first=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_ssd_mixed_sharding_types(
        self,
        sharder_type: str,
        kernel_type: str,
        is_training: bool,
        stochastic_rounding: bool,
        dtype: DataType,
        table_wise_first: bool,
    ) -> None:
        """
        Purpose of this test is to make sure it works with warm up policy.
        """
        self._set_table_weights_precision(dtype)

        is_deterministic = dtype == DataType.FP32 or not stochastic_rounding

        constraints = {
            table.name: ParameterConstraints(
                sharding_types=(
                    [ShardingType.TABLE_WISE.value]
                    if i % 2 == table_wise_first
                    else [ShardingType.ROW_WISE.value]
                ),
                compute_kernels=[kernel_type],
            )
            for i, table in enumerate(self.tables)
        }

        fused_params = {
            "learning_rate": 0.1,
            "stochastic_rounding": stochastic_rounding,
        }
        sharders = [
            EmbeddingBagCollectionSharder(fused_params=fused_params),
        ]

        # pyre-ignore
        models, batch = self._generate_dmps_and_batch(sharders, constraints=constraints)
        m1, m2 = models

        # load state dict for dense modules
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))
        self._copy_ssd_emb_modules(m1, m2)

        if is_training:
            self._train_models(m1, m2, batch)
        self._eval_models(m1, m2, batch, is_deterministic=is_deterministic)
        self._compare_models(m1, m2, is_deterministic=is_deterministic)


class KeyValueSequenceModelParallelStateDictTest(ModelParallelSingleRankBase):
    def setUp(self, backend: str = "nccl") -> None:
        self.shared_features = []
        self.embedding_groups = {}

        super().setUp(backend=backend)

    def _create_tables(self) -> None:
        num_features = 4
        shared_features = 2

        initial_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 1000,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

        shared_features_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i + num_features),
                feature_names=["feature_" + str(i)],
            )
            for i in range(shared_features)
        ]

        self.tables += initial_tables + shared_features_tables
        self.shared_features += [f"feature_{i}" for i in range(shared_features)]

        self.embedding_groups["group_0"] = [
            (f"{feature}@{table.name}" if feature in self.shared_features else feature)
            for table in self.tables
            for feature in table.feature_names
        ]

    def _create_model(self) -> nn.Module:
        return TestSequenceSparseNN(
            tables=self.tables,
            num_float_features=self.num_float_features,
            embedding_groups=self.embedding_groups,
            dense_device=self.device,
            sparse_device=torch.device("meta"),
        )

    @staticmethod
    def _copy_ssd_emb_modules(
        m1: DistributedModelParallel, m2: DistributedModelParallel
    ) -> None:
        """
        Util function to copy and set the SSD TBE modules of two models. It
        requires both DMP modules to have the same sharding plan.
        """
        for lookup1, lookup2 in zip(
            m1.module.sparse.ec._lookups, m2.module.sparse.ec._lookups
        ):
            for emb_module1, emb_module2 in zip(
                lookup1._emb_modules, lookup2._emb_modules
            ):
                ssd_emb_modules = {KeyValueEmbeddingBag, KeyValueEmbedding}
                if type(emb_module1) in ssd_emb_modules:
                    assert type(emb_module1) is type(emb_module2), (
                        "Expect two emb_modules to be of the same type, either both "
                        "SSDEmbeddingBag or SSDEmbeddingBag."
                    )

                    weights = emb_module1.emb_module.debug_split_embedding_weights()
                    # need to set emb_module1 as well, since otherwise emb_module1 would
                    # produce a random debug_split_embedding_weights everytime
                    _load_split_embedding_weights(emb_module1, weights)
                    _load_split_embedding_weights(emb_module2, weights)

                    # purge after loading. This is needed, since we pass a batch
                    # through dmp when instantiating them.
                    emb_module1.purge()
                    emb_module2.purge()

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    # pyre-ignore[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.KEY_VALUE.value,
            ]
        ),
        is_training=st.booleans(),
        stochastic_rounding=st.booleans(),
        dtype=st.sampled_from([DataType.FP32, DataType.FP16]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_ssd_load_state_dict(
        self,
        sharding_type: str,
        kernel_type: str,
        is_training: bool,
        stochastic_rounding: bool,
        dtype: DataType,
    ) -> None:
        """
        This test checks that if SSD TBE is deterministic. That is, if two SSD
        TBEs start with the same state, they would produce the same output.
        """
        self._set_table_weights_precision(dtype)

        fused_params = {
            "learning_rate": 0.1,
            "stochastic_rounding": stochastic_rounding,
        }
        is_deterministic = dtype == DataType.FP32 or not stochastic_rounding
        sharders = [
            cast(
                ModuleSharder[nn.Module],
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    fused_params=fused_params,
                ),
            ),
        ]

        constraints = {
            table.name: ParameterConstraints(
                sharding_types=[sharding_type],
                compute_kernels=[kernel_type],
            )
            for i, table in enumerate(self.tables)
        }

        models, batch = self._generate_dmps_and_batch(sharders, constraints=constraints)
        m1, m2 = models

        # load state dict for dense modules
        m2.load_state_dict(cast("OrderedDict[str, torch.Tensor]", m1.state_dict()))
        self._copy_ssd_emb_modules(m1, m2)

        if is_training:
            self._train_models(m1, m2, batch)
        self._eval_models(m1, m2, batch, is_deterministic=is_deterministic)
        self._compare_models(m1, m2, is_deterministic=is_deterministic)


# TODO: remove after development is done
def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()
