#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict

import torch
from torchrec.fb.modules.hash_mc_evictions import (
    HashZchEvictionConfig,
    HashZchEvictionPolicyName,
)
from torchrec.fb.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.mc_modules import (
    average_threshold_filter,
    DistanceLFU_EvictionPolicy,
    dynamic_threshold_filter,
    LFU_EvictionPolicy,
    LRU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
    probabilistic_threshold_filter,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class TestEvictionPolicy(unittest.TestCase):
    def test_lfu_eviction(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_counts = mc_module._mch_counts
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_counts), [0] * 5)

        # insert some values to zch
        # we have 10 counts of 4 and 1 count of 5
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedSeque...
        mc_module._mch_sorted_raw_ids[0:2] = torch.tensor([4, 5])
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedSeque...
        mc_module._mch_counts[0:2] = torch.tensor([10, 1])

        ids = [3, 4, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 10]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        # 5, empty, empty, empty will be evicted
        # 6, 7, 8 will be added
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
            #  `Union[Tensor, Module]`.
            list(_mch_sorted_raw_ids),
            [4, 6, 7, 8, torch.iinfo(torch.int64).max],
        )
        # 11 counts of 5, 3 counts of 6, 3 counts of 7, 3 counts of 8
        _mch_counts = mc_module._mch_counts
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_counts), [11, 3, 3, 3, torch.iinfo(torch.int64).max])

    def test_lru_eviction(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=LRU_EvictionPolicy(decay_exponent=1.0),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_last_access_iter = mc_module._mch_last_access_iter
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_last_access_iter), [0] * 5)

        ids = [5, 6, 7]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)
        self.assertEqual(mc_module.open_slots().item(), 1)
        ids = [3, 4, 5]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)
        self.assertEqual(mc_module.open_slots().item(), 0)
        ids = [7, 8]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)
        self.assertEqual(mc_module.open_slots().item(), 0)

        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
            #  `Union[Tensor, Module]`.
            list(_mch_sorted_raw_ids),
            [3, 4, 7, 8, torch.iinfo(torch.int64).max],
        )
        _mch_last_access_iter = mc_module._mch_last_access_iter
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_last_access_iter), [2, 2, 3, 3, 3])
        self.assertEqual(mc_module.open_slots().item(), 0)

    def test_distance_lfu_eviction(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=DistanceLFU_EvictionPolicy(decay_exponent=1.0),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_counts = mc_module._mch_counts
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_counts), [0] * 5)
        _mch_last_access_iter = mc_module._mch_last_access_iter
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_last_access_iter), [0] * 5)

        ids = [5, 5, 5, 5, 5, 6]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        ids = [3, 4]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        ids = [7, 8]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
            #  `Union[Tensor, Module]`.
            list(_mch_sorted_raw_ids),
            [3, 5, 7, 8, torch.iinfo(torch.int64).max],
        )
        _mch_counts = mc_module._mch_counts
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_counts), [1, 5, 1, 1, torch.iinfo(torch.int64).max])
        _mch_last_access_iter = mc_module._mch_last_access_iter
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_last_access_iter), [2, 1, 3, 3, 3])

    def test_distance_lfu_eviction_fast_decay(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=DistanceLFU_EvictionPolicy(decay_exponent=10.0),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_counts = mc_module._mch_counts
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_counts), [0] * 5)
        _mch_last_access_iter = mc_module._mch_last_access_iter
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_last_access_iter), [0] * 5)

        ids = [5, 5, 5, 5, 5, 6]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        ids = [3, 4]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        ids = [7, 8]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
            #  `Union[Tensor, Module]`.
            list(_mch_sorted_raw_ids),
            [3, 4, 7, 8, torch.iinfo(torch.int64).max],
        )
        _mch_counts = mc_module._mch_counts
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_counts), [1, 1, 1, 1, torch.iinfo(torch.int64).max])
        _mch_last_access_iter = mc_module._mch_last_access_iter
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_last_access_iter), [2, 2, 3, 3, 3])

    def test_dynamic_threshold_filter(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(
                threshold_filtering_func=lambda tensor: dynamic_threshold_filter(
                    tensor, threshold_skew_multiplier=0.75
                )
            ),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_counts = mc_module._mch_counts
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_counts), [0] * 5)

        ids = [5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 1]
        # threshold is len(ids) / unique_count(ids) * threshold_skew_multiplier
        # = 15 / 5 * 0.5 = 2.25
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
            #  `Union[Tensor, Module]`.
            list(_mch_sorted_raw_ids),
            [3, 4, 5, torch.iinfo(torch.int64).max, torch.iinfo(torch.int64).max],
        )
        _mch_counts = mc_module._mch_counts
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_counts), [3, 4, 5, 0, torch.iinfo(torch.int64).max])

    def test_average_threshold_filter(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(
                threshold_filtering_func=average_threshold_filter
            ),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_counts = mc_module._mch_counts
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_counts), [0] * 5)

        # insert some values to zch
        # we have 10 counts of 4 and 1 count of 5
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedSeque...
        mc_module._mch_sorted_raw_ids[0:2] = torch.tensor([4, 5])
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _NestedSeque...
        mc_module._mch_counts[0:2] = torch.tensor([10, 1])

        ids = [3, 4, 5, 6, 6, 6, 7, 8, 8, 9, 10]
        # threshold is 1.375
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }
        mc_module.profile(features)

        # empty, empty will be evicted
        # 6, 8 will be added
        # 7 is not added because it's below the average threshold
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
            #  `Union[Tensor, Module]`.
            list(_mch_sorted_raw_ids),
            [4, 5, 6, 8, torch.iinfo(torch.int64).max],
        )
        # count for 4 is not updated since it's below the average threshold
        _mch_counts = mc_module._mch_counts
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_counts), [10, 1, 3, 2, torch.iinfo(torch.int64).max])

    def test_probabilistic_threshold_filter(self) -> None:
        mc_module = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(
                threshold_filtering_func=lambda tensor: probabilistic_threshold_filter(
                    tensor,
                    per_id_probability=0.01,
                )
            ),
            eviction_interval=1,
            input_hash_size=100,
        )

        # check initial state
        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_sorted_raw_ids), [torch.iinfo(torch.int64).max] * 5)
        _mch_counts = mc_module._mch_counts
        # pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got
        #  `Union[Tensor, Module]`.
        self.assertEqual(list(_mch_counts), [0] * 5)

        unique_ids = [5, 4, 3, 2, 1]
        id_counts = [100, 80, 60, 40, 10]
        ids = [id for id, count in zip(unique_ids, id_counts) for _ in range(count)]
        # chance of being added is [0.63, 0.55, 0.45, 0.33]
        features: Dict[str, JaggedTensor] = {
            "f1": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([1] * len(ids), dtype=torch.int64),
            )
        }

        torch.manual_seed(42)
        for _ in range(10):
            mc_module.profile(features)

        _mch_sorted_raw_ids = mc_module._mch_sorted_raw_ids
        self.assertEqual(
            # pyre-fixme[29]: `Union[(self: TensorBase) -> list[Any], Tensor,
            #  Module]` is not a function.
            sorted(_mch_sorted_raw_ids.tolist()),
            [2, 3, 4, 5, torch.iinfo(torch.int64).max],
        )
        # _mch_counts is like
        # [80, 180, 160, 800, 9223372036854775807]

    def test_fx_jit_script_not_training(self) -> None:
        model = MCHManagedCollisionModule(
            zch_size=5,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(),
            eviction_interval=1,
            input_hash_size=100,
        )

        model.train(False)
        gm = torch.fx.symbolic_trace(model)
        torch.jit.script(gm)

    def test_mc_module_forward(self) -> None:
        embedding_configs = [
            EmbeddingConfig(
                name="t1",
                num_embeddings=100,
                embedding_dim=8,
                feature_names=["f1", "f2"],
            ),
            EmbeddingConfig(
                name="t2",
                num_embeddings=100,
                embedding_dim=8,
                feature_names=["f3", "f4"],
            ),
        ]

        mc_modules = {
            "t1": HashZchManagedCollisionModule(
                zch_size=100,
                device=torch.device("cpu"),
                total_num_buckets=1,
                eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
                eviction_config=HashZchEvictionConfig(
                    features=[],
                    single_ttl=10,
                ),
            ),
            "t2": HashZchManagedCollisionModule(
                zch_size=100,
                device=torch.device("cpu"),
                total_num_buckets=1,
                eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
                eviction_config=HashZchEvictionConfig(
                    features=[],
                    single_ttl=10,
                ),
            ),
        }
        for mc_module in mc_modules.values():
            mc_module.reset_inference_mode()
        mc_ebc = ManagedCollisionCollection(
            # Pyre-ignore [6]: In call `ManagedCollisionCollection.__init__`, for argument `managed_collision_modules`, expected `Dict[str, ManagedCollisionModule]` but got `Dict[str, HashZchManagedCollisionModule]`
            managed_collision_modules=mc_modules,
            embedding_configs=embedding_configs,
        )
        kjt = KeyedJaggedTensor(
            keys=["f1", "f2", "f3", "f4"],
            values=torch.cat(
                [
                    torch.arange(0, 20, 2, dtype=torch.int64, device="cpu"),
                    torch.arange(30, 60, 3, dtype=torch.int64, device="cpu"),
                    torch.arange(20, 30, 2, dtype=torch.int64, device="cpu"),
                    torch.arange(0, 20, 2, dtype=torch.int64, device="cpu"),
                ]
            ),
            lengths=torch.cat(
                [
                    torch.tensor([4, 6], dtype=torch.int64, device="cpu"),
                    torch.tensor([5, 5], dtype=torch.int64, device="cpu"),
                    torch.tensor([1, 4], dtype=torch.int64, device="cpu"),
                    torch.tensor([7, 3], dtype=torch.int64, device="cpu"),
                ]
            ),
        )
        res = mc_ebc.forward(kjt)
        self.assertTrue(torch.equal(res.lengths(), kjt.lengths()))
        self.assertTrue(
            torch.equal(
                res.lengths(), torch.tensor([4, 6, 5, 5, 1, 4, 7, 3], dtype=torch.int64)
            )
        )
