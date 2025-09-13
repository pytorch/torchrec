#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast
from unittest.mock import patch

import torch
from hypothesis import given, settings, strategies as st
from pyre_extensions import none_throws
from torchrec.distributed.embedding_sharding import bucketize_kjt_before_all2all
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.hash_mc_evictions import (
    HashZchEvictionConfig,
    HashZchEvictionPolicyName,
)
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.mc_modules import (
    ManagedCollisionCollection,
    ManagedCollisionModule,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class TestMCH(unittest.TestCase):
    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_zch_hash_inference(self) -> None:
        # prepare
        m1 = HashZchManagedCollisionModule(
            zch_size=20,
            device=torch.device("cuda"),
            total_num_buckets=2,
            eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
            eviction_config=HashZchEvictionConfig(
                features=[],
                single_ttl=10,
            ),
        )
        self.assertEqual(m1._hash_zch_identities.dtype, torch.int64)
        in1 = {
            "f": JaggedTensor(
                values=torch.arange(0, 20, 2, dtype=torch.int64, device="cuda"),
                lengths=torch.tensor([4, 6], dtype=torch.int64, device="cuda"),
            ),
        }
        o1 = m1(in1)["f"].values()
        self.assertTrue(
            torch.equal(torch.unique(o1), torch.arange(0, 10, device="cuda")),
            f"{torch.unique(o1)=}",
        )

        in2 = {
            "f": JaggedTensor(
                values=torch.arange(1, 20, 2, dtype=torch.int64, device="cuda"),
                lengths=torch.tensor([8, 2], dtype=torch.int64, device="cuda"),
            ),
        }
        o2 = m1(in2)["f"].values()
        self.assertTrue(
            torch.equal(torch.unique(o2), torch.arange(10, 20, device="cuda")),
            f"{torch.unique(o2)=}",
        )

        for device_str in ["cpu", "cuda"]:
            # Inference
            m_infer = HashZchManagedCollisionModule(
                zch_size=20,
                device=torch.device(device_str),
                total_num_buckets=2,
            )

            m_infer.reset_inference_mode()
            m_infer.to(device_str)

            self.assertTrue(
                torch.equal(
                    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                    #  `Union[Tensor, Module]`.
                    none_throws(m_infer.input_mapper._zch_size_per_training_rank),
                    torch.tensor([10, 10], dtype=torch.int64, device=device_str),
                )
            )
            self.assertTrue(
                torch.equal(
                    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                    #  `Union[Tensor, Module]`.
                    none_throws(m_infer.input_mapper._train_rank_offsets),
                    torch.tensor([0, 10], dtype=torch.int64, device=device_str),
                )
            )

            m_infer._hash_zch_identities = torch.nn.Parameter(
                m1._hash_zch_identities[:, :1],
                requires_grad=False,
            )
            in12 = {
                "f": JaggedTensor(
                    values=torch.arange(0, 20, dtype=torch.int64, device=device_str),
                    lengths=torch.tensor(
                        [4, 6, 8, 2], dtype=torch.int64, device=device_str
                    ),
                ),
            }
            m_infer = torch.jit.script(m_infer)
            o_infer = m_infer(in12)["f"].values()
            o12 = torch.stack([o1, o2], dim=1).view(-1).to(device_str)
            self.assertTrue(torch.equal(o_infer, o12), f"{o_infer=} vs {o12=}")

        m3 = HashZchManagedCollisionModule(
            zch_size=10,
            device=torch.device("cuda"),
            total_num_buckets=2,
            eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
            eviction_config=HashZchEvictionConfig(
                features=[],
                single_ttl=10,
            ),
        )
        self.assertEqual(m3._hash_zch_identities.dtype, torch.int64)
        in3 = {
            "f": JaggedTensor(
                values=torch.arange(10, 20, dtype=torch.int64, device="cuda"),
                lengths=torch.tensor([4, 6], dtype=torch.int64, device="cuda"),
            ),
        }
        o3 = m3(in3)["f"].values()
        self.assertTrue(
            torch.equal(torch.unique(o3), torch.arange(0, 10, device="cuda")),
            f"{torch.unique(o3)=}",
        )
        # validate that original ids are assigned to identities
        self.assertTrue(
            torch.equal(
                torch.unique(m3._hash_zch_identities),
                torch.arange(10, 20, device="cuda"),
            ),
            f"{torch.unique(m3._hash_zch_identities)=}",
        )

    def test_scriptability(self) -> None:
        zch_size = 10
        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                HashZchManagedCollisionModule(
                    zch_size=zch_size,
                    device=torch.device("cpu"),
                    eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
                    eviction_config=HashZchEvictionConfig(
                        features=["feature"],
                    ),
                    total_num_buckets=2,
                ),
            )
        }

        embedding_configs = [
            EmbeddingConfig(
                name="t1",
                embedding_dim=8,
                num_embeddings=zch_size,
                feature_names=["f1", "f2"],
            ),
        ]

        mcc_ec = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            embedding_configs=embedding_configs,
        )
        torch.jit.script(mcc_ec)

    def test_scriptability_lru(self) -> None:
        zch_size = 10
        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                HashZchManagedCollisionModule(
                    zch_size=zch_size,
                    device=torch.device("cpu"),
                    total_num_buckets=2,
                    eviction_policy_name=HashZchEvictionPolicyName.LRU_EVICTION,
                    eviction_config=HashZchEvictionConfig(
                        features=["feature"],
                        single_ttl=12,
                    ),
                ),
            )
        }

        embedding_configs = [
            EmbeddingConfig(
                name="t1",
                embedding_dim=8,
                num_embeddings=zch_size,
                feature_names=["f1", "f2"],
            ),
        ]

        mcc_ec = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            embedding_configs=embedding_configs,
        )
        torch.jit.script(mcc_ec)

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-ignore [56]
    @given(hash_size=st.sampled_from([0, 80]), keep_original_indices=st.booleans())
    @settings(max_examples=6, deadline=None)
    def test_zch_hash_train_to_inf_block_bucketize_disabled_in_oss_compatibility(
        self, hash_size: int, keep_original_indices: bool
    ) -> None:
        # rank 0
        world_size = 2
        kjt = KeyedJaggedTensor(
            keys=["f"],
            values=torch.cat(
                [
                    torch.arange(0, 20, 2, dtype=torch.int64, device="cuda"),
                    torch.arange(30, 60, 3, dtype=torch.int64, device="cuda"),
                ]
            ),
            lengths=torch.cat(
                [
                    torch.tensor([4, 6], dtype=torch.int64, device="cuda"),
                    torch.tensor([4, 6], dtype=torch.int64, device="cuda"),
                ]
            ),
        )
        block_sizes = torch.tensor(
            [(size + world_size - 1) // world_size for size in [hash_size]],
            dtype=torch.int64,
            device="cuda",
        )

        bucketized_kjt, _ = bucketize_kjt_before_all2all(
            kjt,
            num_buckets=world_size,
            block_sizes=block_sizes,
            keep_original_indices=keep_original_indices,
        )
        in1, in2 = bucketized_kjt.split([len(kjt.keys())] * world_size)
        in1 = in1.to_dict()
        in2 = in2.to_dict()
        m0 = HashZchManagedCollisionModule(
            zch_size=20,
            device=torch.device("cuda"),
            input_hash_size=hash_size,
            total_num_buckets=2,
            eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
            eviction_config=HashZchEvictionConfig(
                features=[],
                single_ttl=10,
            ),
        )
        m1 = m0.rebuild_with_output_id_range((0, 10))
        m2 = m0.rebuild_with_output_id_range((10, 20))

        # simulate calls to each rank
        o1 = m1(in1)
        o2 = m2(in2)

        m0.reset_inference_mode()
        full_zch_identities = torch.cat(
            [
                m1.state_dict()["_hash_zch_identities"],
                m2.state_dict()["_hash_zch_identities"],
            ]
        )
        state_dict = m0.state_dict()
        state_dict["_hash_zch_identities"] = full_zch_identities
        m0.load_state_dict(state_dict)

        # now pass in original kjt
        inf_input = kjt.to_dict()
        inf_output = m0(inf_input)

        torch.allclose(
            inf_output["f"].values(), torch.cat([o1["f"].values(), o2["f"].values()])
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-ignore [56]
    @given(hash_size=st.sampled_from([0, 80]))
    @settings(max_examples=5, deadline=None)
    def test_zch_hash_train_rescales_two_disabled_in_oss_compatibility(
        self, hash_size: int
    ) -> None:
        keep_original_indices = False
        # rank 0
        world_size = 2
        kjt = KeyedJaggedTensor(
            keys=["f"],
            values=torch.cat(
                [
                    torch.randint(
                        0,
                        hash_size if hash_size > 0 else 1000,
                        (20,),
                        dtype=torch.int64,
                        device="cuda",
                    ),
                ]
            ),
            lengths=torch.cat(
                [
                    torch.tensor([4, 6], dtype=torch.int64, device="cuda"),
                    torch.tensor([4, 6], dtype=torch.int64, device="cuda"),
                ]
            ),
        )
        block_sizes = torch.tensor(
            [(size + world_size - 1) // world_size for size in [hash_size]],
            dtype=torch.int64,
            device="cuda",
        )
        sub_block_sizes = torch.tensor(
            [(size + 2 - 1) // 2 for size in [block_sizes[0]]],
            dtype=torch.int64,
            device="cuda",
        )
        bucketized_kjt, _ = bucketize_kjt_before_all2all(
            kjt,
            num_buckets=world_size,
            block_sizes=block_sizes,
            keep_original_indices=keep_original_indices,
        )
        in1, in2 = bucketized_kjt.split([len(kjt.keys())] * world_size)

        bucketized_in1, _ = bucketize_kjt_before_all2all(
            in1,
            num_buckets=2,
            block_sizes=sub_block_sizes,
            keep_original_indices=keep_original_indices,
        )
        bucketized_in2, _ = bucketize_kjt_before_all2all(
            in2,
            num_buckets=2,
            block_sizes=sub_block_sizes,
            keep_original_indices=keep_original_indices,
        )
        in1_1, in1_2 = bucketized_in1.split([len(kjt.keys())] * 2)
        in2_1, in2_2 = bucketized_in2.split([len(kjt.keys())] * 2)

        in1_1, in1_2 = in1_1.to_dict(), in1_2.to_dict()
        in2_1, in2_2 = in2_1.to_dict(), in2_2.to_dict()

        m0 = HashZchManagedCollisionModule(
            zch_size=20,
            device=torch.device("cuda"),
            input_hash_size=hash_size,
            total_num_buckets=4,
            eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
            eviction_config=HashZchEvictionConfig(
                features=[],
                single_ttl=10,
            ),
        )

        m1_1 = m0.rebuild_with_output_id_range((0, 5))
        m1_2 = m0.rebuild_with_output_id_range((5, 10))
        m2_1 = m0.rebuild_with_output_id_range((10, 15))
        m2_2 = m0.rebuild_with_output_id_range((15, 20))

        # simulate calls to each rank
        o1_1 = m1_1(in1_1)
        o1_2 = m1_2(in1_2)
        o2_1 = m2_1(in2_1)
        o2_2 = m2_2(in2_2)

        m0.reset_inference_mode()

        full_zch_identities = torch.cat(
            [
                m1_1.state_dict()["_hash_zch_identities"],
                m1_2.state_dict()["_hash_zch_identities"],
                m2_1.state_dict()["_hash_zch_identities"],
                m2_2.state_dict()["_hash_zch_identities"],
            ]
        )
        state_dict = m0.state_dict()
        state_dict["_hash_zch_identities"] = full_zch_identities
        m0.load_state_dict(state_dict)

        # now pass in original kjt
        inf_input = kjt.to_dict()
        inf_output = m0(inf_input)
        torch.allclose(
            inf_output["f"].values(),
            torch.cat([x["f"].values() for x in [o1_1, o1_2, o2_1, o2_2]]),
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    # pyre-ignore [56]
    @given(hash_size=st.sampled_from([0, 80]))
    @settings(max_examples=5, deadline=None)
    def test_zch_hash_train_rescales_one(self, hash_size: int) -> None:
        keep_original_indices = True
        kjt = KeyedJaggedTensor(
            keys=["f"],
            values=torch.cat(
                [
                    torch.randint(
                        0,
                        hash_size if hash_size > 0 else 1000,
                        (20,),
                        dtype=torch.int64,
                        device="cuda",
                    ),
                ]
            ),
            lengths=torch.cat(
                [
                    torch.tensor([4, 6], dtype=torch.int64, device="cuda"),
                    torch.tensor([4, 6], dtype=torch.int64, device="cuda"),
                ]
            ),
        )

        # initialize mch with 8 buckets
        m0 = HashZchManagedCollisionModule(
            zch_size=40,
            device=torch.device("cuda"),
            input_hash_size=hash_size,
            total_num_buckets=4,
            eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
            eviction_config=HashZchEvictionConfig(
                features=[],
                single_ttl=10,
            ),
        )

        # start with world_size = 2
        world_size = 2
        block_sizes = torch.tensor(
            [(size + world_size - 1) // world_size for size in [hash_size]],
            dtype=torch.int64,
            device="cuda",
        )

        m1_1 = m0.rebuild_with_output_id_range((0, 20))
        m2_1 = m0.rebuild_with_output_id_range((20, 40))

        # shard, now world size 1!
        if hash_size > 0:
            world_size = 1
            block_sizes = torch.tensor(
                [(size + world_size - 1) // world_size for size in [hash_size]],
                dtype=torch.int64,
                device="cuda",
            )
            # simulate kjt call
            bucketized_kjt, permute = bucketize_kjt_before_all2all(
                kjt,
                num_buckets=world_size,
                block_sizes=block_sizes,
                keep_original_indices=keep_original_indices,
                output_permute=True,
            )
            in1_2 = bucketized_kjt.split([len(kjt.keys())] * world_size)[0]
        else:
            bucketized_kjt, permute = bucketize_kjt_before_all2all(
                kjt,
                num_buckets=world_size,
                block_sizes=block_sizes,
                keep_original_indices=keep_original_indices,
                output_permute=True,
            )
            kjts = bucketized_kjt.split([len(kjt.keys())] * world_size)
            # rebuild kjt
            in1_2 = KeyedJaggedTensor(
                keys=kjts[0].keys(),
                values=torch.cat([kjts[0].values(), kjts[1].values()], dim=0),
                lengths=torch.cat([kjts[0].lengths(), kjts[1].lengths()], dim=0),
            )

        m1_2 = m0.rebuild_with_output_id_range((0, 40))
        m1_zch_identities = torch.cat(
            [
                m1_1.state_dict()["_hash_zch_identities"],
                m2_1.state_dict()["_hash_zch_identities"],
            ]
        )
        m1_zch_metadata = torch.cat(
            [
                m1_1.state_dict()["_hash_zch_metadata"],
                m2_1.state_dict()["_hash_zch_metadata"],
            ]
        )
        state_dict = m1_2.state_dict()
        state_dict["_hash_zch_identities"] = m1_zch_identities
        state_dict["_hash_zch_metadata"] = m1_zch_metadata
        m1_2.load_state_dict(state_dict)
        _ = m1_2(in1_2.to_dict())

        m0.reset_inference_mode()  # just clears out training state
        full_zch_identities = torch.cat(
            [
                m1_2.state_dict()["_hash_zch_identities"],
            ]
        )
        state_dict = m0.state_dict()
        state_dict["_hash_zch_identities"] = full_zch_identities
        m0.load_state_dict(state_dict)

        m1_2.eval()
        assert m0.training is False

        inf_input = kjt.to_dict()

        inf_output = m0(inf_input)
        o1_2 = m1_2(in1_2.to_dict())
        self.assertTrue(
            torch.allclose(
                inf_output["f"].values(),
                torch.index_select(
                    o1_2["f"].values(),
                    dim=0,
                    index=cast(torch.Tensor, permute),
                ),
            )
        )

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "This test requires at least one GPU",
    )
    def test_output_global_offset_tensor(self) -> None:
        m = HashZchManagedCollisionModule(
            zch_size=20,
            device=torch.device("cpu"),
            total_num_buckets=4,
        )
        self.assertIsNone(m._output_global_offset_tensor)

        bucket2 = m.rebuild_with_output_id_range((5, 10))
        self.assertIsNotNone(bucket2._output_global_offset_tensor)
        self.assertTrue(
            # pyre-ignore [6]
            torch.equal(bucket2._output_global_offset_tensor, torch.tensor([5]))
        )
        self.assertEqual(bucket2._start_bucket, 1)

        m.reset_inference_mode()
        bucket3 = m.rebuild_with_output_id_range((10, 15))
        self.assertIsNotNone(bucket3._output_global_offset_tensor)
        self.assertTrue(
            # pyre-ignore [6]
            torch.equal(bucket3._output_global_offset_tensor, torch.tensor([10]))
        )
        self.assertEqual(bucket3._start_bucket, 2)
        self.assertEqual(
            # pyre-ignore [16]
            bucket3._output_global_offset_tensor.device.type,
            "cpu",
        )

        remapped_indices = bucket3.remap(
            {
                "test": JaggedTensor(
                    values=torch.tensor(
                        [6, 10, 14, 18, 22], dtype=torch.int64, device="cpu"
                    ),
                    lengths=torch.tensor([5], dtype=torch.int64, device="cpu"),
                )
            }
        )
        self.assertTrue(
            torch.allclose(
                remapped_indices["test"].values(), torch.tensor([14, 10, 10, 11, 10])
            )
        )

        gpu_zch = HashZchManagedCollisionModule(
            zch_size=20,
            device=torch.device("cuda"),
            total_num_buckets=4,
        )
        bucket4 = gpu_zch.rebuild_with_output_id_range((15, 20))
        self.assertIsNotNone(bucket4._output_global_offset_tensor)
        self.assertTrue(bucket4._output_global_offset_tensor.device.type == "cuda")
        self.assertEqual(
            bucket4._output_global_offset_tensor, torch.tensor([15], device="cuda")
        )

        meta_zch = HashZchManagedCollisionModule(
            zch_size=20,
            device=torch.device("meta"),
            total_num_buckets=4,
        )
        meta_zch.reset_inference_mode()
        self.assertIsNone(meta_zch._output_global_offset_tensor)
        bucket5 = meta_zch.rebuild_with_output_id_range((15, 20))
        self.assertIsNotNone(bucket5._output_global_offset_tensor)
        self.assertTrue(bucket5._output_global_offset_tensor.device.type == "cpu")
        self.assertEqual(bucket5._output_global_offset_tensor, torch.tensor([15]))

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "This test requires at least one GPU",
    )
    def test_dynamically_switch_inference_training_mode(self) -> None:
        m = HashZchManagedCollisionModule(
            zch_size=4,
            device=torch.device("cuda"),
            total_num_buckets=2,
            eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
            eviction_config=HashZchEvictionConfig(
                features=[],
                single_ttl=10,
            ),
            max_probe=4,
        )
        jt = JaggedTensor(
            values=torch.arange(0, 4, dtype=torch.int64, device="cuda"),
            lengths=torch.tensor([4], dtype=torch.int64, device="cuda"),
        )

        with patch("time.time") as mock_time:
            mock_time.return_value = 360000  # hour 100
            m.remap({"test": jt})

        self.assertTrue(m.training)
        self.assertFalse(m._is_inference)
        self.assertEqual(m._hash_zch_metadata.shape[0], 4)
        self.assertTrue(torch.all(m._hash_zch_metadata == 110))
        self.assertEqual(
            m._eviction_policy_name, HashZchEvictionPolicyName.SINGLE_TTL_EVICTION
        )

        m.reset_intrainer_bulk_eval_mode()
        self.assertFalse(m.training)
        self.assertTrue(m._is_inference)
        self.assertTrue(m._eviction_policy_name is None)
        self.assertTrue(m._eviction_module is None)

        with patch("time.time") as mock_time:
            mock_time.return_value = 540000  # hour 150
            m.remap({"test": jt})

        # check self._hash_zch_metadata is frozen
        self.assertTrue(torch.all(m._hash_zch_metadata == 110))

        m.reset_training_mode()
        self.assertTrue(m.training)
        self.assertFalse(m._is_inference)
        self.assertEqual(
            m._eviction_policy_name, HashZchEvictionPolicyName.SINGLE_TTL_EVICTION
        )
        self.assertTrue(m._eviction_module is not None)

        with patch("time.time") as mock_time:
            mock_time.return_value = 540000
            m.remap({"test": jt})
            # check self._hash_zch_metadata is updated
            self.assertTrue(torch.all(m._hash_zch_metadata == 160))

        m.reset_inference_mode()
        self.assertFalse(m.training)
        self.assertTrue(m._is_inference)
        self.assertTrue(m._eviction_policy_name is None)
        self.assertTrue(m._eviction_module is None)
