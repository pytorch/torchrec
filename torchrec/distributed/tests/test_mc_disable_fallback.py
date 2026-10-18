#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import cast, List, Optional

import torch
import torch.nn as nn
from pyre_extensions import none_throws
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.mc_embedding import (
    ManagedCollisionEmbeddingCollectionSharder,
    ShardedManagedCollisionEmbeddingCollection,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import construct_module_sharding_plan, row_wise
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.train_pipeline.utils import _override_input_dist_forwards
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
from torchrec.modules.mc_modules import ManagedCollisionCollection
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class

WORLD_SIZE = 2
_TABLE = "table_0"
_FEATURE = "media"
_NUM_IDS = 8


class _DisableFallbackSparseArch(nn.Module):
    def __init__(
        self, table: EmbeddingConfig, device: torch.device, no_bag: bool = True
    ) -> None:
        super().__init__()
        self._mc_ec: ManagedCollisionEmbeddingCollection = (
            ManagedCollisionEmbeddingCollection(
                EmbeddingCollection(tables=[table], device=device),
                ManagedCollisionCollection(
                    managed_collision_modules={
                        table.name: HashZchManagedCollisionModule(
                            zch_size=table.num_embeddings,
                            device=device,
                            total_num_buckets=WORLD_SIZE,
                            input_hash_size=0,
                            name=table.name,
                            # logs every lookup, misses included
                            tb_logging_frequency=1,
                            disable_fallback=True,
                            no_bag=no_bag,
                        )
                    },
                    embedding_configs=[table],
                ),
            )
        )

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        ec_out, _ = self._mc_ec(kjt)
        return ec_out[_FEATURE].values()


def _table() -> EmbeddingConfig:
    return EmbeddingConfig(
        name=_TABLE,
        num_embeddings=128,
        embedding_dim=8,
        feature_names=[_FEATURE],
    )


def _kjt(ids: List[int], device: torch.device) -> KeyedJaggedTensor:
    return KeyedJaggedTensor.from_lengths_sync(
        keys=[_FEATURE],
        values=torch.tensor(ids, dtype=torch.int64),
        lengths=torch.ones(len(ids), dtype=torch.int64),
    ).to(device)


def _shard(
    sparse_arch: _DisableFallbackSparseArch,
    sharder: ModuleSharder[nn.Module],
    ctx: MultiProcessContext,
    world_size: int,
) -> nn.Module:
    module_sharding_plan = construct_module_sharding_plan(
        sparse_arch._mc_ec,
        per_param_sharding={_TABLE: row_wise(num_buckets=world_size)},
        local_size=ctx.local_size,
        world_size=world_size,
        device_type="cuda",
        sharder=sharder,
    )
    return _shard_modules(
        module=copy.deepcopy(sparse_arch),
        plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
        # pyrefly: ignore[bad-argument-type]
        env=ShardingEnv.from_process_group(ctx.pg),
        sharders=[sharder],
        device=ctx.device,
    )


def _test_disable_fallback_lookup(
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    """
    Inserts ids in train mode, then looks them up in eval mode next to ids that
    were never inserted, which miss now that the table has no fallback.
    """
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sparse_arch = _DisableFallbackSparseArch(_table(), torch.device("meta"))
        apply_optimizer_in_backward(
            RowWiseAdagrad,
            # pyrefly: ignore[bad-argument-type]
            [sparse_arch._mc_ec._embedding_collection.embeddings[_TABLE].weight],
            {"lr": 0.01},
        )
        sharded_sparse_arch = _shard(sparse_arch, sharder, ctx, world_size)
        mc_ec = cast(
            ShardedManagedCollisionEmbeddingCollection, sharded_sparse_arch._mc_ec
        )
        tc = unittest.TestCase()

        # The same draw on every rank: the ids each rank inserts, then the misses.
        ids = (
            torch.randperm(100_000, generator=torch.Generator().manual_seed(0))[
                : 2 * world_size * _NUM_IDS
            ]
            .view(2 * world_size, _NUM_IDS)
            .tolist()
        )
        inserted = ids[:world_size]
        inserted_embs = sharded_sparse_arch(_kjt(inserted[rank], ctx.device))

        sharded_sparse_arch.eval()
        missed = ids[world_size + rank]
        others = [i for r in range(world_size) if r != rank for i in inserted[r]]
        # Misses first, so rows shifted by a dropped miss would not line up. The
        # repeated ids collapse under index dedup.
        query = _kjt(
            missed + inserted[rank] + others + [missed[0], inserted[rank][0]],
            ctx.device,
        )

        embs = sharded_sparse_arch(query)
        tc.assertEqual(embs.size(0), query.values().numel())
        torch.testing.assert_close(embs[_NUM_IDS : 2 * _NUM_IDS], inserted_embs)
        torch.testing.assert_close(embs[-1], inserted_embs[0])
        expected_mask = (
            [False] * len(missed)
            + [True] * (len(inserted[rank]) + len(others))
            + [False, True]
        )
        tc.assertEqual(
            mc_ec.lookup_remapped_lengths_mask(query).tolist(), expected_mask
        )
        # A train pipeline overrides the input dists to return the splits for it
        # to fuse.
        _override_input_dist_forwards([mc_ec])
        tc.assertEqual(
            mc_ec.lookup_remapped_lengths_mask(query).tolist(), expected_mask
        )


def _test_disable_fallback_needs_no_bag(
    rank: int,
    world_size: int,
    sharder: ModuleSharder[nn.Module],
    backend: str,
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        sparse_arch = _DisableFallbackSparseArch(
            _table(), torch.device("meta"), no_bag=False
        )
        with unittest.TestCase().assertRaisesRegex(AssertionError, "no_bag=True"):
            _shard(sparse_arch, sharder, ctx, world_size)


@skip_if_asan_class
class McDisableFallbackTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() < WORLD_SIZE,
        f"needs {WORLD_SIZE} GPUs",
    )
    def test_disable_fallback_lookup(self) -> None:
        for use_index_dedup in (False, True):
            self._run_multi_process_test(
                callable=_test_disable_fallback_lookup,
                world_size=WORLD_SIZE,
                sharder=ManagedCollisionEmbeddingCollectionSharder(
                    ec_sharder=EmbeddingCollectionSharder(
                        use_index_dedup=use_index_dedup
                    ),
                ),
                backend="nccl",
            )

    @unittest.skipIf(
        torch.cuda.device_count() < WORLD_SIZE,
        f"needs {WORLD_SIZE} GPUs",
    )
    def test_disable_fallback_needs_no_bag(self) -> None:
        self._run_multi_process_test(
            callable=_test_disable_fallback_needs_no_bag,
            world_size=WORLD_SIZE,
            sharder=ManagedCollisionEmbeddingCollectionSharder(),
            backend="nccl",
        )


class DisableFallbackRuntimeMetaTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 1, "needs a GPU")
    def test_remap_with_write_weights(self) -> None:
        m = HashZchManagedCollisionModule(
            zch_size=10,
            device=torch.device("cuda"),
            total_num_buckets=2,
            write_runtime_meta_dim=2,
            max_probe=4,
            disable_fallback=True,
            no_bag=True,
        )
        inserted = torch.tensor([31, 7, 18], dtype=torch.int64, device="cuda")
        m.remap(
            {"test": JaggedTensor(values=inserted, lengths=torch.ones_like(inserted))},
            write_weights=torch.tensor(
                [[11, 12], [13, 14], [15, 16]], dtype=torch.int64, device="cuda"
            ),
        )
        m.reset_inference_mode()
        runtime_meta = none_throws(m._hash_zch_runtime_meta).data.clone()
        # 42 and 95 miss, so the kernel returns -1 for them.
        values = torch.tensor([42, 7, 95], dtype=torch.int64, device="cuda")
        output = m.remap(
            {"test": JaggedTensor(values=values, lengths=torch.ones_like(values))},
            write_weights=torch.tensor(
                [[21, 22], [23, 24], [25, 26]], dtype=torch.int64, device="cuda"
            ),
        )
        runtime_meta[output["test"].values()[1]] = torch.tensor(
            [23, 24], dtype=torch.int64, device="cuda"
        )
        torch.testing.assert_close(
            none_throws(m._hash_zch_runtime_meta).data, runtime_meta, rtol=0, atol=0
        )
