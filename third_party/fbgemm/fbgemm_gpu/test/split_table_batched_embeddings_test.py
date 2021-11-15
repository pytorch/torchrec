#!/usr/bin/env python3

# pyre-ignore-all-errors[56]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import Callable, List, Optional, Tuple, TypeVar

import fbgemm_gpu.split_table_batched_embeddings_ops as split_table_batched_embeddings_ops
import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    OptimType,
    SparseType,
    RecordCacheMetrics,
    BoundsCheckMode,
)
from fbgemm_gpu.test.test_utils import gpu_available, gpu_unavailable

from hypothesis import HealthCheck, Verbosity, assume, given, settings
from torch import Tensor


MAX_EXAMPLES = 40

# For long running tests reduce the number of iterations to reduce timeout errors.
MAX_EXAMPLES_LONG_RUNNING = 20

Deviceable = TypeVar("Deviceable", torch.nn.EmbeddingBag, Tensor)


def round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def get_offsets_from_dense(indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    (B, L) = indices.size()
    return (
        indices.contiguous().view(-1),
        torch.tensor(
            np.cumsum(np.asarray([0] + [L for _ in range(B)])[:-1]).astype(np.int64)
        ),
    )


def to_device(t: Deviceable, use_cpu: bool) -> Deviceable:
    # pyre-fixme[7]: Expected `Deviceable` but got `Union[Tensor,
    #  torch.nn.EmbeddingBag]`.
    return t.cpu() if use_cpu else t.cuda()


def b_indices(
    b: Callable[..., torch.Tensor],
    x: torch.Tensor,
    per_sample_weights: Optional[torch.Tensor] = None,
    use_cpu: bool = False,
    do_pooling: bool = True,
) -> torch.Tensor:
    (indices, offsets) = get_offsets_from_dense(x)
    if not do_pooling:
        offsets = torch.arange(
            0, indices.numel(), device=indices.device, dtype=offsets.dtype
        )
    return b(
        to_device(indices, use_cpu),
        to_device(offsets, use_cpu),
        per_sample_weights=per_sample_weights,
    )


def get_table_batched_offsets_from_dense(
    merged_indices: torch.Tensor, use_cpu: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        to_device(merged_indices.contiguous().view(-1), use_cpu),
        to_device(
            torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).long(),
            use_cpu,
        ),
    )


def generate_requests(
    iters: int,
    B: int,
    T: int,
    L: int,
    E: int,
    # inter-batch indices reuse rate
    reuse: float = 0.0,
    # alpha <= 1.0: use uniform distribution
    # alpha > 1.0: use zjpf distribution
    alpha: float = 1.0,
    weights_precision: SparseType = SparseType.FP32,
    weighted: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    if alpha <= 1.0:
        all_indices = torch.randint(
            low=0,
            high=E,
            size=(iters, T, B * L),
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
    else:
        all_indices = (
            torch.as_tensor(np.random.zipf(a=alpha, size=(iters, T, B * L)))
            .to(torch.cuda.current_device())
            .int()
            % E
        )
    for it in range(iters - 1):
        for t in range(T):
            reused_indices = torch.randperm(B * L, device=torch.cuda.current_device())[
                : int(B * L * reuse)
            ]
            all_indices[it + 1, t, reused_indices] = all_indices[it, t, reused_indices]

    rs = []
    for it in range(iters):
        weight_tensor = (
            None
            if not weighted
            else torch.randn(
                T * B * L,
                device=torch.cuda.current_device(),
                dtype=torch.float16 if weights_precision else torch.float32,
            )
        )
        rs.append(
            get_table_batched_offsets_from_dense(all_indices[it].view(T, B, L))
            + (weight_tensor,)
        )
    return rs


class SplitTableBatchedEmbeddingsTest(unittest.TestCase):
    def execute_forward_(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: split_table_batched_embeddings_ops.CacheAlgorithm,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
    ) -> None:
        # NOTE: cache is not applicable to CPU version.
        assume(not use_cpu or not use_cache)
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        assume(not use_cpu or T * B * L * D <= 2048)
        assume(not (use_cpu and weights_precision == SparseType.FP16))

        assume(
            pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            or not weighted
        )
        if pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
            emb_op = (
                split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen
            )
        elif pooling_mode == split_table_batched_embeddings_ops.PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
            emb_op = (
                split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen
            )
        elif pooling_mode == split_table_batched_embeddings_ops.PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
            T = 1  # PoolingMode.None only works for T = 1
            emb_op = split_table_batched_embeddings_ops.SequenceEmbeddingCodegen
        else:
            # This proves that we have exhaustively checked all PoolingModes
            raise RuntimeError("Unknown PoolingMode!")

        E = int(10 ** log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [int(1e4)] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        compute_device = split_table_batched_embeddings_ops.ComputeDevice.CUDA
        if use_cpu:
            managed = [split_table_batched_embeddings_ops.EmbeddingLocation.HOST] * T
            compute_device = split_table_batched_embeddings_ops.ComputeDevice.CPU
        elif use_cache:
            managed = [
                split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING
            ] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
                        if d < average_D
                        else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        bs = [
            to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
            for (E, D) in zip(Es, Ds)
        ]
        if weights_precision == SparseType.INT8:
            for t in range(T):
                bs[t].weight.data.copy_(
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                            bs[t].weight.data
                        )
                    )
                )

        if weights_precision == SparseType.FP16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            bs = [b.half() for b in bs]

        xs = [to_device(torch.randint(low=0, high=e, size=(B, L)), use_cpu) for e in Es]
        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(T)]
        xws_acc_type = copy.deepcopy(xws)

        if weights_precision == SparseType.FP16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            xws = [xw.half() for xw in xws]

        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, xs)
            ]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        if do_pooling:
            f = torch.cat([f.view(B, -1) for f in fs], dim=1)
        else:
            f = torch.cat([f.view(-1) for f in fs], dim=0)

        cc = emb_op(
            embedding_specs=[
                (
                    E,
                    D,
                    split_table_batched_embeddings_ops.EmbeddingLocation(M),
                    compute_device,
                )
                for (E, D, M) in zip(Es, Ds, managed)
            ],
            weights_precision=weights_precision,
            optimizer=OptimType.EXACT_SGD,
            learning_rate=0.05,
            cache_algorithm=cache_algorithm,
            pooling_mode=pooling_mode,
        )
        if do_pooling:
            # NOTE: test TorchScript-compatible!
            cc = torch.jit.script(cc)

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(
                bs[t].weight
                if weights_precision != SparseType.INT8
                else torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(bs[t].weight)
            )

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu)
        if not do_pooling:
            offsets = None
        fc2 = (
            cc(indices, offsets)
            if not weighted
            else cc(indices, offsets, to_device(xw.contiguous().view(-1), use_cpu))
        )
        if not do_pooling:
            fc2 = fc2.view(-1)
        torch.testing.assert_allclose(
            fc2.float(),
            f.float(),
            atol=8.0e-3 if weights_precision == SparseType.FP16 else 1.0e-5,
            rtol=8.0e-3 if weights_precision == SparseType.FP16 else 1.0e-5,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_precision=st.just(SparseType.INT8),
        weighted=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(
            split_table_batched_embeddings_ops.CacheAlgorithm
        ),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
                split_table_batched_embeddings_ops.PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_int8(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: split_table_batched_embeddings_ops.CacheAlgorithm,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
    ) -> None:
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_precision=st.just(SparseType.FP16),
        weighted=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(
            split_table_batched_embeddings_ops.CacheAlgorithm
        ),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
                split_table_batched_embeddings_ops.PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_fp16(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: split_table_batched_embeddings_ops.CacheAlgorithm,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
    ) -> None:
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_precision=st.just(SparseType.FP32),
        weighted=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(
            split_table_batched_embeddings_ops.CacheAlgorithm
        ),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
                split_table_batched_embeddings_ops.PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_forward_fp32(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: split_table_batched_embeddings_ops.CacheAlgorithm,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
    ) -> None:
        self.execute_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weights_precision,
            weighted,
            mixed,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        # FIXME: switch to
        # pooled_embedding_precision=st.sampled_from([SparseType.FP16, SparseType.INT8]),
        # after v0/v2 is landed.
        pooled_embedding_precision=st.sampled_from([SparseType.FP32]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_forward_fused_pooled_emb_quant(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        pooled_embedding_precision: SparseType,
    ) -> None:
        Ds = [
            round_up(np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)), 4)
            for _ in range(T)
        ]
        E = int(10 ** log_E)
        Es = [np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)]

        op = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    E,
                    D,
                    split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                    split_table_batched_embeddings_ops.ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            pooled_output_precision=pooled_embedding_precision,
            device=torch.cuda.current_device(),
        )
        op_ref = (
            split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        E,
                        D,
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.ComputeDevice.CUDA,
                    )
                    for (E, D) in zip(Es, Ds)
                ],
                pooled_output_precision=SparseType.FP32,
                device=torch.cuda.current_device(),
            )
        )
        # sync weights between two ops
        split_weights = op.split_embedding_weights()
        ref_split_weights = op_ref.split_embedding_weights()
        for t in range(T):
            split_weights[t].data.copy_(ref_split_weights[t])

        requests = generate_requests(2, B, T, L, min(Es), reuse=0.1)

        for indices, offsets, _ in requests:
            lowp_pooled_output = op(
                indices=indices,
                offsets=offsets,
            )
            fp32_pooled_output = op_ref(
                indices=indices,
                offsets=offsets,
            )

            lowp_pooled_emb_split = [
                d + 8 if pooled_embedding_precision == SparseType.INT8 else d
                for d in op.dims
            ]
            lowp_pooled_output_per_table = torch.split(
                lowp_pooled_output, lowp_pooled_emb_split, dim=1
            )

            deq_lowp_pooled_output_per_table = [
                torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(t.contiguous())
                if pooled_embedding_precision == SparseType.INT8
                else t.float()
                for t in lowp_pooled_output_per_table
            ]
            fp32_pooled_output_per_table = torch.split(
                fp32_pooled_output, op.dims, dim=1
            )

            dq_fp32_pooled_output_per_table = [
                torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                    torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                        t.contiguous()
                    ).contiguous()
                )
                if pooled_embedding_precision == SparseType.INT8
                else t.half().float()
                for t in fp32_pooled_output_per_table
            ]

            cat_deq_lowp_pooled_output = torch.cat(
                deq_lowp_pooled_output_per_table, dim=1
            )

            cat_dq_fp32_pooled_output = torch.cat(
                dq_fp32_pooled_output_per_table, dim=1
            )
            assert torch.allclose(cat_deq_lowp_pooled_output, cat_dq_fp32_pooled_output)

    @given(
        T=st.integers(min_value=1, max_value=3),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=32),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=10),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
                split_table_batched_embeddings_ops.PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_dense(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        long_segments: bool,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
    ) -> None:
        # NOTE: torch.autograd.gradcheck() is too time-consuming for CPU version
        #       so we have to limit (T * B * L * D)!
        assume(not use_cpu or T * B * L * D <= 2048)
        assume(
            pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            or not weighted
        )
        assume(not (use_cpu and weights_precision == SparseType.FP16))

        if pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
            emb_op = (
                split_table_batched_embeddings_ops.DenseTableBatchedEmbeddingBagsCodegen
            )
        elif pooling_mode == split_table_batched_embeddings_ops.PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
            emb_op = (
                split_table_batched_embeddings_ops.DenseTableBatchedEmbeddingBagsCodegen
            )
        elif pooling_mode == split_table_batched_embeddings_ops.PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
            T = 1  # PoolingMode.None only works for T = 1
            # emb_op = split_table_batched_embeddings_ops.DenseTableBatchedEmbeddingBagsCodegen
            emb_op = split_table_batched_embeddings_ops.DenseSequenceEmbeddingCodegen
        else:
            raise RuntimeError("Unknown PoolingMode!")

        E = int(10 ** log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2 * E)) for _ in range(T)
            ]
        bs = [
            to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=False), use_cpu)
            for (E, D) in zip(Es, Ds)
        ]

        if weights_precision == SparseType.FP16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            bs = [b.half() for b in bs]

        xs = [
            to_device(
                torch.from_numpy(
                    np.random.choice(range(e), size=(B, L), replace=True).astype(
                        np.int64
                    )
                ),
                use_cpu,
            )
            for e in Es
        ]
        if long_segments and L > 0 and weights_precision != SparseType.FP16:
            for x in xs:
                x[:, 0] = 0

        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(T)]
        xws_acc_type = copy.deepcopy(xws)

        if weights_precision == SparseType.FP16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            xws = [xw.half() for xw in xws]

        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, xs)
            ]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        gos = [torch.randn_like(f) for f in fs]
        [f.backward(go) for (f, go) in zip(fs, gos)]

        grad_weights = torch.cat([b.weight.grad.view(-1) for b in bs])
        if weights_precision == SparseType.FP16 and not use_cpu:
            grad_weights = grad_weights.half()

        cc = emb_op(
            embedding_specs=[(E, D) for (E, D) in zip(Es, Ds)],
            pooling_mode=pooling_mode,
            use_cpu=use_cpu,
        )
        if weights_precision == SparseType.FP16 and not use_cpu:
            cc = cc.half()
        if do_pooling:
            # NOTE: test TorchScript-compatible!
            cc = torch.jit.script(cc)

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu)
        fc2 = (
            cc(indices, offsets)
            if not weighted
            else cc(indices, offsets, to_device(xw.contiguous().view(-1), use_cpu))
        )

        if do_pooling:
            f = torch.cat([f.view(B, -1) for f in fs], dim=1)
        else:
            f = torch.cat(fs, dim=1)

        torch.testing.assert_allclose(
            fc2.float(),
            f.float(),
            atol=5.0e-3 if weights_precision == SparseType.FP16 else 1.0e-5,
            rtol=5.0e-3 if weights_precision == SparseType.FP16 else 1.0e-5,
        )
        if do_pooling:
            goc = torch.cat([go.view(B, -1) for go in gos], dim=1).contiguous()
        else:
            goc = torch.cat(gos, dim=1).contiguous()
        fc2.backward(goc)
        torch.testing.assert_allclose(
            cc.weights.grad,
            grad_weights,
            atol=5.0e-3 if weights_precision == SparseType.FP16 else 1.0e-4,
            rtol=5.0e-3 if weights_precision == SparseType.FP16 else 1.0e-4,
        )

        # pyre-fixme[29]: `Union[Tensor, torch.nn.Module]` is not a function.
        cc = split_table_batched_embeddings_ops.DenseTableBatchedEmbeddingBagsCodegen(
            [(E, D) for (E, D) in zip(Es, Ds)],
            # NOTE: only SUM pooling can work with per_sample_weights!
            pooling_mode=split_table_batched_embeddings_ops.PoolingMode.SUM,
            use_cpu=use_cpu,
        ).double()
        per_sample_weights = to_device(xw.contiguous().view(-1), use_cpu).double()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        for param in cc.parameters():
            param.requires_grad = False
        torch.autograd.gradcheck(cc, (indices, offsets, per_sample_weights))

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weights_precision=st.sampled_from([SparseType.FP16, SparseType.FP32]),
        weighted=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(
            split_table_batched_embeddings_ops.CacheAlgorithm
        ),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
                split_table_batched_embeddings_ops.PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        exact=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_sgd(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weights_precision: SparseType,
        weighted: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: split_table_batched_embeddings_ops.CacheAlgorithm,
        long_segments: bool,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
        exact: bool,
    ) -> None:
        # NOTE: cache is not applicable to CPU version.
        assume(not use_cpu or not use_cache)
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        assume(not use_cpu or T * B * L * D <= 2048)
        assume(not (use_cpu and weights_precision == SparseType.FP16))
        # GPU only does exact sgd
        assume((use_cpu and not long_segments) or exact)

        assume(
            pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            or not weighted
        )

        if pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
            emb_op = (
                split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen
            )
        elif pooling_mode == split_table_batched_embeddings_ops.PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
            emb_op = (
                split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen
            )
        elif pooling_mode == split_table_batched_embeddings_ops.PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
            T = 1  # PoolingMode.None only works for T = 1
            emb_op = split_table_batched_embeddings_ops.SequenceEmbeddingCodegen
        else:
            raise RuntimeError("Unknown PoolingMode!")

        E = int(10 ** log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        compute_device = split_table_batched_embeddings_ops.ComputeDevice.CUDA
        if use_cpu:
            managed = [split_table_batched_embeddings_ops.EmbeddingLocation.HOST] * T
            compute_device = split_table_batched_embeddings_ops.ComputeDevice.CPU
        elif use_cache:
            managed = [
                split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING
            ] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
                        if d < average_D
                        else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        bs = [
            to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
            for (E, D) in zip(Es, Ds)
        ]

        if weights_precision == SparseType.FP16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            bs = [b.half() for b in bs]

        feature_table_map = list(range(T))
        if exact:
            table_to_replicate = T // 2
            bs.insert(table_to_replicate, bs[table_to_replicate])
            feature_table_map.insert(table_to_replicate, table_to_replicate)

        xs = [
            to_device(
                torch.from_numpy(
                    np.random.choice(range(Es[t]), size=(B, L), replace=exact).astype(
                        np.int64
                    )
                ),
                use_cpu,
            )
            for t in feature_table_map
        ]

        if long_segments and L > 0:
            for x in xs:
                x[:, 0] = 0

        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(len(xs))]
        xws_acc_type = copy.deepcopy(xws)

        if weights_precision == SparseType.FP16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            xws = [xw.half() for xw in xws]

        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, xs)
            ]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        gos = [torch.randn_like(f) for f in fs]
        [f.backward(go) for (f, go) in zip(fs, gos)]
        # do SGD update
        lr = 0.05
        if exact:
            # pyre-fixme[61]: `table_to_replicate` may not be initialized here.
            del bs[table_to_replicate]
        new_weights = [(b.weight - b.weight.grad * lr) for b in bs]

        cc = emb_op(
            embedding_specs=[
                (E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)
            ],
            optimizer=OptimType.EXACT_SGD if exact else OptimType.SGD,
            feature_table_map=feature_table_map,
            learning_rate=lr,
            weights_precision=weights_precision,
            cache_algorithm=cache_algorithm,
            pooling_mode=pooling_mode,
        )

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu)
        fc2 = (
            cc(indices, offsets)
            if not weighted
            else cc(indices, offsets, to_device(xw.contiguous().view(-1), use_cpu))
        )
        if do_pooling:
            goc = torch.cat([go.view(B, -1) for go in gos], dim=1).contiguous()
        else:
            goc = torch.cat(gos, dim=1).contiguous()
        fc2.backward(goc)
        if use_cache:
            cc.flush()
        for t in range(T):
            torch.testing.assert_allclose(
                cc.split_embedding_weights()[t],
                # pyre-fixme[16]: `float` has no attribute `half`.
                new_weights[t].half()
                if weights_precision == SparseType.FP16 and not use_cpu
                else new_weights[t],
                atol=(1.0e-2 if long_segments else 5.0e-3)
                if weights_precision == SparseType.FP16
                else 1.0e-5,
                rtol=2.0e-2 if weights_precision == SparseType.FP16 else 1.0e-5,
            )

    def execute_backward_adagrad_(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        D_gradcheck: int,
        weights_precision: SparseType,
        stochastic_rounding: bool,
        weighted: bool,
        row_wise: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: split_table_batched_embeddings_ops.CacheAlgorithm,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
        exact: bool,
    ) -> None:
        # NOTE: cache is not applicable to CPU version.
        assume(not use_cpu or not use_cache)
        # Approx AdaGrad only works with row_wise on CPU
        assume((use_cpu and row_wise) or exact)

        # NOTE: torch.autograd.gradcheck() is too time-consuming for CPU version
        #       so we have to limit (T * B * L * D)!
        assume(not use_cpu or T * B * L * D <= 1024)
        assume(not (use_cpu and weights_precision == SparseType.FP16))

        assume(
            pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            or not weighted
        )
        if pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
            emb_op = (
                split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen
            )
        elif pooling_mode == split_table_batched_embeddings_ops.PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
            emb_op = (
                split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen
            )
        elif pooling_mode == split_table_batched_embeddings_ops.PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
            T = 1  # PoolingMode.None only works for T = 1
            emb_op = split_table_batched_embeddings_ops.SequenceEmbeddingCodegen
        else:
            raise RuntimeError("Unknown PoolingMode!")

        # stochastic rounding only implemented for rowwise
        assume(not stochastic_rounding or row_wise)
        # need unique indices for non-exact tests
        assume(exact or int(10 ** log_E) > int(2.1 * B * L))
        # only row-wise supports caching
        assume(row_wise or not use_cache)

        E = int(10 ** log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        compute_device = split_table_batched_embeddings_ops.ComputeDevice.CUDA
        if use_cpu:
            managed = [split_table_batched_embeddings_ops.EmbeddingLocation.HOST] * T
            compute_device = split_table_batched_embeddings_ops.ComputeDevice.CPU
        elif use_cache:
            managed = [
                split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING
            ] * T
            if mixed:
                average_D = sum(Ds) // T
                for t, d in enumerate(Ds):
                    managed[t] = (
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
                        if d < average_D
                        else managed[t]
                    )
        else:
            managed = [
                np.random.choice(
                    [
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        bs = [
            to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
            for (E, D) in zip(Es, Ds)
        ]

        if weights_precision == SparseType.FP16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            bs = [b.half() for b in bs]

        feature_table_map = list(range(T))
        if exact:
            # autograd with shared embedding only works for exact
            table_to_replicate = T // 2
            bs.insert(table_to_replicate, bs[table_to_replicate])
            feature_table_map.insert(table_to_replicate, table_to_replicate)

        xs = [
            to_device(
                torch.from_numpy(
                    np.random.choice(range(Es[t]), size=(B, L), replace=exact).astype(
                        np.int64
                    )
                ),
                use_cpu,
            )
            for t in feature_table_map
        ]
        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(len(xs))]
        xws_acc_type = copy.deepcopy(xws)

        if weights_precision == SparseType.FP16 and not use_cpu:
            # NOTE: CPU version of torch.nn.EmbeddingBag doesn't support fp16.
            xws = [xw.half() for xw in xws]

        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, xs)
            ]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        gos = [torch.randn_like(f) for f in fs]
        [f.backward(go) for (f, go) in zip(fs, gos)]
        # do SGD update
        lr = 0.5
        eps = 0.2

        optimizer = (
            (OptimType.EXACT_ROWWISE_ADAGRAD if exact else OptimType.ROWWISE_ADAGRAD)
            if row_wise
            else OptimType.EXACT_ADAGRAD
        )
        cc = emb_op(
            embedding_specs=[
                (E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)
            ],
            feature_table_map=feature_table_map,
            optimizer=optimizer,
            learning_rate=lr,
            eps=eps,
            weights_precision=weights_precision,
            stochastic_rounding=stochastic_rounding,
            pooling_mode=pooling_mode,
        )

        if exact:
            # pyre-fixme[61]: `table_to_replicate` may not be initialized here.
            del bs[table_to_replicate]
        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu)
        fc2 = (
            cc(indices, offsets)
            if not weighted
            else cc(indices, offsets, to_device(xw.contiguous().view(-1), use_cpu))
        )
        if do_pooling:
            goc = torch.cat([go.view(B, -1) for go in gos], dim=1)
        else:
            goc = torch.cat(gos, dim=1).contiguous()
        fc2.backward(goc)
        cc.flush()
        split_optimizer_states = [s for (s,) in cc.split_optimizer_states()]
        for t in range(T):
            ref_optimizer_state = bs[t].weight.grad.float().cpu().to_dense().pow(2)
            torch.testing.assert_allclose(
                split_optimizer_states[t].float().cpu(),
                ref_optimizer_state.mean(dim=1) if row_wise else ref_optimizer_state,
                atol=1.0e-2 if weights_precision == SparseType.FP16 else 1.0e-4,
                rtol=1.0e-2 if weights_precision == SparseType.FP16 else 1.0e-4,
            )
        for t in range(T):
            # optimizer_state = squares (no row-wise) or sum squares (row-wise)
            torch.testing.assert_allclose(
                cc.split_embedding_weights()[t].float().cpu(),
                torch.addcdiv(
                    bs[t].weight.float().cpu(),
                    value=-lr,
                    tensor1=bs[t].weight.grad.float().cpu().to_dense(),
                    tensor2=split_optimizer_states[t]
                    .float()
                    .sqrt_()
                    .add_(eps)
                    .view(Es[t], 1 if row_wise else Ds[t])
                    .cpu(),
                ),
                atol=1.0e-2 if weights_precision == SparseType.FP16 else 1.0e-4,
                rtol=1.0e-2 if weights_precision == SparseType.FP16 else 1.0e-4,
            )
        if use_cpu:
            D_gradcheck = (D_gradcheck + 15) // 16 * 4
        else:
            D_gradcheck = D_gradcheck * 4
        cc = emb_op(
            embedding_specs=[
                (E, D_gradcheck, M, compute_device) for (E, M) in zip(Es, managed)
            ],
            feature_table_map=feature_table_map,
            optimizer=optimizer,
            learning_rate=0.0,
            eps=eps,
            weights_precision=weights_precision,
            stochastic_rounding=stochastic_rounding,
            # NOTE: only SUM pooling can work with per_sample_weights!
            pooling_mode=split_table_batched_embeddings_ops.PoolingMode.SUM,
        )
        if use_cpu:
            # NOTE: GPU version of SplitTableBatchedEmbeddingBagsCodegen doesn't support double.
            # pyre-fixme[29]: `Union[Tensor, torch.nn.Module]` is not a function.
            cc = cc.double()

        per_sample_weights = to_device(xw.contiguous().view(-1), use_cpu)
        if use_cpu:
            per_sample_weights = per_sample_weights.double()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        for param in cc.parameters():
            param.requires_grad = False
        torch.autograd.gradcheck(cc, (indices, offsets, per_sample_weights))

        per_sample_weights = to_device(xw.contiguous().view(-1), use_cpu)
        if use_cpu:
            per_sample_weights = per_sample_weights.double()
        per_sample_weights.requires_grad = True
        indices.requires_grad = False
        offsets.requires_grad = False
        for param in cc.parameters():
            param.requires_grad = False
        y = cc(indices, offsets, per_sample_weights)
        y.sum().backward()
        indice_weight_grad_all = per_sample_weights.grad.clone().cpu()
        T_ = len(xws)
        feature_requires_grad = to_device(
            torch.tensor(np.random.choice([0, 1], replace=True, size=(T_,))).int(),
            use_cpu,
        )
        per_sample_weights = per_sample_weights.detach().clone()
        per_sample_weights.requires_grad = True
        y = cc(
            indices,
            offsets,
            per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        )
        y.sum().backward()
        indice_weight_grad_mask = per_sample_weights.grad.clone().cpu()
        for t in range(T_):
            if feature_requires_grad[t]:
                torch.testing.assert_allclose(
                    indice_weight_grad_mask.view(T_, B, L)[t],
                    indice_weight_grad_all.view(T_, B, L)[t],
                )
            else:
                torch.testing.assert_allclose(
                    indice_weight_grad_mask.view(T_, B, L)[t],
                    torch.zeros_like(indice_weight_grad_mask.view(T_, B, L)[t]),
                )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        D_gradcheck=st.integers(min_value=1, max_value=2),
        weights_precision=st.just(SparseType.FP16),
        stochastic_rounding=st.booleans(),
        weighted=st.booleans(),
        row_wise=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(
            split_table_batched_embeddings_ops.CacheAlgorithm
        ),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
                split_table_batched_embeddings_ops.PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        exact=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_adagrad_fp16(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        D_gradcheck: int,
        weights_precision: SparseType,
        stochastic_rounding: bool,
        weighted: bool,
        row_wise: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: split_table_batched_embeddings_ops.CacheAlgorithm,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
        exact: bool,
    ) -> None:
        self.execute_backward_adagrad_(
            T,
            D,
            B,
            log_E,
            L,
            D_gradcheck,
            weights_precision,
            stochastic_rounding,
            weighted,
            row_wise,
            mixed,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            exact,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        D_gradcheck=st.integers(min_value=1, max_value=2),
        weights_precision=st.just(SparseType.FP32),
        stochastic_rounding=st.booleans(),
        weighted=st.booleans(),
        row_wise=st.booleans(),
        mixed=st.booleans(),
        use_cache=st.booleans(),
        cache_algorithm=st.sampled_from(
            split_table_batched_embeddings_ops.CacheAlgorithm
        ),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
                split_table_batched_embeddings_ops.PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        exact=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_adagrad_fp32(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        D_gradcheck: int,
        weights_precision: SparseType,
        stochastic_rounding: bool,
        weighted: bool,
        row_wise: bool,
        mixed: bool,
        use_cache: bool,
        cache_algorithm: split_table_batched_embeddings_ops.CacheAlgorithm,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
        exact: bool,
    ) -> None:
        self.execute_backward_adagrad_(
            T,
            D,
            B,
            log_E,
            L,
            D_gradcheck,
            weights_precision,
            stochastic_rounding,
            weighted,
            row_wise,
            mixed,
            use_cache,
            cache_algorithm,
            pooling_mode,
            use_cpu,
            exact,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=1, max_value=20),
        mixed=st.booleans(),
        cache_algorithm=st.sampled_from(
            split_table_batched_embeddings_ops.CacheAlgorithm
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_pipeline(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        mixed: bool,
        cache_algorithm: split_table_batched_embeddings_ops.CacheAlgorithm,
    ) -> None:
        iters = 3
        E = int(10 ** log_E)
        D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        managed = [
            split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING
        ] * T
        if mixed:
            average_D = sum(Ds) // T
            for t, d in enumerate(Ds):
                managed[t] = (
                    split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
                    if d < average_D
                    else managed[t]
                )
        cc_ref = (
            split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
                [
                    (
                        E,
                        D,
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.ComputeDevice.CUDA,
                    )
                    for (E, D) in zip(Es, Ds)
                ],
            )
        )
        cc = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
            [
                (E, D, M, split_table_batched_embeddings_ops.ComputeDevice.CUDA)
                for (E, D, M) in zip(Es, Ds, managed)
            ],
            cache_algorithm=cache_algorithm,
        )
        for t in range(T):
            assert (
                cc.split_embedding_weights()[t].size()
                == cc_ref.split_embedding_weights()[t].size()
            )
            cc.split_embedding_weights()[t].data.copy_(
                cc_ref.split_embedding_weights()[t]
            )

        requests = generate_requests(iters, B, T, L, min(Es), reuse=0.1)
        grad_output = torch.randn(B, sum(Ds)).cuda()

        for indices, offsets, _ in requests:
            output = cc(indices, offsets)
            output_ref = cc_ref(indices, offsets)
            torch.testing.assert_allclose(output, output_ref)
            output.backward(grad_output)
            output_ref.backward(grad_output)
        cc.flush()
        for t in range(T):
            torch.testing.assert_allclose(
                cc.split_embedding_weights()[t], cc_ref.split_embedding_weights()[t]
            )

    def execute_backward_optimizers_(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        optimizer: OptimType,
        long_segments: bool,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
    ) -> None:
        # NOTE: limit (T * B * L * D) to avoid timeout for CPU version!
        assume(not use_cpu or T * B * L * D <= 2048)
        assume(
            not use_cpu
            or optimizer
            in [
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
                OptimType.EXACT_SGD,
                OptimType.SGD,
            ]
        )

        assume(
            pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            or not weighted
        )
        if pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM:
            mode = "sum"
            do_pooling = True
            emb_op = (
                split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen
            )
        elif pooling_mode == split_table_batched_embeddings_ops.PoolingMode.MEAN:
            mode = "mean"
            do_pooling = True
            emb_op = (
                split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen
            )
        elif pooling_mode == split_table_batched_embeddings_ops.PoolingMode.NONE:
            mode = "sum"
            do_pooling = False
            T = 1  # PoolingMode.None only works for T = 1
            emb_op = split_table_batched_embeddings_ops.SequenceEmbeddingCodegen
        else:
            raise RuntimeError("Unknown PoolingMode!")

        E = int(10 ** log_E)
        if use_cpu:
            D = (D + 15) // 16 * 4
        else:
            D = D * 4
        if not mixed:
            Ds = [D] * T
            Es = [E] * T
        else:
            Ds = [
                round_up(np.random.randint(low=int(0.25 * D), high=int(1.0 * D)), 4)
                for _ in range(T)
            ]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]
        compute_device = split_table_batched_embeddings_ops.ComputeDevice.CUDA
        if use_cpu:
            managed = [split_table_batched_embeddings_ops.EmbeddingLocation.HOST] * T
            compute_device = split_table_batched_embeddings_ops.ComputeDevice.CPU
        else:
            managed = [
                np.random.choice(
                    [
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]
        bs = [
            to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
            for (E, D) in zip(Es, Ds)
        ]

        xs = [
            to_device(
                torch.from_numpy(
                    np.random.choice(range(e), size=(B, L), replace=True).astype(
                        np.int64
                    )
                ),
                use_cpu,
            )
            for e in Es
        ]
        if long_segments and L > 0:
            for x, e in zip(xs, Es):
                x[:, 0] = np.random.randint(low=0, high=e)

        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(T)]
        xws_acc_type = copy.deepcopy(xws)

        fs = (
            [
                b_indices(b, x, use_cpu=use_cpu, do_pooling=do_pooling)
                for (b, x) in zip(bs, xs)
            ]
            if not weighted
            else [
                b_indices(
                    b,
                    x,
                    per_sample_weights=xw.view(-1),
                    use_cpu=use_cpu,
                    do_pooling=do_pooling,
                )
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        gos = [torch.randn_like(f) for f in fs]
        [f.backward(go) for (f, go) in zip(fs, gos)]
        # do SGD update

        optimizer_kwargs = {"learning_rate": 0.5}
        (lr, eps, beta1, beta2, weight_decay, momentum, eta) = (
            0.5,
            1e-4,
            0.9,
            0.99,
            0.01,
            0.9,
            0.01,
        )
        if optimizer in (OptimType.EXACT_ROWWISE_ADAGRAD, OptimType.EXACT_ADAGRAD):
            optimizer_kwargs["eps"] = eps

        if optimizer in (OptimType.PARTIAL_ROWWISE_ADAM, OptimType.ADAM):
            optimizer_kwargs["eps"] = eps
            optimizer_kwargs["beta1"] = beta1
            optimizer_kwargs["beta2"] = beta2
            optimizer_kwargs["weight_decay"] = weight_decay

        if optimizer in (OptimType.PARTIAL_ROWWISE_LAMB, OptimType.LAMB):
            optimizer_kwargs["eps"] = eps
            optimizer_kwargs["beta1"] = beta1
            optimizer_kwargs["beta2"] = beta2
            optimizer_kwargs["weight_decay"] = weight_decay

        if optimizer == OptimType.LARS_SGD:
            optimizer_kwargs["weight_decay"] = weight_decay
            optimizer_kwargs["momentum"] = momentum
            optimizer_kwargs["eta"] = eta

        cc = emb_op(
            embedding_specs=[
                (E, D, M, compute_device) for (E, D, M) in zip(Es, Ds, managed)
            ],
            optimizer=optimizer,
            pooling_mode=pooling_mode,
            # pyre-fixme[6]: Expected `CacheAlgorithm` for 5th param but got `float`.
            **optimizer_kwargs,
        )

        for t in range(T):
            cc.split_embedding_weights()[t].data.copy_(bs[t].weight)

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu)
        fc2 = (
            cc(indices, offsets)
            if not weighted
            else cc(indices, offsets, to_device(xw.contiguous().view(-1), use_cpu))
        )
        if do_pooling:
            goc = torch.cat([go.view(B, -1) for go in gos], dim=1)
        else:
            goc = torch.cat(gos, dim=1).contiguous()
        fc2.backward(goc)
        cc.flush()

        split_optimizer_states = cc.split_optimizer_states()
        assert len(split_optimizer_states) == T
        split_weights = cc.split_embedding_weights()

        if optimizer in (OptimType.EXACT_ROWWISE_ADAGRAD, OptimType.EXACT_ADAGRAD):
            rowwise = optimizer == OptimType.EXACT_ROWWISE_ADAGRAD
            for t in range(T):
                (m1,) = split_optimizer_states[t]
                # to_dense in GPU is non-deterministic due to atmomics used in
                # coalescing and floating point non-associativity.
                dense_cpu_grad = bs[t].weight.grad.cpu().to_dense()
                m1_ref = (
                    dense_cpu_grad.pow(2)
                    if not rowwise
                    else dense_cpu_grad.pow(2).mean(dim=1)
                )
                torch.testing.assert_allclose(
                    m1.float().cpu(), m1_ref.float(), atol=1.0e-4, rtol=1.0e-4
                )
                weights_new = split_weights[t]
                weights_ref = bs[t].weight.cpu() - lr * dense_cpu_grad / (
                    torch.sqrt(
                        m1_ref if not rowwise else m1_ref.view(m1_ref.numel(), 1)
                    )
                    + eps
                )
                # TODO: why is tolerance off here?
                torch.testing.assert_allclose(
                    weights_new.float().cpu(),
                    weights_ref.float(),
                    atol=1.0e-2,
                    rtol=1.0e-2,
                )

        if optimizer in (OptimType.PARTIAL_ROWWISE_ADAM, OptimType.ADAM):
            rowwise = optimizer == OptimType.PARTIAL_ROWWISE_ADAM
            for t in range(T):
                (m1, m2) = split_optimizer_states[t]
                dense_cpu_grad = bs[t].weight.grad.cpu().to_dense()
                m2_ref = (
                    dense_cpu_grad.pow(2)
                    if not rowwise
                    else dense_cpu_grad.pow(2).mean(dim=1)
                ) * (1.0 - beta2)
                torch.testing.assert_allclose(
                    m2.cpu(), m2_ref, atol=1.0e-4, rtol=1.0e-4
                )
                m1_ref = dense_cpu_grad * (1.0 - beta1)
                torch.testing.assert_allclose(
                    m1.cpu(), m1_ref, atol=1.0e-4, rtol=1.0e-4
                )
                iter_ = cc.iter.item()
                v_hat_t = m2_ref / (1 - beta2 ** iter_)
                v_hat_t = v_hat_t if not rowwise else v_hat_t.view(v_hat_t.numel(), 1)
                m_hat_t = m1_ref / (1 - beta1 ** iter_)
                weights_new = split_weights[t]
                weights_ref = (
                    torch.addcdiv(
                        bs[t].weight.cpu(),
                        value=-lr,
                        tensor1=m_hat_t,
                        tensor2=v_hat_t.sqrt_().add_(eps),
                    )
                    - lr * weight_decay * bs[t].weight.cpu()
                )
                torch.testing.assert_allclose(
                    weights_new.index_select(dim=0, index=x[t].view(-1)).cpu(),
                    weights_ref.index_select(dim=0, index=x[t].view(-1).cpu()),
                    atol=1.0e-3,
                    rtol=1.0e-3,
                )

        if optimizer in (OptimType.PARTIAL_ROWWISE_LAMB, OptimType.LAMB):
            rowwise = optimizer == OptimType.PARTIAL_ROWWISE_LAMB
            for t in range(T):
                (m1, m2) = split_optimizer_states[t]
                dense_cpu_grad = bs[t].weight.grad.cpu().to_dense()
                m2_ref = (
                    dense_cpu_grad.pow(2)
                    if not rowwise
                    else dense_cpu_grad.pow(2).mean(dim=1)
                ) * (1.0 - beta2)
                torch.testing.assert_allclose(
                    m2.cpu(), m2_ref, atol=1.0e-4, rtol=1.0e-4
                )
                m1_ref = dense_cpu_grad * (1.0 - beta1)
                torch.testing.assert_allclose(
                    m1.cpu(), m1_ref, atol=1.0e-4, rtol=1.0e-4
                )
                iter_ = cc.iter.item()
                v_hat_t = m2_ref / (1 - beta2 ** iter_)
                v_hat_t = v_hat_t if not rowwise else v_hat_t.view(v_hat_t.numel(), 1)
                m_hat_t = m1_ref / (1 - beta1 ** iter_)
                rtw = (m_hat_t / (torch.sqrt(v_hat_t) + eps)) + weight_decay * bs[
                    t
                ].weight.cpu()
                true_ratio = torch.linalg.norm(bs[t].weight, dim=1, ord=2).view(
                    m1.shape[0], 1
                ).cpu() / torch.linalg.norm(rtw, dim=1, ord=2).view(m1.shape[0], 1)
                weights_new = split_weights[t]
                weights_ref = bs[t].weight.cpu() - lr * true_ratio * rtw
                torch.testing.assert_allclose(
                    weights_new.index_select(dim=0, index=x[t].view(-1)).cpu(),
                    weights_ref.index_select(dim=0, index=x[t].view(-1).cpu()),
                    atol=1.0e-3,
                    rtol=1.0e-3,
                )

        if optimizer == OptimType.LARS_SGD:
            for t in range(T):
                (m1,) = split_optimizer_states[t]
                weight_norm = (
                    torch.linalg.norm(bs[t].weight, dim=1, ord=2)
                    .view(m1.shape[0], 1)
                    .cpu()
                )
                dense_cpu_grad = bs[t].weight.grad.cpu().to_dense()
                grad_norm = torch.linalg.norm(dense_cpu_grad, dim=1, ord=2).view(
                    m1.shape[0], 1
                )
                adjusted_lr = (
                    lr * eta * weight_norm / (grad_norm + weight_decay * weight_norm)
                )
                m1_ref = adjusted_lr * (
                    dense_cpu_grad + weight_decay * bs[t].weight.cpu()
                )

                torch.testing.assert_allclose(
                    m1.index_select(dim=0, index=x[t].view(-1)).cpu(),
                    # pyre-fixme[16]: `float` has no attribute `index_select`.
                    m1_ref.index_select(dim=0, index=x[t].view(-1).cpu()),
                    atol=1.0e-4,
                    rtol=1.0e-4,
                )
                weights_new = split_weights[t]
                weights_ref = bs[t].weight.cpu() - m1_ref
                torch.testing.assert_allclose(
                    weights_new.index_select(dim=0, index=x[t].view(-1)).cpu(),
                    weights_ref.index_select(dim=0, index=x[t].view(-1).cpu()),
                    atol=1.0e-4,
                    rtol=1.0e-4,
                )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        optimizer=st.sampled_from(
            [
                OptimType.ADAM,
                OptimType.PARTIAL_ROWWISE_ADAM,
            ]
        ),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
                split_table_batched_embeddings_ops.PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_optimizers_adam(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        optimizer: OptimType,
        long_segments: bool,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
    ) -> None:
        self.execute_backward_optimizers_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            optimizer,
            long_segments,
            pooling_mode,
            use_cpu,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        optimizer=st.sampled_from(
            [
                OptimType.EXACT_ADAGRAD,
                OptimType.EXACT_ROWWISE_ADAGRAD,
            ]
        ),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
                split_table_batched_embeddings_ops.PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_optimizers_adagrad(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        optimizer: OptimType,
        long_segments: bool,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
    ) -> None:
        self.execute_backward_optimizers_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            optimizer,
            long_segments,
            pooling_mode,
            use_cpu,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        optimizer=st.sampled_from(
            [
                OptimType.LAMB,
                OptimType.PARTIAL_ROWWISE_LAMB,
            ]
        ),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
                split_table_batched_embeddings_ops.PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_optimizers_lamb(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        optimizer: OptimType,
        long_segments: bool,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
    ) -> None:
        self.execute_backward_optimizers_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            optimizer,
            long_segments,
            pooling_mode,
            use_cpu,
        )

    @given(
        T=st.integers(min_value=1, max_value=5),
        D=st.integers(min_value=2, max_value=256),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        weighted=st.booleans(),
        mixed=st.booleans(),
        optimizer=st.just(OptimType.LARS_SGD),
        long_segments=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
                split_table_batched_embeddings_ops.PoolingMode.NONE,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
    )
    def test_backward_optimizers_lars(  # noqa C901
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        optimizer: OptimType,
        long_segments: bool,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        use_cpu: bool,
    ) -> None:
        self.execute_backward_optimizers_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            optimizer,
            long_segments,
            pooling_mode,
            use_cpu,
        )

    def execute_nbit_forward_(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        weights_ty: SparseType,
        use_cpu: bool,
        use_array_for_index_remapping: bool,
        mixed_weights_ty: bool,
        output_dtype: SparseType,
    ) -> None:
        assume(
            pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM
            or not weighted
        )

        mode = "sum"
        if pooling_mode == split_table_batched_embeddings_ops.PoolingMode.SUM:
            mode = "sum"
        elif pooling_mode == split_table_batched_embeddings_ops.PoolingMode.MEAN:
            mode = "mean"
        E = int(10 ** log_E)

        if not mixed_weights_ty:
            weights_ty_list = [weights_ty] * T
        else:
            weights_ty_list = [
                SparseType.from_int(np.random.randint(low=0, high=4)) for _ in range(T)
            ]

        D_alignment = max(weights_ty_list[t].align_size() for t in range(T))
        D = round_up(D, D_alignment)

        if not mixed:
            Ds = [D] * T
            Es = [int(1e4)] * T
        else:
            Ds = [
                round_up(
                    np.random.randint(low=int(max(0.25 * D, 1)), high=int(1.0 * D)),
                    D_alignment,
                )
                for _ in range(T)
            ]
            Ds = [min(D, 128) for D in Ds]
            Es = [
                np.random.randint(low=int(0.5 * E), high=int(2.0 * E)) for _ in range(T)
            ]

        bs = [
            to_device(torch.nn.EmbeddingBag(E, D, mode=mode, sparse=True), use_cpu)
            for (E, D) in zip(Es, Ds)
        ]

        if use_cpu:
            managed = [split_table_batched_embeddings_ops.EmbeddingLocation.HOST] * T
        else:
            managed = [
                np.random.choice(
                    [
                        split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED,
                    ]
                )
                for _ in range(T)
            ]

        xs = [to_device(torch.randint(low=0, high=e, size=(B, L)), use_cpu) for e in Es]
        xws = [to_device(torch.randn(size=(B, L)), use_cpu) for _ in range(T)]

        xws_acc_type = copy.deepcopy(xws)
        cc = split_table_batched_embeddings_ops.IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    E,
                    D,
                    W_TY,
                    split_table_batched_embeddings_ops.EmbeddingLocation(M),
                )
                for (E, D, M, W_TY) in zip(Es, Ds, managed, weights_ty_list)
            ],
            pooling_mode=pooling_mode,
            index_remapping=[torch.arange(E, dtype=torch.int32) for E in Es]
            if B != 0
            else None,
            device="cpu" if use_cpu else torch.cuda.current_device(),
            use_array_for_index_remapping=use_array_for_index_remapping,
            output_dtype=output_dtype,
        )
        # Initilize the random weights for int nbit table split embedding bag
        cc.fill_random_weights()
        # NOTE: test TorchScript-compatible!
        cc = torch.jit.script(cc)

        for t in range(T):
            (weights, scale_shift) = cc.split_embedding_weights()[t]
            if scale_shift is not None:
                (E, R) = scale_shift.shape
                assert R == 4
                if weights_ty_list[t] == SparseType.INT4:
                    scales = np.random.uniform(0.01, 0.1, size=(E,)).astype(np.float16)
                    shifts = np.random.uniform(-2, 2, size=(E,)).astype(np.float16)
                if weights_ty_list[t] == SparseType.INT8:
                    scales = np.random.uniform(0.001, 0.01, size=(E,)).astype(
                        np.float16
                    )
                    shifts = np.random.uniform(-2, 2, size=(E,)).astype(np.float16)

                scale_shift[:, :] = torch.tensor(
                    np.stack([scales, shifts], axis=1).astype(np.float16).view(np.uint8)
                )

        for t in range(T):
            (weights, scale_shift) = cc.split_embedding_weights()[t]
            np_weights = weights.contiguous().cpu().numpy()

            if scale_shift is not None:
                scale_shift: np.ndarray = (
                    scale_shift.cpu()
                    .contiguous()
                    .numpy()
                    .view(np.float16)
                    .astype(np.float32)
                )

            if weights_ty_list[t] == SparseType.INT4:
                (E, D_2) = np_weights.shape
                D = D_2 * 2

                def comp(i: int) -> np.ndarray:
                    subs = np_weights.view(np.uint16) >> (i * 4)
                    sub_mask = subs & 0xF
                    result = sub_mask.astype(np.float32) * scale_shift[:, 0].reshape(
                        -1, 1
                    ).astype(np.float32) + scale_shift[:, 1].reshape(-1, 1).astype(
                        np.float32
                    )
                    return result.astype(np.float16).astype(np.float32)

                comps = [comp(i) for i in range(4)]
                comps = np.stack(comps)
                comps = comps.transpose(1, 2, 0)
                comps = comps.reshape(E, D)
                bs[t].weight.detach().copy_(torch.tensor(comps).cuda())

            elif weights_ty_list[t] == SparseType.INT2:
                (E, D_4) = np_weights.shape
                D = D_4 * 4

                # pyre-fixme[53]: Captured variable `scale_shift` is not annotated.
                # pyre-fixme[53]: Captured variable `weights` is not annotated.
                def comp(i: int) -> np.ndarray:
                    subs = np_weights.view(np.uint8) >> (i * 2)
                    sub_mask = subs & 0x3
                    result = sub_mask.astype(np.float32) * scale_shift[:, 0].reshape(
                        -1, 1
                    ).astype(np.float32) + scale_shift[:, 1].reshape(-1, 1).astype(
                        np.float32
                    )
                    return result.astype(np.float16).astype(np.float32)

                comps = [comp(i) for i in range(4)]
                comps = np.stack(comps)
                comps = comps.transpose(1, 2, 0)
                comps = comps.reshape(E, D)
                bs[t].weight.detach().copy_(torch.tensor(comps).cuda())

            elif weights_ty_list[t] == SparseType.INT8:
                (E, D) = np_weights.shape
                # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
                comps = np_weights.astype(np.float32) * scale_shift[:, 0].reshape(
                    -1, 1
                ).astype(np.float32) + scale_shift[:, 1].reshape(-1, 1).astype(
                    np.float32
                )
                bs[t].weight.detach().copy_(torch.tensor(comps).cuda())

            elif weights_ty_list[t] == SparseType.FP16:
                comps = bs[t].weight.detach().half().cpu().numpy().view(np.uint8)
                weights.copy_(torch.tensor(comps))
            elif weights_ty_list[t] == SparseType.FP32:
                comps = bs[t].weight.detach().float().cpu().numpy().view(np.uint8)
                weights.copy_(torch.tensor(comps))

        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        xw = torch.cat([xw.view(1, B, L) for xw in xws_acc_type], dim=0)

        (indices, offsets) = get_table_batched_offsets_from_dense(x, False)
        if not use_cpu:
            fc2 = (
                cc(indices.int(), offsets.int())
                if not weighted
                else cc(indices.int(), offsets.int(), xw.contiguous().view(-1))
            )
        else:
            cc = cc.cpu()
            indices, offsets = indices.cpu(), offsets.cpu()
            fc2 = (
                cc(indices.int(), offsets.int())
                if not weighted
                else cc(indices.int(), offsets.int(), xw.contiguous().view(-1).cpu())
            )

        if B == 0:
            self.assertEqual(fc2.size(), (0, cc.total_D))
            return

        fs = (
            [b_indices(b, x, use_cpu=use_cpu) for (b, x) in zip(bs, xs)]
            if not weighted
            else [
                b_indices(b, x, per_sample_weights=xw.view(-1), use_cpu=use_cpu)
                for (b, x, xw) in zip(bs, xs, xws)
            ]
        )
        f = torch.cat([f.view(B, -1) for f in fs], dim=1)
        torch.testing.assert_allclose(
            fc2.float().cpu(),
            f.float().cpu(),
            atol=1.0e-2,
            rtol=1.0e-2,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=50),
        D=st.integers(min_value=2, max_value=1024 - 8),
        B=st.integers(min_value=0, max_value=128),
        log_E=st.integers(min_value=2, max_value=4),
        L=st.integers(min_value=0, max_value=32),
        weighted=st.booleans(),
        mixed=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
            ]
        ),
        weights_ty=st.sampled_from(
            [
                SparseType.INT8,
                SparseType.INT4,
                # TODO: implement for SparseType.INT2,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        use_array_for_index_remapping=st.booleans(),
        mixed_weights_ty=st.booleans(),
        output_dtype=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
            ]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
    )
    def test_nbit_forward_int(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        weights_ty: SparseType,
        use_cpu: bool,
        use_array_for_index_remapping: bool,
        mixed_weights_ty: bool,
        output_dtype: SparseType,
    ) -> None:
        self.execute_nbit_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            pooling_mode,
            weights_ty,
            use_cpu,
            use_array_for_index_remapping,
            mixed_weights_ty,
            output_dtype,
        )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        T=st.integers(min_value=1, max_value=50),
        D=st.integers(min_value=2, max_value=1024 - 8),
        B=st.integers(min_value=0, max_value=128),
        log_E=st.integers(min_value=2, max_value=4),
        L=st.integers(min_value=0, max_value=32),
        weighted=st.booleans(),
        mixed=st.booleans(),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
            ]
        ),
        weights_ty=st.sampled_from(
            [
                SparseType.FP16,
                SparseType.FP32,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        use_array_for_index_remapping=st.booleans(),
        mixed_weights_ty=st.booleans(),
        output_dtype=st.sampled_from(
            [
                SparseType.FP32,
                SparseType.FP16,
            ]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=MAX_EXAMPLES_LONG_RUNNING,
        deadline=None,
    )
    def test_nbit_forward_fp(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        weighted: bool,
        mixed: bool,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        weights_ty: SparseType,
        use_cpu: bool,
        use_array_for_index_remapping: bool,
        mixed_weights_ty: bool,
        output_dtype: SparseType,
    ) -> None:
        self.execute_nbit_forward_(
            T,
            D,
            B,
            log_E,
            L,
            weighted,
            mixed,
            pooling_mode,
            weights_ty,
            use_cpu,
            use_array_for_index_remapping,
            mixed_weights_ty,
            output_dtype,
        )

    @given(
        T=st.integers(min_value=1, max_value=10),
        B=st.integers(min_value=1, max_value=64),
        L=st.integers(min_value=0, max_value=64),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        use_cpu_hashtable=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_nbit_forward_pruning(
        self,
        T: int,
        B: int,
        L: int,
        use_cpu: bool,
        use_cpu_hashtable: bool,
    ) -> None:
        if use_cpu_hashtable:
            assume(use_cpu)
        indices = torch.randint(low=0, high=np.iinfo(np.int32).max - 1, size=(T, B, L))
        for t in range(T):
            while (
                torch.unique(
                    indices[t], return_counts=False, return_inverse=False
                ).numel()
                != indices[t].numel()
            ):
                indices[t] = torch.randint(
                    low=0, high=np.iinfo(np.int32).max, size=(B, L)
                )

        indices = indices.view(-1).int()
        dense_indices = (
            torch.randint(low=0, high=int(1e5), size=(T, B, L)).view(-1).int()
        )
        offsets = torch.tensor([L * b_t for b_t in range(B * T + 1)]).int()

        def next_power_of_2(x: int) -> int:
            return 1 if x == 0 else 2 ** (x - 1).bit_length()

        LOAD_FACTOR = 0.8
        capacities = [int(B * L / LOAD_FACTOR) + 1 for _ in range(T)]

        hash_table = torch.empty(
            (sum(capacities), 2),
            dtype=torch.int32,
        )
        # initialize
        hash_table[:, :] = -1
        hash_table_offsets = torch.tensor([0] + np.cumsum(capacities).tolist()).long()

        torch.ops.fb.pruned_hashmap_insert(
            indices, dense_indices, offsets, hash_table, hash_table_offsets
        )

        if use_cpu_hashtable:
            ht = torch.classes.fb.PrunedMapCPU()
            ht.insert(indices, dense_indices, offsets, T)

        if not use_cpu:
            (indices, dense_indices, offsets, hash_table, hash_table_offsets) = (
                indices.cuda(),
                dense_indices.cuda(),
                offsets.cuda(),
                hash_table.cuda(),
                hash_table_offsets.cuda(),
            )
        if use_cpu_hashtable:
            dense_indices_ = ht.lookup(indices, offsets)
        else:
            dense_indices_ = torch.ops.fb.pruned_hashmap_lookup(
                indices, offsets, hash_table, hash_table_offsets
            )

        torch.testing.assert_allclose(dense_indices, dense_indices_)

        # now, use a value that does not exist in the original set of indices
        # and so should be pruned out.
        indices[:] = np.iinfo(np.int32).max

        if use_cpu_hashtable:
            dense_indices_ = ht.lookup(indices, offsets)
        else:
            dense_indices_ = torch.ops.fb.pruned_hashmap_lookup(
                indices, offsets, hash_table, hash_table_offsets
            )

        torch.testing.assert_allclose(dense_indices.clone().fill_(-1), dense_indices_)

    @unittest.skipIf(*gpu_unavailable)
    @given(
        sizes=st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=4)
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_is_uvm_tensor(self, sizes: List[int]) -> None:
        uvm_t = torch.ops.fb.new_managed_tensor(
            torch.zeros(*sizes, device="cuda:0", dtype=torch.float), sizes
        )
        assert torch.ops.fb.is_uvm_tensor(uvm_t)

    @unittest.skipIf(*gpu_unavailable)
    @given(
        sizes=st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=4)
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_uvm_to_cpu(self, sizes: List[int]) -> None:
        uvm_t = torch.ops.fb.new_managed_tensor(
            torch.zeros(*sizes, device="cuda:0", dtype=torch.float), sizes
        )
        cpu_t = torch.ops.fb.uvm_to_cpu(uvm_t)
        assert not torch.ops.fb.is_uvm_tensor(cpu_t)
        uvm_t.copy_(cpu_t)
        assert torch.ops.fb.is_uvm_tensor(uvm_t)
        # Test use of cpu tensor after freeing the uvm tensor
        del uvm_t
        cpu_t.mul_(42)

    @given(
        L=st.integers(min_value=0, max_value=16),
        H=st.integers(min_value=512, max_value=1024),
        S=st.integers(min_value=0, max_value=128),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_update_function(self, L: int, H: int, S: int) -> None:
        # Generate synthetic data
        linear_cache_indices_cpu = torch.randint(L, H, (S,))
        lxu_cache_locations_cpu = torch.clone(linear_cache_indices_cpu)

        indices = [True if np.random.rand() < 0.5 else False for _ in range(S)]
        lxu_cache_locations_cpu[indices] = -1

        cache_miss_ids = torch.clone(linear_cache_indices_cpu)
        cache_miss_ids[lxu_cache_locations_cpu != -1] = -2

        # Calculate the correct output
        unique_cache_miss_ids = torch.unique(cache_miss_ids)
        expect_out = sum(unique_cache_miss_ids >= 0)
        linear_cache_indices = to_device(
            torch.tensor(linear_cache_indices_cpu, dtype=torch.int64), use_cpu=False
        )
        lxu_cache_locations = to_device(
            torch.tensor(lxu_cache_locations_cpu, dtype=torch.int32), use_cpu=False
        )

        # Create an abstrat splittable
        D = 8
        T = 2
        E = 10 ** 3
        Ds = [D] * T
        Es = [E] * T
        emb_op = (
            split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen
        )
        cc = emb_op(
            embedding_specs=[
                (
                    E,
                    D,
                    split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING,
                    split_table_batched_embeddings_ops.ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            record_cache_metrics=RecordCacheMetrics(True, False),
        )
        cc._update_cache_miss_counter(lxu_cache_locations, linear_cache_indices)
        (
            cache_miss_forward_count,
            unique_cache_miss_count,
        ) = cc.get_cache_miss_counter().cpu()

        assert unique_cache_miss_count == expect_out
        assert cache_miss_forward_count <= unique_cache_miss_count

    @given(N=st.integers(min_value=1, max_value=8))
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_cache_miss_counter(self, N: int) -> None:

        # Create an abstrat splittable
        D = 8
        T = 2
        E = 10 ** 3
        Ds = [D] * T
        Es = [E] * T
        emb_op = (
            split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen
        )
        cc = emb_op(
            embedding_specs=[
                (
                    E,
                    D,
                    split_table_batched_embeddings_ops.EmbeddingLocation.MANAGED_CACHING,
                    split_table_batched_embeddings_ops.ComputeDevice.CUDA,
                )
                for (E, D) in zip(Es, Ds)
            ],
            record_cache_metrics=RecordCacheMetrics(True, True),
        )

        # Create fake input data and the target output
        xs = []
        x1 = torch.Tensor([[[1], [1]], [[3], [4]]])
        x1 = to_device(torch.tensor(x1, dtype=torch.int64), use_cpu=False)

        x2 = torch.Tensor([[[2], [1]], [[3], [4]]])
        x2 = to_device(torch.tensor(x2, dtype=torch.int64), use_cpu=False)

        x3 = torch.Tensor([[[5], [6]], [[7], [8]]])
        x3 = to_device(torch.tensor(x3, dtype=torch.int64), use_cpu=False)

        xs.append(x1)
        xs.append(x2)
        xs.append(x3)

        target_counter_list = [[1, 3], [2, 4], [3, 8]]
        target_tablewise_cache_miss_list = [[1, 2], [2, 2], [4, 4]]
        for x, t_counter, t_tablewise_cache_miss in zip(
            xs, target_counter_list, target_tablewise_cache_miss_list
        ):
            (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=False)
            for _ in range(N):
                cc(indices, offsets)
                (
                    cache_miss_forward_count,
                    unique_cache_miss_count,
                ) = cc.get_cache_miss_counter().cpu()
                tablewise_cache_miss = cc.get_table_wise_cache_miss().cpu()
                assert cache_miss_forward_count == t_counter[0]
                assert unique_cache_miss_count == t_counter[1]
                for i in range(len(tablewise_cache_miss)):
                    assert t_tablewise_cache_miss[i] == tablewise_cache_miss[i]

    @given(
        T=st.integers(min_value=1, max_value=64),
        B=st.integers(min_value=1, max_value=64),
        max_L=st.integers(min_value=1, max_value=64),
        bounds_check_mode=st.sampled_from(
            [
                BoundsCheckMode.FATAL,
                BoundsCheckMode.WARNING,
                BoundsCheckMode.IGNORE,
            ]
        ),
        use_cpu=st.booleans(),
        dtype=st.sampled_from(
            [
                torch.int64,
                torch.int32,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_bounds_check(
        self,
        T: int,
        B: int,
        max_L: int,
        bounds_check_mode: BoundsCheckMode,
        use_cpu: bool,
        dtype: torch.dtype,
    ) -> None:
        rows_per_table = torch.tensor(
            np.random.randint(low=1, high=1000, size=(T,))
        ).long()
        Ls = np.random.randint(low=0, high=max_L, size=(T, B))
        indices = [
            np.random.randint(low=0, high=rows_per_table[t], size=Ls[t, b])
            for t in range(T)
            for b in range(B)
        ]
        indices = torch.tensor(np.concatenate(indices, axis=0)).to(dtype)
        offsets = torch.tensor([0] + np.cumsum(Ls.flatten()).tolist()).to(dtype)
        warning = torch.tensor([0]).long()

        assert indices.numel() == np.sum(Ls).item()
        assert offsets[-1] == np.sum(Ls).item()
        if not use_cpu:
            indices, offsets, rows_per_table, warning = (
                indices.cuda(),
                offsets.cuda(),
                rows_per_table.cuda(),
                warning.cuda(),
            )
        indices_copy = indices.clone()
        torch.ops.fb.bounds_check_indices(
            rows_per_table, indices, offsets, bounds_check_mode, warning
        )
        # we don't modify when we are in-bounds.
        torch.testing.assert_allclose(indices_copy, indices)
        indices[:] = torch.iinfo(dtype).max
        if bounds_check_mode != BoundsCheckMode.FATAL:
            torch.ops.fb.bounds_check_indices(
                rows_per_table, indices, offsets, bounds_check_mode, warning
            )
            torch.testing.assert_allclose(indices, torch.zeros_like(indices))
            if bounds_check_mode == BoundsCheckMode.WARNING:
                self.assertEqual(warning.item(), indices.numel())
        else:
            if use_cpu and indices.numel():
                with self.assertRaises(RuntimeError):
                    torch.ops.fb.bounds_check_indices(
                        rows_per_table, indices, offsets, bounds_check_mode, warning
                    )
            # It would be nice to test the CUDA implementation of BoundsCheckMode==FATAL,
            # but the device assert kills the CUDA context and requires a process restart,
            # which is a bit inconvenient.


if __name__ == "__main__":
    unittest.main()
