#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Optional

import torch
import torch._dynamo.config
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.utils.data import IterableDataset
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.random import RandomRecDataset
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs_registry,
    QCommsConfig,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from tqdm import tqdm


def _get_random_dataset(
    num_embeddings: int,
    batch_size: int = 32,
) -> IterableDataset:
    return RandomRecDataset(
        keys=DEFAULT_CAT_NAMES,
        batch_size=batch_size,
        hash_size=num_embeddings,
        ids_per_feature=3,
        num_dense=len(DEFAULT_INT_NAMES),
    )


@record
def main() -> None:
    train()


def train(
    num_embeddings: int = 1024**2,
    embedding_dim: int = 128,
    dense_arch_layer_sizes: Optional[List[int]] = None,
    over_arch_layer_sizes: Optional[List[int]] = None,
    learning_rate: float = 0.1,
    num_iterations: int = 1000,
    qcomm_forward_precision: Optional[CommType] = CommType.FP16,
    qcomm_backward_precision: Optional[CommType] = CommType.BF16,
) -> None:
    """
    Constructs and trains a DLRM model (using random dummy data). Each script is run on each process (rank) in SPMD fashion.
    The embedding layers will be sharded across available ranks

    qcomm_forward_precision: Compression used in forwards pass. FP16 is the recommended usage. INT8 and FP8 are in development, but feel free to try them out.
    qcomm_backward_precision: Compression used in backwards pass. We recommend using BF16 to ensure training stability.

    The effects of quantized comms will be most apparent in large training jobs across multiple nodes where inter host communication is expensive.
    """
    if dense_arch_layer_sizes is None:
        dense_arch_layer_sizes = [64, embedding_dim]
    if over_arch_layer_sizes is None:
        over_arch_layer_sizes = [64, 1]

    # Init process_group , device, rank, backend
    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)

    # Construct DLRM module
    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]
    dlrm_model = DLRM(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("meta")
        ),
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=dense_arch_layer_sizes,
        over_arch_layer_sizes=over_arch_layer_sizes,
        dense_device=device,
    )
    train_model = DLRMTrain(dlrm_model)

    apply_optimizer_in_backward(
        RowWiseAdagrad,
        train_model.model.sparse_arch.parameters(),
        {"lr": learning_rate},
    )
    qcomm_codecs_registry = (
        get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                # pyre-ignore
                forward_precision=qcomm_forward_precision,
                # pyre-ignore
                backward_precision=qcomm_backward_precision,
            )
        )
        if backend == "nccl"
        else None
    )
    sharder = EmbeddingBagCollectionSharder(qcomm_codecs_registry=qcomm_codecs_registry)

    model = DistributedModelParallel(
        module=train_model,
        device=device,
        # pyre-ignore
        sharders=[sharder],
    )

    non_fused_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        lambda params: torch.optim.Adagrad(params, lr=learning_rate),
    )
    # Overlap comm/compute/device transfer during training through train_pipeline
    train_pipeline = TrainPipelineSparseDist(
        model,
        non_fused_optimizer,
        device,
    )

    # train model
    train_iterator = iter(
        _get_random_dataset(
            num_embeddings=num_embeddings,
        )
    )

    # First time we run the model it does some register_buffer which Dynamo
    # chokes on
    print(model(next(train_iterator).to(device)))  # warmup, input dists

    #train_model.forward = torch.compile(fullgraph=True, backend="eager")(train_model.forward)

    print(model(next(train_iterator).to(device)))
    print(model(next(train_iterator).to(device)))
    print(model(next(train_iterator).to(device)))

    #for _ in tqdm(range(int(num_iterations)), mininterval=5.0):
    #    train_pipeline.progress(train_iterator)

import torch.library
fbgemm_meta_lib = torch.library.Library("fbgemm", "IMPL", "Meta")

def register_meta(op_name, overload_name="default"):
    def wrapper(fn):
        fbgemm_meta_lib.impl(getattr(getattr(torch.ops.fbgemm, op_name), overload_name), fn)
        return fn

    return wrapper


from torch._prims_common import check
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode

def getScalarType(output_dtype: int):
    return SparseType.from_int(output_dtype).as_dtype()

@register_meta("bounds_check_indices")
def bounds_check_indices_meta(rows_per_table, indices, offsets, bounds_check_mode, warning, weights=None, B_offsets=None, max_B=-1):
    pass

@register_meta("split_embedding_codegen_forward_cpu")
def split_embedding_codegen_forward_cpu_meta(weights, weights_offsets, D_offsets, total_D, hash_size_cumsum, indices, offsets, pooling_mode, indice_weights, output_dtype):
    T = D_offsets.numel() - 1
    check(T > 0, lambda: f"expect T > 0, but got {T} (from D_offsets.size() = {D_offsets.size()})")
    B = (offsets.size(0) - 1) // T
    check(B >= 0, lambda: f"expect B >= 0, but got {B} (from {offsets.size(0)})")

    dt = getScalarType(output_dtype)
    output = weights.new_empty((B, total_D), dtype=dt)

    assert indice_weights is None or indice_weights.dtype != torch.float16

    return output

"""
# this is the cpu one
@register_meta("split_embedding_codegen_lookup_rowwise_adagrad_function")
def split_embedding_codegen_lookup_rowwise_adagrad_function_meta(host_weights, weights_placements, weights_offsets, D_offsets, total_D, max_D, hash_size_cumsum, total_hash_size_bits, indices, offsets, pooling_mode, indice_weights, feature_requires_grad, gradient_clipping, max_gradient, stochastic_rounding, momentum1_host, momentum1_placements, momentum1_offsets, eps = 0, learning_rate = 0, weight_decay = 0.0, weight_decay_mode = 0, max_norm = 0.0, output_dtype=0):
    return split_embedding_codegen_forward_cpu_meta(host_weights, weights_offsets, D_offsets, total_D, hash_size_cumsum, indices, offsets, pooling_mode, indice_weights, output_dtype)
"""

kINT8QparamsBytes = 8

@register_meta("split_embedding_codegen_lookup_rowwise_adagrad_function")
def split_embedding_codegen_lookup_rowwise_adagrad_function_meta(
    placeholder_autograd_tensor,
    dev_weights, uvm_weights,
    lxu_cache_weights,
    weights_placements,
    weights_offsets,
    D_offsets,
    total_D,
    max_D,
    hash_size_cumsum,
    total_hash_size_bits,
    indices,
    offsets,
    pooling_mode,
    indice_weights,
    feature_requires_grad,
    lxu_cache_locations,
    gradient_clipping,
    max_gradient,
    stochastic_rounding,
    momentum1_dev,
    momentum1_uvm,
    momentum1_placements,
    momentum1_offsets,
    eps = 0,
    learning_rate = 0,
    weight_decay = 0.0,
    weight_decay_mode = 0,
    max_norm = 0.0,
    output_dtype=0,
    B_offsets=None,
    vbe_output_offsets_feature_rank=None,
    vbe_B_offsets_rank_per_feature=None,
    max_B=-1,
    max_B_feature_rank=-1,
    vbe_output_size=-1,
    is_experimental=False,
):
    if B_offsets is not None:
        assert False
    else:
        if pooling_mode is PoolingMode.NONE:
            # SplitNoBagLookupFunction_rowwise_adagrad_Op
            # -> split_embedding_nobag_codegen_forward_unweighted_cuda
            total_L = indices.numel()
            T = weights_offsets.numel()
            check(T > 0, lambda: "T > 0")
            total_B = offsets.size(0) - 1
            B = total_B // T
            check(B >= 0, lambda: "B >= 0")
            D = max_D  # per SplitNoBagLookupFunction_rowwise_adagrad_Op
            check(D > 0, lambda: "D > 0")
            check(D % 4 == 0, lambda: "D % 4 == 0")
            assert SparseType.from_int(output_dtype) in (SparseType.FP32, SparseType.FP16, SparseType.BF16, SparseType.INT8)
            adjusted_D = D
            if output_dtype == SparseType.INT8:
                adjusted_D += T * kINT8QparamsBytes
            output = dev_weights.new_empty((total_L, adjusted_D), dtype=getScalarType(output_dtype))
            return output
        else:
            # SplitLookupFunction_rowwise_adagrad_Op
            if indice_weights is None:
                # split_embedding_codegen_forward_unweighted_cuda
                T = weights_offsets.numel()
                check(T > 0, lambda: "T > 0")
                total_B = offsets.size(0) - 1
                B = total_B // T
                check(B >= 0, lambda: "B >= 0")
                check(total_D >= 0, lambda: "")
                check(total_D % 4 == 0, lambda: "")
                check(max_D <= 1024, lambda: "")
                assert SparseType.from_int(output_dtype) in (SparseType.FP32, SparseType.FP16, SparseType.BF16, SparseType.INT8)
                total_adjusted_D = total_D
                if output_dtype == SparseType.INT8:
                    total_adjusted_D += T * kINT8QparamsBytes
                output = dev_weights.new_empty((B, total_adjusted_D), dtype=getScalarType(output_dtype))
                return output

            else:
                assert False

@register_meta("permute_2D_sparse_data")
def permute_2D_sparse_data_meta(permute, lengths, values, weights=None, permuted_lengths_sum=None):
    check(lengths.dim() == 2, lambda: "")
    T = permute.numel()
    B = lengths.size(1)
    indices = values
    permuted_lengths = lengths.new_empty([T, B])
    permuted_indices_size = 0
    if permuted_lengths_sum is not None:
        permuted_indices_size = permuted_lengths_sum
    else:
        raise NotImplementedError("TODO: data dependent permute_2D")
    permuted_indices = indices.new_empty(permuted_indices_size)
    permuted_weights = None
    if weights is not None:
        permuted_weights = weights.new_empty(permuted_indices_size)
    return permuted_lengths, permuted_indices, permuted_weights


@register_meta("permute_1D_sparse_data")
def permute_1D_sparse_data_meta(permute, lengths, values, weights=None, permuted_lengths_sum=None):
    indices = values
    permuted_lengths_size = permute.numel()
    permuted_lengths = lengths.new_empty([permuted_lengths_size])
    permuted_indices_size = 0
    if permuted_lengths_sum is not None:
        permuted_indices_size = permuted_lengths_sum
    else:
        raise NotImplementedError("TODO: data dependent permute_1D")
    permuted_indices = indices.new_empty(permuted_indices_size)
    permuted_weights = None
    if weights is not None:
        permuted_weights = weights.new_empty(permuted_indices_size)
    return permuted_lengths, permuted_indices, permuted_weights



torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

if __name__ == "__main__":
    main()
