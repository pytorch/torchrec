#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)


def populate_fx_names(quant_ebc: QuantEmbeddingBagCollection) -> None:
    """
    Assigns fx path to non registered lookup modules. This allows the Torchrec tracer to fallback to
    emb_module._fx_path for table batched embeddings.
    """
    for emb_configs, emb_module in zip(
        quant_ebc._key_to_tables, quant_ebc._emb_modules
    ):
        table_names = []
        for config in emb_configs:
            table_names.append(config.name)
        joined_table_names = ",".join(table_names)
        emb_module._fx_path = f"emb_module.{joined_table_names}"


def meta_to_cpu_placement(module: DistributedModelParallel) -> None:
    assert hasattr(module, "_dmp_wrapped_module")
    _meta_to_cpu_placement(module.module, module, "_dmp_wrapped_module")


def _meta_to_cpu_placement(
    module: nn.Module, root_module: nn.Module, name: str
) -> None:
    if isinstance(module, QuantEmbeddingBagCollection) and module.device.type == "meta":
        qebc_cpu = QuantEmbeddingBagCollection(
            tables=module.embedding_bag_configs(),
            is_weighted=module.is_weighted(),
            device=torch.device("cpu"),
            output_dtype=module.output_dtype(),
        )
        setattr(root_module, name, qebc_cpu)
        return
    for name, submodule in module.named_children():
        _meta_to_cpu_placement(submodule, module, name)
