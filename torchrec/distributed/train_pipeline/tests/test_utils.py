#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchrec.distributed.train_pipeline.utils import _get_node_args

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestUtils(unittest.TestCase):
    def test_get_node_args_helper_call_module_kjt(self) -> None:
        graph = torch.fx.Graph()
        kjt_args = []

        kjt_args.append(
            torch.fx.Node(graph, "values", "placeholder", "torch.Tensor", (), {})
        )
        kjt_args.append(
            torch.fx.Node(graph, "lengths", "placeholder", "torch.Tensor", (), {})
        )
        kjt_args.append(
            torch.fx.Node(
                graph, "weights", "call_module", "PositionWeightedModule", (), {}
            )
        )

        kjt_node = torch.fx.Node(
            graph,
            "keyed_jagged_tensor",
            "call_function",
            KeyedJaggedTensor,
            tuple(kjt_args),
            {},
        )

        num_found = 0
        _, num_found = _get_node_args(kjt_node)

        # Weights is call_module node, so we should only find 2 args unmodified
        self.assertEqual(num_found, len(kjt_args) - 1)
