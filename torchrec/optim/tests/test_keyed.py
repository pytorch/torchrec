#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from typing import Dict, Any, List

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torchrec.optim.keyed import (
    CombinedOptimizer,
    KeyedOptimizer,
    OptimizerWrapper,
)
from torchrec.tests.utils import get_free_port


class TestKeyedOptimizer(unittest.TestCase):
    def _assert_state_dict_equals(
        self, dict1: Dict[str, Any], dict2: Dict[str, Any]
    ) -> None:
        self.assertEqual(dict1["param_groups"], dict2["param_groups"])
        self.assertEqual(
            dict1["state"]["param_2"],
            dict2["state"]["param_2"],
        )
        torch.testing.assert_close(
            dict1["state"]["param_1"]["tensor"],
            dict2["state"]["param_1"]["tensor"],
        )

        torch.testing.assert_close(
            dict1["state"]["param_1"]["sharded_tensor"].local_shards()[0].tensor,
            dict2["state"]["param_1"]["sharded_tensor"].local_shards()[0].tensor,
        )

    def test_load_state_dict(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)

        # Set up example KeyedOptimizer.
        param_1_t, param_2_t = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
        param_1, param_2 = Variable(param_1_t), Variable(param_2_t)
        keyed_optimizer = KeyedOptimizer(
            {"param_1": param_1, "param_2": param_2},
            {
                param_1: {
                    "one": 1.0,
                    "tensor": torch.tensor([5.0, 6.0]),
                    "sharded_tensor": dist._sharded_tensor.full(
                        # pyre-ignore [28]
                        dist._sharded_tensor.ChunkShardingSpec(
                            dim=0, placements=["rank:0/cpu"]
                        ),
                        (4,),
                        fill_value=1.0,
                    ),
                },
                param_2: {"two": 2.0},
            },
            [
                {
                    "params": [param_1],
                    "param_group_val_0": 3.0,
                    "param_group_val_1": 4.0,
                },
                {
                    "params": [param_2],
                    "param_group_val_0": 5.0,
                    "param_group_val_1": 6.0,
                },
            ],
        )

        # Assert state_dict is as expected.
        state: Dict[str, Any] = {
            "param_1": {
                "one": 1.0,
                "tensor": torch.tensor([5.0, 6.0]),
                "sharded_tensor": dist._sharded_tensor.full(
                    # pyre-ignore [28]
                    dist._sharded_tensor.ChunkShardingSpec(
                        dim=0, placements=["rank:0/cpu"]
                    ),
                    (4,),
                    fill_value=1.0,
                ),
            },
            "param_2": {"two": 2.0},
        }
        param_groups: List[Dict[str, Any]] = [
            {
                "params": ["param_1"],
                "param_group_val_0": 3.0,
                "param_group_val_1": 4.0,
            },
            {
                "params": ["param_2"],
                "param_group_val_0": 5.0,
                "param_group_val_1": 6.0,
            },
        ]
        expected_state_dict = {
            "state": state,
            "param_groups": param_groups,
        }
        self._assert_state_dict_equals(
            expected_state_dict, keyed_optimizer.state_dict()
        )

        # Modify state dict and call load_state_dict.
        # pyre-ignore [6]
        expected_state_dict["state"]["param_1"]["one"] = 10.0
        # pyre-ignore [6]
        expected_state_dict["state"]["param_1"]["tensor"] = torch.tensor([50.0, 60.0])
        # pyre-ignore [6]
        expected_state_dict["state"]["param_1"][
            "sharded_tensor"
        ] = dist._sharded_tensor.full(
            # pyre-ignore [28]
            dist._sharded_tensor.ChunkShardingSpec(dim=0, placements=["rank:0/cpu"]),
            (4,),
            fill_value=10.0,
        )
        # pyre-ignore [6]
        expected_state_dict["param_groups"][0]["param_group_val_0"] = 8.0
        # pyre-ignore [6]
        expected_state_dict["param_groups"][1]["param_group_val_1"] = 9.0

        keyed_optimizer.load_state_dict(expected_state_dict)
        self._assert_state_dict_equals(
            expected_state_dict, keyed_optimizer.state_dict()
        )

    def test_non_param_state_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "All state keys must be params."):
            param_1_t = torch.tensor([1.0, 2.0])
            param_1 = Variable(param_1_t)
            KeyedOptimizer(
                {"param_1": param_1},
                {param_1: 1.0, "non_param_state_key": 2.0},
                [{"params": [param_1], "param_group_val_0": 3.0}],
            )


class TestCombinedOptimizer(unittest.TestCase):
    def test_load_state_dict(self) -> None:
        # Set up example KeyedOptimizer 1.
        param_1_t = torch.tensor([1.0, 2.0])
        param_1 = Variable(param_1_t)
        keyed_optimizer_1 = KeyedOptimizer(
            {"param_1": param_1},
            {param_1: {"one": 1.0}},
            [{"params": [param_1], "param_group_val_0": 2.0}],
        )

        # Set up example KeyedOptimizer 2.
        param_2_t = torch.tensor([-1.0, -2.0])
        param_2 = Variable(param_2_t)
        keyed_optimizer_2 = KeyedOptimizer(
            {"param_2": param_2},
            {param_2: {"two": -1.0}},
            [{"params": [param_2], "param_group_val_0": -2.0}],
        )

        combined_optimizer = CombinedOptimizer(
            [("ko1", keyed_optimizer_1), ("", keyed_optimizer_2)]
        )

        combined_optimizer_state_dict = combined_optimizer.state_dict()
        combined_optimizer_state_dict["state"]["ko1.param_1"] = {"one": 999}
        combined_optimizer_state_dict["state"]["param_2"] = {"two": 998}
        combined_optimizer_state_dict["param_groups"][0]["param_group_val_0"] = 997
        combined_optimizer_state_dict["param_groups"][1]["param_group_val_0"] = 996

        combined_optimizer.load_state_dict(combined_optimizer_state_dict)

        # Check that optimizers in the combined optimizer have their state and
        # param_groups updated.
        self.assertEqual(keyed_optimizer_1.state[param_1], {"one": 999})
        self.assertEqual(keyed_optimizer_2.state[param_2], {"two": 998})
        # pyre-ignore[16]
        self.assertEqual(keyed_optimizer_1.param_groups[0]["param_group_val_0"], 997)
        self.assertEqual(keyed_optimizer_2.param_groups[0]["param_group_val_0"], 996)


class TestOptimizerWrapper(unittest.TestCase):
    def test_load_state_dict(self) -> None:
        param_1_t = torch.tensor([1.0, 2.0])
        param_1 = Variable(param_1_t)
        keyed_optimizer = KeyedOptimizer(
            {"param_1": param_1},
            {param_1: {"one": 1.0}},
            [{"params": [param_1], "param_group_val_0": 2.0}],
        )
        optimizer_wrapper = OptimizerWrapper(keyed_optimizer)

        optimizer_wrapper_state_dict = optimizer_wrapper.state_dict()
        optimizer_wrapper_state_dict["state"]["param_1"] = {"one": 999}
        optimizer_wrapper_state_dict["param_groups"][0]["param_group_val_0"] = 998
        optimizer_wrapper.load_state_dict(optimizer_wrapper_state_dict)

        # Check that both keyed_optimizer and optimizer_wrapper have their state and
        # param_groups updated.
        self.assertEqual(keyed_optimizer.state[param_1], {"one": 999})
        self.assertEqual(optimizer_wrapper.state[param_1], {"one": 999})
        # pyre-ignore[16]
        self.assertEqual(keyed_optimizer.param_groups[0]["param_group_val_0"], 998)
        self.assertEqual(optimizer_wrapper.param_groups[0]["param_group_val_0"], 998)


if __name__ == "__main__":
    unittest.main()
