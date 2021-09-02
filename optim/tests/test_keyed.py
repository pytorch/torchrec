#!/usr/bin/env python3

import unittest

import torch
from torch.autograd import Variable
from torchrec.optim.keyed import (
    CombinedOptimizer,
    KeyedOptimizer,
    OptimizerWrapper,
)


class TestKeyedOptimizer(unittest.TestCase):
    def test_load_state_dict(self) -> None:
        # Set up example KeyedOptimizer.
        param_1_t, param_2_t = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
        param_1, param_2 = Variable(param_1_t), Variable(param_2_t)
        keyed_optimizer = KeyedOptimizer(
            {"param_1": param_1, "param_2": param_2},
            {param_1: 1.0, param_2: 2.0},
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
        state_dict = keyed_optimizer.state_dict()
        expected_state_dict = {
            "state": {"param_1": 1.0, "param_2": 2.0},
            "param_groups": [
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
            ],
        }
        self.assertEqual(state_dict, expected_state_dict)

        # Modify state dict and call load_state_dict.
        state_dict["state"]["param_1"] = 7.0
        state_dict["param_groups"][0]["param_group_val_0"] = 8.0
        state_dict["param_groups"][1]["param_group_val_1"] = 9.0

        keyed_optimizer.load_state_dict(state_dict)

        # Assert keyed_optimizer state was correctly updated based on modified
        # state_dict.
        self.assertEqual(
            keyed_optimizer.params, {"param_1": param_1, "param_2": param_2}
        )
        self.assertEqual(
            keyed_optimizer.state,
            {param_1: 7.0, param_2: 2.0},
        )
        self.assertEqual(
            keyed_optimizer.param_groups,
            [
                {
                    "params": [param_1],
                    "param_group_val_0": 8.0,
                    "param_group_val_1": 4.0,
                },
                {
                    "params": [param_2],
                    "param_group_val_0": 5.0,
                    "param_group_val_1": 9.0,
                },
            ],
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
            {param_1: 1.0},
            [{"params": [param_1], "param_group_val_0": 2.0}],
        )

        # Set up example KeyedOptimizer 2.
        param_2_t = torch.tensor([-1.0, -2.0])
        param_2 = Variable(param_2_t)
        keyed_optimizer_2 = KeyedOptimizer(
            {"param_2": param_2},
            {param_2: -1.0},
            [{"params": [param_2], "param_group_val_0": -2.0}],
        )

        combined_optimizer = CombinedOptimizer(
            [("ko1", keyed_optimizer_1), ("", keyed_optimizer_2)]
        )

        combined_optimizer_state_dict = combined_optimizer.state_dict()
        combined_optimizer_state_dict["state"]["ko1.param_1"] = 999
        combined_optimizer_state_dict["state"]["param_2"] = 998
        combined_optimizer_state_dict["param_groups"][0]["param_group_val_0"] = 997
        combined_optimizer_state_dict["param_groups"][1]["param_group_val_0"] = 996

        combined_optimizer.load_state_dict(combined_optimizer_state_dict)

        # Check that optimizers in the combined optimizer have their state and
        # param_groups updated.
        self.assertEqual(keyed_optimizer_1.state[param_1], 999)
        self.assertEqual(keyed_optimizer_2.state[param_2], 998)
        # pyre-ignore[16]
        self.assertEqual(keyed_optimizer_1.param_groups[0]["param_group_val_0"], 997)
        self.assertEqual(keyed_optimizer_2.param_groups[0]["param_group_val_0"], 996)


class TestOptimizerWrapper(unittest.TestCase):
    def test_load_state_dict(self) -> None:
        param_1_t = torch.tensor([1.0, 2.0])
        param_1 = Variable(param_1_t)
        keyed_optimizer = KeyedOptimizer(
            {"param_1": param_1},
            {param_1: 1.0},
            [{"params": [param_1], "param_group_val_0": 2.0}],
        )
        optimizer_wrapper = OptimizerWrapper(keyed_optimizer)

        optimizer_wrapper_state_dict = optimizer_wrapper.state_dict()
        optimizer_wrapper_state_dict["state"]["param_1"] = 999
        optimizer_wrapper_state_dict["param_groups"][0]["param_group_val_0"] = 998
        optimizer_wrapper.load_state_dict(optimizer_wrapper_state_dict)

        # Check that both keyed_optimizer and optimizer_wrapper have their state and
        # param_groups updated.
        self.assertEqual(keyed_optimizer.state[param_1], 999)
        self.assertEqual(optimizer_wrapper.state[param_1], 999)
        # pyre-ignore[16]
        self.assertEqual(keyed_optimizer.param_groups[0]["param_group_val_0"], 998)
        self.assertEqual(optimizer_wrapper.param_groups[0]["param_group_val_0"], 998)


if __name__ == "__main__":
    unittest.main()
