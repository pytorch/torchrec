#!/usr/bin/env python3

import unittest
from typing import Any, Dict, List

import torch
from torch.autograd import Variable
from torchrec.optim.keyed import KeyedOptimizer
from torchrec.optim.warmup import WarmupOptimizer, WarmupStage, WarmupPolicy


class DummyKeyedOptimizer(KeyedOptimizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # pyre-ignore[2]
    def step(self, closure: Any) -> None:
        pass  # Override NotImplementedError.


class TestWarmupOptimizer(unittest.TestCase):
    def test_load_state_dict(self) -> None:
        def get_optimizer() -> WarmupOptimizer:
            param_1_t = torch.tensor([1.0, 2.0])
            param_1 = Variable(param_1_t)
            keyed_optimizer = DummyKeyedOptimizer(
                {"param_1": param_1}, {}, [{"params": [param_1]}]
            )
            warmup_optimizer = WarmupOptimizer(
                keyed_optimizer,
                stages=[
                    WarmupStage(
                        WarmupPolicy.LINEAR, max_iters=100, value=1e-2, lr_scale=1
                    ),
                ],
            )
            return warmup_optimizer

        warmup_optimizer_1 = get_optimizer()
        num_iters = 10
        for _ in range(num_iters):
            warmup_optimizer_1.zero_grad()
            warmup_optimizer_1.step()
        self.assertTrue(
            warmup_optimizer_1.param_groups[0]["iter"],  # pyre-ignore[16]
            num_iters + 1,
        )

        warmup_optimizer_state_dict = warmup_optimizer_1.state_dict()

        warmup_optimizer_2 = get_optimizer()
        warmup_optimizer_2.load_state_dict(warmup_optimizer_state_dict)

        # pyre-ignore[2, 3]
        def remove_params(param_groups: List[Dict[Any, Any]]) -> List[Dict[Any, Any]]:
            return [
                {k: v for (k, v) in param_groups.items() if k != "params"}
                for param_groups in param_groups
            ]

        self.assertEqual(warmup_optimizer_1.state, warmup_optimizer_2.state)
        self.assertEqual(
            remove_params(warmup_optimizer_1.param_groups),  # pyre-ignore[6]
            remove_params(warmup_optimizer_2.param_groups),  # pyre-ignore[6]
        )


if __name__ == "__main__":
    unittest.main()
