#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from typing import Dict, List, Type

import torch
import torch.distributed as dist
from torchrec.metrics.ctr import CTRMetric
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    rec_metric_value_test_helper,
    rec_metric_value_test_launcher,
    TestMetric,
)


class TestCTRMetric(TestMetric):
    @staticmethod
    def _get_states(
        labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        ctr_num = torch.sum(labels * weights)
        ctr_denom = torch.sum(weights)
        num_samples = torch.tensor(labels.size()[0]).double()
        return {"ctr_num": ctr_num, "ctr_denom": ctr_denom, "num_samples": num_samples}

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.where(
            states["ctr_denom"] == 0.0, 0.0, states["ctr_num"] / states["ctr_denom"]
        ).double()


WORLD_SIZE = 4


class CTRMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = CTRMetric
    task_name: str = "ctr"

    @staticmethod
    def _test_ctr(
        target_clazz: Type[RecMetric],
        target_compute_mode: RecComputeMode,
        task_names: List[str],
        fused_update_limit: int = 0,
        compute_on_all_ranks: bool = False,
        should_validate_update: bool = False,
    ) -> None:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend="gloo",
            world_size=world_size,
            rank=rank,
        )

        ctr_metrics, test_metrics = rec_metric_value_test_helper(
            target_clazz=target_clazz,
            target_compute_mode=target_compute_mode,
            test_clazz=TestCTRMetric,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=False,
            should_validate_update=should_validate_update,
            world_size=world_size,
            my_rank=rank,
            task_names=task_names,
        )

        if rank == 0:
            for name in task_names:
                assert torch.allclose(
                    ctr_metrics[f"ctr-{name}|lifetime_ctr"], test_metrics[0][name]
                )
                assert torch.allclose(
                    ctr_metrics[f"ctr-{name}|window_ctr"], test_metrics[1][name]
                )
                assert torch.allclose(
                    ctr_metrics[f"ctr-{name}|local_lifetime_ctr"], test_metrics[2][name]
                )
                assert torch.allclose(
                    ctr_metrics[f"ctr-{name}|local_window_ctr"], test_metrics[3][name]
                )
        dist.destroy_process_group()

    def test_unfused_ctr(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=CTRMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestCTRMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_ctr,
        )

    def test_fused_ctr(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=CTRMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestCTRMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_ctr,
        )
