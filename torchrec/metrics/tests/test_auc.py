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
from torchrec.metrics.auc import AUCMetric
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric, RecTaskInfo
from torchrec.metrics.tests.test_utils import (
    rec_metric_value_test_helper,
    rec_metric_value_test_launcher,
    TestMetric,
)


def compute_auc(
    predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    _, sorted_index = torch.sort(predictions, descending=True)
    sorted_labels = torch.index_select(labels, dim=0, index=sorted_index)
    sorted_weights = torch.index_select(weights, dim=0, index=sorted_index)
    cum_fp = torch.cumsum(sorted_weights * (1.0 - sorted_labels), dim=0)
    cum_tp = torch.cumsum(sorted_weights * sorted_labels, dim=0)
    auc = torch.where(
        cum_fp[-1] * cum_tp[-1] == 0,
        0.5,  # 0.5 is the no-signal default value for auc.
        torch.trapz(cum_tp, cum_fp) / cum_fp[-1] / cum_tp[-1],
    )
    return auc


class TestAUCMetric(TestMetric):
    def __init__(
        self,
        world_size: int,
        rec_tasks: List[RecTaskInfo],
    ) -> None:
        super().__init__(
            world_size,
            rec_tasks,
            compute_lifetime_metric=False,
            local_compute_lifetime_metric=False,
        )

    @staticmethod
    def _aggregate(
        states: Dict[str, torch.Tensor], new_states: Dict[str, torch.Tensor]
    ) -> None:
        for k, v in new_states.items():
            if k not in states:
                states[k] = v.double().detach().clone()
            else:
                states[k] = torch.cat([states[k], v.double()])

    @staticmethod
    def _get_states(
        labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            "predictions": predictions,
            "weights": weights,
            "labels": labels,
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_auc(states["predictions"], states["labels"], states["weights"])


WORLD_SIZE = 4


class AUCMetricTest(unittest.TestCase):
    clazz: Type[RecMetric] = AUCMetric
    task_name: str = "auc"

    @staticmethod
    def _test_auc(
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

        auc_metrics, test_metrics = rec_metric_value_test_helper(
            target_clazz=target_clazz,
            target_compute_mode=target_compute_mode,
            test_clazz=TestAUCMetric,
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
                    auc_metrics[f"auc-{name}|window_auc"], test_metrics[1][name]
                ), (auc_metrics[f"auc-{name}|window_auc"], test_metrics[1][name])
                assert torch.allclose(
                    auc_metrics[f"auc-{name}|local_window_auc"], test_metrics[3][name]
                ), (auc_metrics[f"auc-{name}|local_window_auc"], test_metrics[3][name])
        dist.destroy_process_group()

    def test_unfused_auc(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AUCMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestAUCMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_auc,
        )

    def test_fused_auc(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=AUCMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestAUCMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_auc,
        )


class AUCMetricValueTest(unittest.TestCase):
    r"""This set of tests verify the computation logic of AUC in several
    corner cases that we know the computation results. The goal is to
    provide some confidence of the correctness of the math formula.
    """

    def setUp(self) -> None:
        self.predictions = {"DefaultTask": None}
        self.weights = {"DefaultTask": None}
        self.labels = {"DefaultTask": None}
        self.batches = {
            "predictions": self.predictions,
            "weights": self.weights,
            "labels": self.labels,
        }
        self.auc = AUCMetric(
            world_size=1,
            my_rank=0,
            batch_size=20000,
            tasks=[DefaultTaskInfo],
        )

    def test_calc_auc_perfect(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(10000)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor([[0] * 10000 + [1] * 10000])
        self.weights["DefaultTask"] = torch.Tensor(
            [[1] * 5000 + [0] * 10000 + [1] * 5000]
        )

        expected_auc = torch.tensor([1], dtype=torch.double)
        self.auc.update(**self.batches)
        actual_auc = self.auc.compute()["auc-DefaultTask|window_auc"]
        torch.allclose(expected_auc, actual_auc)

    def test_calc_auc_zero(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(10000)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor([[0] * 10000 + [1] * 10000])
        self.weights["DefaultTask"] = torch.Tensor(
            [[0] * 5000 + [1] * 10000 + [0] * 5000]
        )

        expected_auc = torch.tensor([0], dtype=torch.double)
        self.auc.update(**self.batches)
        actual_auc = self.auc.compute()["auc-DefaultTask|window_auc"]
        torch.allclose(expected_auc, actual_auc)

    def test_calc_auc_balanced(self) -> None:
        self.predictions["DefaultTask"] = torch.Tensor(
            [[0.0001 * x for x in range(10000)] * 2]
        )
        self.labels["DefaultTask"] = torch.Tensor([[0] * 10000 + [1] * 10000])
        self.weights["DefaultTask"] = torch.ones([1, 20000])

        expected_auc = torch.tensor([0.5], dtype=torch.double)
        self.auc.update(**self.batches)
        actual_auc = self.auc.compute()["auc-DefaultTask|window_auc"]
        torch.allclose(expected_auc, actual_auc)
