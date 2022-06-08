#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from functools import partial, update_wrapper
from typing import Callable, Dict, List, Type

import torch
import torch.distributed as dist
from torchrec.metrics.ne import compute_cross_entropy, compute_ne, NEMetric
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.tests.test_utils import (
    rec_metric_value_test_helper,
    rec_metric_value_test_launcher,
    TestMetric,
)


WORLD_SIZE = 4


class TestNEMetric(TestMetric):
    eta: float = 1e-12

    @staticmethod
    def _get_states(
        labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        cross_entropy = compute_cross_entropy(
            labels, predictions, weights, TestNEMetric.eta
        )
        cross_entropy_sum = torch.sum(cross_entropy)
        weighted_num_samples = torch.sum(weights)
        pos_labels = torch.sum(weights * labels)
        neg_labels = torch.sum(weights * (1.0 - labels))
        return {
            "cross_entropy_sum": cross_entropy_sum,
            "weighted_num_samples": weighted_num_samples,
            "pos_labels": pos_labels,
            "neg_labels": neg_labels,
            "num_samples": torch.tensor(labels.size()).long(),
        }

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_ne(
            states["cross_entropy_sum"],
            states["weighted_num_samples"],
            pos_labels=states["pos_labels"],
            neg_labels=states["neg_labels"],
            eta=TestNEMetric.eta,
        )


class NEMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = NEMetric
    target_compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION
    task_name: str = "ne"

    @staticmethod
    def _test_ne(
        target_clazz: Type[RecMetric],
        target_compute_mode: RecComputeMode,
        task_names: List[str],
        fused_update_limit: int = 0,
        compute_on_all_ranks: bool = False,
        should_validate_update: bool = False,
        batch_window_size: int = 5,
    ) -> None:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend="gloo",
            world_size=world_size,
            rank=rank,
        )

        ne_metrics, test_metrics = rec_metric_value_test_helper(
            target_clazz=target_clazz,
            target_compute_mode=target_compute_mode,
            test_clazz=TestNEMetric,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=False,
            should_validate_update=should_validate_update,
            world_size=world_size,
            my_rank=rank,
            task_names=task_names,
            batch_window_size=batch_window_size,
        )

        if rank == 0:
            for name in task_names:
                assert torch.allclose(
                    ne_metrics[f"ne-{name}|lifetime_ne"], test_metrics[0][name]
                )
                assert torch.allclose(
                    ne_metrics[f"ne-{name}|window_ne"], test_metrics[1][name]
                )
                assert torch.allclose(
                    ne_metrics[f"ne-{name}|local_lifetime_ne"], test_metrics[2][name]
                )
                assert torch.allclose(
                    ne_metrics[f"ne-{name}|local_window_ne"], test_metrics[3][name]
                )
        dist.destroy_process_group()

    _test_ne_large_window_size: Callable[..., None] = partial(
        # pyre-fixme[16]: `Callable` has no attribute `__func__`.
        _test_ne.__func__,
        batch_window_size=10,
    )
    update_wrapper(_test_ne_large_window_size, _test_ne.__func__)

    def test_ne_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_ne,
        )

    def test_ne_fused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_ne,
        )

    def test_ne_update_fused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=5,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_ne,
        )

        rec_metric_value_test_launcher(
            target_clazz=NEMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestNEMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=100,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_ne_large_window_size,
        )

        # TODO(stellaya): support the usage of fused_tasks_computation and
        # fused_update for the same RecMetric
        # rec_metric_value_test_launcher(
        #     target_clazz=NEMetric,
        #     target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
        #     test_clazz=TestNEMetric,
        #     task_names=["t1", "t2", "t3"],
        #     fused_update_limit=5,
        #     compute_on_all_ranks=False,
        #     should_validate_update=False,
        #     world_size=WORLD_SIZE,
        #     entry_point=self._test_ne,
        # )

        # rec_metric_value_test_launcher(
        #     target_clazz=NEMetric,
        #     target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
        #     test_clazz=TestNEMetric,
        #     task_names=["t1", "t2", "t3"],
        #     fused_update_limit=100,
        #     compute_on_all_ranks=False,
        #     should_validate_update=False,
        #     world_size=WORLD_SIZE,
        #     entry_point=self._test_ne_large_window_size,
        # )
