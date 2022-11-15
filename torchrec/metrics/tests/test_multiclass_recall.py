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
from torchrec.metrics.multiclass_recall import (
    compute_multiclass_recall_at_k,
    get_multiclass_recall_states,
    MulticlassRecallMetric,
)
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric
from torchrec.metrics.test_utils import (
    rec_metric_value_test_helper,
    rec_metric_value_test_launcher,
    TestMetric,
)

N_CLASSES = 4
WORLD_SIZE = 4


class TestMulticlassRecallMetric(TestMetric):
    n_classes: int = N_CLASSES

    @staticmethod
    def _get_states(
        labels: torch.Tensor,
        predictions: torch.Tensor,
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        states = get_multiclass_recall_states(
            predictions, labels, weights, TestMulticlassRecallMetric.n_classes
        )
        return states

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_multiclass_recall_at_k(
            states["tp_at_k"],
            states["total_weights"],
        )


class MulticlassRecallMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = MulticlassRecallMetric
    target_compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION
    task_name: str = "multiclass_recall"

    @staticmethod
    def _test_multiclass_recall(
        target_clazz: Type[RecMetric],
        target_compute_mode: RecComputeMode,
        task_names: List[str],
        fused_update_limit: int = 0,
        compute_on_all_ranks: bool = False,
        should_validate_update: bool = False,
        batch_window_size: int = 5,
        n_classes: int = N_CLASSES,
    ) -> None:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend="gloo",
            world_size=world_size,
            rank=rank,
        )

        multiclass_recall_metrics, test_metrics = rec_metric_value_test_helper(
            target_clazz=target_clazz,
            target_compute_mode=target_compute_mode,
            test_clazz=TestMulticlassRecallMetric,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=False,
            should_validate_update=should_validate_update,
            world_size=world_size,
            my_rank=rank,
            task_names=task_names,
            batch_window_size=batch_window_size,
            n_classes=n_classes,
        )

        if rank == 0:
            for name in task_names:
                assert torch.allclose(
                    multiclass_recall_metrics[
                        f"multiclass_recall-{name}|lifetime_multiclass_recall"
                    ],
                    test_metrics[0][name],
                )
                assert torch.allclose(
                    multiclass_recall_metrics[
                        f"multiclass_recall-{name}|window_multiclass_recall"
                    ],
                    test_metrics[1][name],
                )
                assert torch.allclose(
                    multiclass_recall_metrics[
                        f"multiclass_recall-{name}|local_lifetime_multiclass_recall"
                    ],
                    test_metrics[2][name],
                )
                assert torch.allclose(
                    multiclass_recall_metrics[
                        f"multiclass_recall-{name}|local_window_multiclass_recall"
                    ],
                    test_metrics[3][name],
                )
        dist.destroy_process_group()

    _test_multiclass_recall_large_window_size: Callable[..., None] = partial(
        # pyre-fixme[16]: `Callable` has no attribute `__func__`.
        _test_multiclass_recall.__func__,
        batch_window_size=10,
    )
    update_wrapper(
        _test_multiclass_recall_large_window_size, _test_multiclass_recall.__func__
    )

    def test_multiclass_recall_unfused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MulticlassRecallMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestMulticlassRecallMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_multiclass_recall,
            n_classes=N_CLASSES,
        )

    def test_multiclass_recall_fused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MulticlassRecallMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestMulticlassRecallMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_multiclass_recall,
            n_classes=N_CLASSES,
        )

    def test_multiclass_recall_update_fused(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=MulticlassRecallMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestMulticlassRecallMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=5,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_multiclass_recall,
            n_classes=N_CLASSES,
        )

        rec_metric_value_test_launcher(
            target_clazz=MulticlassRecallMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestMulticlassRecallMetric,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=100,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_multiclass_recall_large_window_size,
            n_classes=N_CLASSES,
        )
