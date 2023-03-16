#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from functools import partial, update_wrapper
from typing import Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric, RecTaskInfo
from torchrec.metrics.test_utils import (
    gen_test_batch,
    gen_test_tasks,
    metric_test_helper,
    rec_metric_value_test_launcher,
    TestMetric,
)
from torchrec.metrics.tower_qps import TowerQPSMetric

WORLD_SIZE = 4
WARMUP_STEPS = 100
DURING_WARMUP_NSTEPS = 10
AFTER_WARMUP_NSTEPS = 120


TestRecMetricOutput = Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]


class TestTowerQPSMetric(TestMetric):
    def __init__(
        self,
        world_size: int,
        rec_tasks: List[RecTaskInfo],
    ) -> None:
        super().__init__(world_size, rec_tasks)

    # The abstract _get_states method in TestMetric has to be overwritten
    # For tower qps the time_lapse state is not generated from labels, predictions
    # or weights
    @staticmethod
    def _get_states(
        labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {}

    @staticmethod
    def _reduce(states: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        reduced_states: Dict[str, torch.Tensor] = {}
        # Need to check if states is empty, because we only update the states after warmup
        if states:
            reduced_states["num_samples"] = torch.sum(
                torch.stack(states["num_samples"]), dim=0
            )
            reduced_states["time_lapse"] = torch.max(
                torch.stack(states["time_lapse"]), dim=0
            ).values
        return reduced_states

    @staticmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "num_samples" not in states or "time_lapse" not in states:
            # This is to match the default 0.0 output from TowerQPSMetric if warmup is not done
            return torch.tensor(float("0.0"), dtype=torch.double)

        return torch.where(
            states["time_lapse"] <= 0.0,
            0.0,
            states["num_samples"] / states["time_lapse"],
        ).double()

    def compute(
        self,
        model_outs: List[Dict[str, torch.Tensor]],
        nsteps: int,
        batch_window_size: int,
        timestamps: Optional[List[float]],
    ) -> TestRecMetricOutput:
        assert timestamps is not None
        lifetime_states, window_states, local_lifetime_states, local_window_states = (
            {task_info.name: {} for task_info in self._rec_tasks} for _ in range(4)
        )
        for i in range(WARMUP_STEPS, nsteps):
            for task_info in self._rec_tasks:
                local_states = {
                    "num_samples": torch.tensor(
                        model_outs[i][task_info.label_name].shape[-1],
                        dtype=torch.long,
                    ),
                    "time_lapse": torch.tensor(
                        timestamps[i] - timestamps[i - 1], dtype=torch.double
                    ),
                }
                self._aggregate(local_lifetime_states[task_info.name], local_states)
                if nsteps - batch_window_size <= i:
                    self._aggregate(local_window_states[task_info.name], local_states)

        for task_info in self._rec_tasks:
            aggregated_lifetime_state = {}
            for k, v in local_lifetime_states[task_info.name].items():
                aggregated_lifetime_state[k] = [
                    torch.zeros_like(v) for _ in range(self.world_size)
                ]
                dist.all_gather(aggregated_lifetime_state[k], v)
            lifetime_states[task_info.name] = self._reduce(aggregated_lifetime_state)

            aggregated_window_state = {}
            for k, v in local_window_states[task_info.name].items():
                aggregated_window_state[k] = [
                    torch.zeros_like(v) for _ in range(self.world_size)
                ]
                dist.all_gather(aggregated_window_state[k], v)
            window_states[task_info.name] = self._reduce(aggregated_window_state)

        lifetime_metrics = {}
        window_metrics = {}
        local_lifetime_metrics = {}
        local_window_metrics = {}
        for task_info in self._rec_tasks:
            lifetime_metrics[task_info.name] = self._compute(
                lifetime_states[task_info.name]
            )
            window_metrics[task_info.name] = self._compute(
                window_states[task_info.name]
            )
            local_lifetime_metrics[task_info.name] = self._compute(
                local_lifetime_states[task_info.name]
            )
            local_window_metrics[task_info.name] = self._compute(
                local_window_states[task_info.name]
            )
        return (
            lifetime_metrics,
            window_metrics,
            local_lifetime_metrics,
            local_window_metrics,
        )


class TowerQPSMetricTest(unittest.TestCase):
    target_clazz: Type[RecMetric] = TowerQPSMetric
    task_names: str = "qps"

    _test_tower_qps: Callable[..., None] = partial(
        metric_test_helper,
        is_time_dependent=True,
        time_dependent_metric={TowerQPSMetric: "torchrec.metrics.tower_qps"},
    )
    update_wrapper(_test_tower_qps, metric_test_helper)

    def test_unfused_tower_qps_during_warmup(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=TowerQPSMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestTowerQPSMetric,
            metric_name=TowerQPSMetricTest.task_names,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_tower_qps,
        )

    def test_unfused_tower_qps(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=TowerQPSMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestTowerQPSMetric,
            metric_name=TowerQPSMetricTest.task_names,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_tower_qps,
            test_nsteps=DURING_WARMUP_NSTEPS,
        )

    def test_fused_tower_qps(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=TowerQPSMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestTowerQPSMetric,
            metric_name=TowerQPSMetricTest.task_names,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=False,
            world_size=WORLD_SIZE,
            entry_point=self._test_tower_qps,
            test_nsteps=AFTER_WARMUP_NSTEPS,
        )

    def test_unfused_check_update_tower_qps(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=TowerQPSMetric,
            target_compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            test_clazz=TestTowerQPSMetric,
            metric_name=TowerQPSMetricTest.task_names,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=True,
            world_size=WORLD_SIZE,
            entry_point=self._test_tower_qps,
            test_nsteps=AFTER_WARMUP_NSTEPS,
        )

    def test_fused_check_update_tower_qps(self) -> None:
        rec_metric_value_test_launcher(
            target_clazz=TowerQPSMetric,
            target_compute_mode=RecComputeMode.FUSED_TASKS_COMPUTATION,
            test_clazz=TestTowerQPSMetric,
            metric_name=TowerQPSMetricTest.task_names,
            task_names=["t1", "t2", "t3"],
            fused_update_limit=0,
            compute_on_all_ranks=False,
            should_validate_update=True,
            world_size=WORLD_SIZE,
            entry_point=self._test_tower_qps,
            test_nsteps=AFTER_WARMUP_NSTEPS,
        )

    def test_warmup_checkpointing(self) -> None:
        warmup_steps = 5
        extra_steps = 2
        batch_size = 128
        qps = TowerQPSMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=[DefaultTaskInfo],
            warmup_steps=warmup_steps,
            compute_on_all_ranks=False,
            should_validate_update=False,
        )
        model_output = gen_test_batch(batch_size)
        for i in range(5):
            for _ in range(warmup_steps + extra_steps):
                qps.update(
                    predictions={"DefaultTask": model_output["prediction"]},
                    labels={"DefaultTask": model_output["label"]},
                    weights={"DefaultTask": model_output["weight"]},
                )
            self.assertEquals(
                qps._metrics_computations[0].warmup_examples,
                batch_size * warmup_steps * (i + 1),
            )
            self.assertEquals(
                qps._metrics_computations[0].num_examples,
                batch_size * (warmup_steps + extra_steps) * (i + 1),
            )
            # Mimic trainer crashing and loading a checkpoint.
            qps._metrics_computations[0]._steps = 0

    def test_mtml_empty_update(self) -> None:
        warmup_steps = 2
        extra_steps = 2
        batch_size = 128
        task_names = ["t1", "t2"]
        tasks = gen_test_tasks(task_names)
        qps = TowerQPSMetric(
            world_size=1,
            my_rank=0,
            batch_size=batch_size,
            tasks=tasks,
            warmup_steps=warmup_steps,
            compute_on_all_ranks=False,
            should_validate_update=False,
        )
        for step in range(warmup_steps + extra_steps):
            _model_output = [
                gen_test_batch(
                    label_name=task.label_name,
                    prediction_name=task.prediction_name,
                    weight_name=task.weight_name,
                    batch_size=batch_size,
                )
                for task in tasks
            ]
            model_output = {k: v for d in _model_output for k, v in d.items()}
            labels, predictions, weights, _ = parse_task_model_outputs(
                tasks, model_output
            )
            if step % 2 == 0:
                del labels["t1"]
            else:
                del labels["t2"]
            qps.update(predictions=predictions, labels=labels, weights=weights)
            self.assertEquals(
                qps._metrics_computations[0].num_examples, (step + 1) // 2 * batch_size
            )
            self.assertEquals(
                qps._metrics_computations[1].num_examples, (step + 2) // 2 * batch_size
            )
