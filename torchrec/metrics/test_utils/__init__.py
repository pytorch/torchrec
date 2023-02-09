#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os
import random
import tempfile
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.rec_metric import RecComputeMode, RecMetric, RecTaskInfo

TestRecMetricOutput = Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]


def gen_test_batch(
    batch_size: int,
    label_name: str = "label",
    prediction_name: str = "prediction",
    weight_name: str = "weight",
    tensor_name: str = "tensor",
    mask_tensor_name: Optional[str] = None,
    label_value: Optional[torch.Tensor] = None,
    prediction_value: Optional[torch.Tensor] = None,
    weight_value: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    n_classes: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if label_value is not None:
        label = label_value
    else:
        label = torch.randint(0, n_classes or 2, (batch_size,)).double()
    if prediction_value is not None:
        prediction = prediction_value
    else:
        prediction = (
            torch.rand(batch_size, dtype=torch.double)
            if n_classes is None
            else torch.rand(batch_size, n_classes, dtype=torch.double)
        )
    if weight_value is not None:
        weight = weight_value
    else:
        weight = torch.rand(batch_size, dtype=torch.double)
    test_batch = {
        label_name: label,
        prediction_name: prediction,
        weight_name: weight,
        tensor_name: torch.rand(batch_size, dtype=torch.double),
    }
    if mask_tensor_name is not None:
        if mask is None:
            mask = torch.ones(batch_size, dtype=torch.double)
        test_batch[mask_tensor_name] = mask

    return test_batch


def gen_test_tasks(
    task_names: List[str],
) -> List[RecTaskInfo]:
    return [
        RecTaskInfo(
            name=task_name,
            label_name=f"{task_name}-label",
            prediction_name=f"{task_name}-prediction",
            weight_name=f"{task_name}-weight",
        )
        for task_name in task_names
    ]


def gen_test_timestamps(
    nsteps: int,
) -> List[float]:
    timestamps = [0.0 for _ in range(nsteps)]
    for step in range(1, nsteps):
        time_lapse = random.uniform(1.0, 5.0)
        timestamps[step] = timestamps[step - 1] + time_lapse
    return timestamps


class TestMetric(abc.ABC):
    def __init__(
        self,
        world_size: int,
        rec_tasks: List[RecTaskInfo],
        compute_lifetime_metric: bool = True,
        compute_window_metric: bool = True,
        local_compute_lifetime_metric: bool = True,
        local_compute_window_metric: bool = True,
    ) -> None:
        self.world_size = world_size
        self._rec_tasks = rec_tasks
        self._compute_lifetime_metric = compute_lifetime_metric
        self._compute_window_metric = compute_window_metric
        self._local_compute_lifetime_metric = local_compute_lifetime_metric
        self._local_compute_window_metric = local_compute_window_metric

    @staticmethod
    def _aggregate(
        states: Dict[str, torch.Tensor], new_states: Dict[str, torch.Tensor]
    ) -> None:
        for k, v in new_states.items():
            if k not in states:
                states[k] = torch.zeros_like(v)
            states[k] += v

    @staticmethod
    @abc.abstractmethod
    def _get_states(
        labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        pass

    @staticmethod
    @abc.abstractmethod
    def _compute(states: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def compute(
        self,
        model_outs: List[Dict[str, torch.Tensor]],
        nsteps: int,
        batch_window_size: int,
        timestamps: Optional[List[float]],
    ) -> TestRecMetricOutput:
        aggregated_model_out = {}
        lifetime_states, window_states, local_lifetime_states, local_window_states = (
            {task_info.name: {} for task_info in self._rec_tasks} for _ in range(4)
        )
        for i in range(nsteps):
            for k, v in model_outs[i].items():
                aggregated_list = [torch.zeros_like(v) for _ in range(self.world_size)]
                dist.all_gather(aggregated_list, v)
                aggregated_model_out[k] = torch.cat(aggregated_list)
            for task_info in self._rec_tasks:
                states = self._get_states(
                    aggregated_model_out[task_info.label_name],
                    aggregated_model_out[task_info.prediction_name],
                    aggregated_model_out[task_info.weight_name],
                )
                if self._compute_lifetime_metric:
                    self._aggregate(lifetime_states[task_info.name], states)
                if self._compute_window_metric and nsteps - batch_window_size <= i:
                    self._aggregate(window_states[task_info.name], states)
                local_states = self._get_states(
                    model_outs[i][task_info.label_name],
                    model_outs[i][task_info.prediction_name],
                    model_outs[i][task_info.weight_name],
                )
                if self._local_compute_lifetime_metric:
                    self._aggregate(local_lifetime_states[task_info.name], local_states)
                if (
                    self._local_compute_window_metric
                    and nsteps - batch_window_size <= i
                ):
                    self._aggregate(local_window_states[task_info.name], local_states)
        lifetime_metrics = {}
        window_metrics = {}
        local_lifetime_metrics = {}
        local_window_metrics = {}
        for task_info in self._rec_tasks:
            lifetime_metrics[task_info.name] = (
                self._compute(lifetime_states[task_info.name])
                if self._compute_lifetime_metric
                else torch.tensor(0.0)
            )
            window_metrics[task_info.name] = (
                self._compute(window_states[task_info.name])
                if self._compute_window_metric
                else torch.tensor(0.0)
            )
            local_lifetime_metrics[task_info.name] = (
                self._compute(local_lifetime_states[task_info.name])
                if self._local_compute_lifetime_metric
                else torch.tensor(0.0)
            )
            local_window_metrics[task_info.name] = (
                self._compute(local_window_states[task_info.name])
                if self._local_compute_window_metric
                else torch.tensor(0.0)
            )
        return (
            lifetime_metrics,
            window_metrics,
            local_lifetime_metrics,
            local_window_metrics,
        )


BATCH_SIZE = 32
BATCH_WINDOW_SIZE = 5
NSTEPS = 10


def rec_metric_value_test_helper(
    target_clazz: Type[RecMetric],
    target_compute_mode: RecComputeMode,
    test_clazz: Optional[Type[TestMetric]],
    fused_update_limit: int,
    compute_on_all_ranks: bool,
    should_validate_update: bool,
    world_size: int,
    my_rank: int,
    task_names: List[str],
    batch_size: int = BATCH_SIZE,
    nsteps: int = NSTEPS,
    batch_window_size: int = BATCH_WINDOW_SIZE,
    is_time_dependent: bool = False,
    time_dependent_metric: Optional[Dict[Type[RecMetric], str]] = None,
    n_classes: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], ...]]:
    tasks = gen_test_tasks(task_names)
    model_outs = []
    for _ in range(nsteps):
        _model_outs = [
            gen_test_batch(
                label_name=task.label_name,
                prediction_name=task.prediction_name,
                weight_name=task.weight_name,
                batch_size=batch_size,
                n_classes=n_classes,
            )
            for task in tasks
        ]
        model_outs.append({k: v for d in _model_outs for k, v in d.items()})

    def get_target_rec_metric_value(
        model_outs: List[Dict[str, torch.Tensor]],
        tasks: List[RecTaskInfo],
        timestamps: Optional[List[float]] = None,
        time_mock: Optional[Mock] = None,
    ) -> Dict[str, torch.Tensor]:
        window_size = world_size * batch_size * batch_window_size
        kwargs: Dict[str, Any] = {}
        if n_classes:
            kwargs["number_of_classes"] = n_classes
        target_metric_obj = target_clazz(
            world_size=world_size,
            my_rank=my_rank,
            batch_size=batch_size,
            tasks=tasks,
            compute_mode=target_compute_mode,
            window_size=window_size,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=compute_on_all_ranks,
            should_validate_update=should_validate_update,
            **kwargs,
        )
        for i in range(nsteps):
            labels, predictions, weights, _ = parse_task_model_outputs(
                tasks, model_outs[i]
            )
            if target_compute_mode == RecComputeMode.FUSED_TASKS_COMPUTATION:
                labels = torch.stack(list(labels.values()))
                predictions = torch.stack(list(predictions.values()))
                weights = torch.stack(list(weights.values()))

            if timestamps is not None:
                time_mock.return_value = timestamps[i]
            target_metric_obj.update(
                predictions=predictions, labels=labels, weights=weights
            )
        result_metrics = target_metric_obj.compute()
        result_metrics.update(target_metric_obj.local_compute())
        return result_metrics

    def get_test_rec_metric_value(
        model_outs: List[Dict[str, torch.Tensor]],
        tasks: List[RecTaskInfo],
        timestamps: Optional[List[float]] = None,
    ) -> TestRecMetricOutput:
        test_metrics: TestRecMetricOutput = ({}, {}, {}, {})
        if test_clazz is not None:
            # pyre-ignore[45]: Cannot instantiate abstract class `TestMetric`.
            test_metric_obj = test_clazz(world_size, tasks)
            test_metrics = test_metric_obj.compute(
                model_outs, nsteps, batch_window_size, timestamps
            )
        return test_metrics

    if is_time_dependent:
        timestamps: Optional[List[float]] = (
            gen_test_timestamps(nsteps) if is_time_dependent else None
        )
        assert time_dependent_metric is not None  # avoid typing issue
        time_dependent_target_clazz_path = time_dependent_metric[target_clazz]
        with patch(time_dependent_target_clazz_path + ".time.monotonic") as time_mock:
            result_metrics = get_target_rec_metric_value(
                model_outs, tasks, timestamps, time_mock
            )
        test_metrics = get_test_rec_metric_value(model_outs, tasks, timestamps)
    else:
        result_metrics = get_target_rec_metric_value(model_outs, tasks)
        test_metrics = get_test_rec_metric_value(model_outs, tasks)

    return result_metrics, test_metrics


def get_launch_config(world_size: int, rdzv_endpoint: str) -> pet.LaunchConfig:
    return pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=world_size,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint=rdzv_endpoint,
        rdzv_configs={"store_type": "file"},
        start_method="spawn",
        monitor_interval=1,
        max_restarts=0,
    )


def rec_metric_value_test_launcher(
    target_clazz: Type[RecMetric],
    target_compute_mode: RecComputeMode,
    test_clazz: Type[TestMetric],
    task_names: List[str],
    fused_update_limit: int,
    compute_on_all_ranks: bool,
    should_validate_update: bool,
    world_size: int,
    entry_point: Callable[..., None],
    test_nsteps: int = 1,
    n_classes: Optional[int] = None,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lc = get_launch_config(
            world_size=world_size, rdzv_endpoint=os.path.join(tmpdir, "rdzv")
        )

        # Call the same helper as the actual test to make code coverage visible to
        # the testing system.
        rec_metric_value_test_helper(
            target_clazz,
            target_compute_mode,
            test_clazz=None,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=compute_on_all_ranks,
            should_validate_update=should_validate_update,
            world_size=1,
            my_rank=0,
            task_names=task_names,
            batch_size=32,
            nsteps=test_nsteps,
            batch_window_size=1,
            n_classes=n_classes,
        )
        pet.elastic_launch(lc, entrypoint=entry_point)(
            target_clazz,
            target_compute_mode,
            task_names,
            fused_update_limit,
            compute_on_all_ranks,
            should_validate_update,
        )


def rec_metric_accuracy_test_helper(
    world_size: int, entry_point: Callable[..., None]
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        lc = get_launch_config(
            world_size=world_size, rdzv_endpoint=os.path.join(tmpdir, "rdzv")
        )
        pet.elastic_launch(lc, entrypoint=entry_point)()
