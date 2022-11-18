#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import abc
import logging
import time
from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torchrec.metrics.auc import AUCMetric
from torchrec.metrics.calibration import CalibrationMetric
from torchrec.metrics.ctr import CTRMetric
from torchrec.metrics.metrics_config import (
    MetricsConfig,
    RecMetricEnum,
    RecMetricEnumBase,
    RecTaskInfo,
    StateMetricEnum,
)
from torchrec.metrics.metrics_namespace import (
    compose_customized_metric_key,
    compose_metric_namespace,
    MetricNamespace,
)
from torchrec.metrics.model_utils import parse_task_model_outputs
from torchrec.metrics.mse import MSEMetric
from torchrec.metrics.ne import NEMetric
from torchrec.metrics.rec_metric import RecMetric, RecMetricList
from torchrec.metrics.throughput import ThroughputMetric


logger: logging.Logger = logging.getLogger(__name__)

REC_METRICS_MAPPING: Dict[RecMetricEnumBase, Type[RecMetric]] = {
    RecMetricEnum.NE: NEMetric,
    RecMetricEnum.CTR: CTRMetric,
    RecMetricEnum.CALIBRATION: CalibrationMetric,
    RecMetricEnum.AUC: AUCMetric,
    RecMetricEnum.MSE: MSEMetric,
}


# Label used for emitting model metrics to the coresponding trainer publishers.
MODEL_METRIC_LABEL: str = "model"


MEMORY_AVG_WARNING_PERCENTAGE = 20
MEMORY_AVG_WARNING_WARMUP = 100

MetricValue = Union[torch.Tensor, float]


class StateMetric(abc.ABC):
    """
    The interface of state metrics for a component (e.g., optimizer, qat).
    """

    @abc.abstractmethod
    def get_metrics(self) -> Dict[str, MetricValue]:
        pass


class RecMetricModule(nn.Module):
    r"""
    For the current recommendation models, we assume there will be three
    types of metrics, 1.) RecMetric, 2.) Throughput, 3.) StateMetric.

    RecMetric is a metric that is computed from the model outputs (labels,
    predictions, weights).

    Throughput is being a standalone type as its unique characteristic, time-based.

    StateMetric is a metric that is computed based on a model componenet
    (e.g., Optimizer) internal logic.

    Args:
        batch_size (int): batch size used by this trainer.
        world_size (int): the number of trainers.
        rec_tasks (Optional[List[RecTaskInfo]]): the information of the model tasks.
        rec_metrics (Optional[RecMetricList]): the list of the RecMetrics.
        throughput_metric (Optional[ThroughputMetric]): the ThroughputMetric.
        state_metrics (Optional[Dict[str, StateMetric]]): the dict of StateMetrics.
        compute_interval_steps (int): the intervals between two compute calls in the unit of batch number
        memory_usage_limit_mb (float): the memory usage limit for OOM check

    Call Args:
        Not supported.

    Returns:
        Not supported.

    Example:
        >>> config = dataclasses.replace(
        >>>     DefaultMetricsConfig, state_metrics=[StateMetricEnum.OPTIMIZERS]
        >>> )
        >>>
        >>> metricModule = generate_metric_module(
        >>>     metric_class=RecMetricModule,
        >>>     metrics_config=config,
        >>>     batch_size=128,
        >>>     world_size=64,
        >>>     my_rank=0,
        >>>     state_metrics_mapping={StateMetricEnum.OPTIMIZERS: mock_optimizer},
        >>>     device=torch.device("cpu"),
        >>>     pg=dist.new_group([0]),
        >>> )
    """

    batch_size: int
    world_size: int
    rec_tasks: List[RecTaskInfo]
    rec_metrics: RecMetricList
    throughput_metric: Optional[ThroughputMetric]
    state_metrics: Dict[str, StateMetric]
    memory_usage_limit_mb: float
    memory_usage_mb_avg: float
    oom_count: int
    compute_count: int
    last_compute_time: float

    # TODO(chienchin): Reorganize the argument to directly accept a MetricsConfig.
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        rec_tasks: Optional[List[RecTaskInfo]] = None,
        rec_metrics: Optional[RecMetricList] = None,
        throughput_metric: Optional[ThroughputMetric] = None,
        state_metrics: Optional[Dict[str, StateMetric]] = None,
        compute_interval_steps: int = 100,
        min_compute_interval: float = 0.0,
        max_compute_interval: float = float("inf"),
        memory_usage_limit_mb: float = 512,
    ) -> None:
        super().__init__()
        self.rec_tasks = rec_tasks if rec_tasks else []
        self.rec_metrics = rec_metrics if rec_metrics else RecMetricList([])
        self.throughput_metric = throughput_metric
        self.state_metrics = state_metrics if state_metrics else {}
        self.trained_batches: int = 0
        self.batch_size = batch_size
        self.world_size = world_size
        self.memory_usage_limit_mb = memory_usage_limit_mb
        self.memory_usage_mb_avg = 0.0
        self.oom_count = 0
        self.compute_count = 0

        self.compute_interval_steps = compute_interval_steps
        self.min_compute_interval = min_compute_interval
        self.max_compute_interval = max_compute_interval
        if self.min_compute_interval == 0.0 and self.max_compute_interval == float(
            "inf"
        ):
            self.min_compute_interval = -1.0
            self.max_compute_interval = -1.0
        else:
            if self.max_compute_interval <= 0.0:
                raise ValueError("Max compute interval should not be smaller than 0.0.")
            if self.min_compute_interval < 0.0:
                raise ValueError("Min compute interval should not be smaller than 0.0.")
        self.register_buffer(
            "_compute_interval_steps",
            torch.zeros(1, dtype=torch.int32),
            persistent=False,
        )
        self.last_compute_time = -1.0

    def get_memory_usage(self) -> int:
        r"""Total memory of unique RecMetric tensors in bytes"""
        total = {}
        for metric in self.rec_metrics.rec_metrics:
            total.update(metric.get_memory_usage())
        return sum(total.values())

    def check_memory_usage(self, compute_count: int) -> None:
        memory_usage_mb = self.get_memory_usage() / (10**6)
        if memory_usage_mb > self.memory_usage_limit_mb:
            self.oom_count += 1
            logger.warning(
                f"MetricModule is using {memory_usage_mb}MB. "
                f"This is larger than the limit{self.memory_usage_limit_mb}MB. "
                f"This is the f{self.oom_count}th OOM."
            )

        if (
            compute_count > MEMORY_AVG_WARNING_WARMUP
            and memory_usage_mb
            > self.memory_usage_mb_avg * ((100 + MEMORY_AVG_WARNING_PERCENTAGE) / 100)
        ):
            logger.warning(
                f"MetricsModule is using more than {MEMORY_AVG_WARNING_PERCENTAGE}% of "
                f"the average memory usage. Current usage: {memory_usage_mb}MB."
            )

        self.memory_usage_mb_avg = (
            self.memory_usage_mb_avg * (compute_count - 1) + memory_usage_mb
        ) / compute_count

    def _update_rec_metrics(self, model_out: Dict[str, torch.Tensor]) -> None:
        r"""the internal update function to parse the model output.
        Override this function if the implementation cannot support
        the model output format.
        """
        if self.rec_metrics and self.rec_tasks:
            labels, predictions, weights = parse_task_model_outputs(
                self.rec_tasks, model_out
            )
            self.rec_metrics.update(
                predictions=predictions, labels=labels, weights=weights
            )

    def update(self, model_out: Dict[str, torch.Tensor]) -> None:
        r"""update() is called per batch, usually right after forward() to
        update the local states of metrics based on the model_output.

        Throughput.update() is also called due to the implementation sliding window
        throughput.
        """
        self._update_rec_metrics(model_out)
        if self.throughput_metric:
            self.throughput_metric.update()
        self.trained_batches += 1

    def _adjust_compute_interval(self) -> None:
        """
        Adjust the compute interval (in batches) based on the first two time
        elapsed between the first two compute().
        """
        if self.last_compute_time > 0 and self.min_compute_interval >= 0:
            now = time.time()
            interval = now - self.last_compute_time
            if not (self.max_compute_interval >= interval >= self.min_compute_interval):
                per_step_time = interval / self.compute_interval_steps

                assert (
                    self.max_compute_interval != float("inf")
                    or self.min_compute_interval != 0.0
                ), (
                    "The compute time interval is "
                    f"[{self.max_compute_interval}, {self.min_compute_interval}]. "
                    "Something is not correct of this range. __init__() should have "
                    "captured this earlier."
                )
                if self.max_compute_interval == float("inf"):
                    # The `per_step_time` is not perfectly measured -- each
                    # step training time can vary. Since max_compute_interval
                    # is set to infinite, adding 1.0 to the `min_compute_interval`
                    # increase the chance that the final compute interval is
                    # indeed larger than `min_compute_interval`.
                    self._compute_interval_steps[0] = int(
                        (self.min_compute_interval + 1.0) / per_step_time
                    )
                elif self.min_compute_interval == 0.0:
                    # Similar to the above if, subtracting 1.0 from
                    # `max_compute_interval` to compute `_compute_interval_steps`
                    # can increase the chance that the final compute interval
                    # is indeed smaller than `max_compute_interval`
                    offset = 0.0 if self.max_compute_interval <= 1.0 else 1.0
                    self._compute_interval_steps[0] = int(
                        (self.max_compute_interval - offset) / per_step_time
                    )
                else:
                    self._compute_interval_steps[0] = int(
                        (self.max_compute_interval + self.min_compute_interval)
                        / 2
                        / per_step_time
                    )
                dist.all_reduce(self._compute_interval_steps, op=dist.ReduceOp.MAX)
            self.compute_interval_steps = int(self._compute_interval_steps.item())
            self.min_compute_interval = -1.0
            self.max_compute_interval = -1.0
        self.last_compute_time = time.time()

    def should_compute(self) -> bool:
        return self.trained_batches % self.compute_interval_steps == 0

    def compute(self) -> Dict[str, MetricValue]:
        r"""compute() is called when the global metrics are required, usually
        right before logging the metrics results to the data sink.
        """
        self.compute_count += 1
        self.check_memory_usage(self.compute_count)

        ret: Dict[str, MetricValue] = {}
        if self.rec_metrics:
            self._adjust_compute_interval()
            ret.update(self.rec_metrics.compute())
        if self.throughput_metric:
            ret.update(self.throughput_metric.compute())
        if self.state_metrics:
            for namespace, component in self.state_metrics.items():
                ret.update(
                    {
                        f"{compose_customized_metric_key(namespace, metric_name)}": metric_value
                        for metric_name, metric_value in component.get_metrics().items()
                    }
                )
        return ret

    def local_compute(self) -> Dict[str, MetricValue]:
        r"""local_compute() is called when per-trainer metrics are required. It's
        can be used for debugging. Currently only rec_metrics is supported.
        """
        ret: Dict[str, MetricValue] = {}
        if self.rec_metrics:
            ret.update(self.rec_metrics.local_compute())
        return ret

    def sync(self) -> None:
        self.rec_metrics.sync()

    def unsync(self) -> None:
        self.rec_metrics.unsync()

    def reset(self) -> None:
        self.rec_metrics.reset()

    def get_required_inputs(self) -> List[str]:
        return self.rec_metrics.get_required_inputs()


def _generate_rec_metrics(
    metrics_config: MetricsConfig,
    world_size: int,
    my_rank: int,
    batch_size: int,
    process_group: Optional[dist.ProcessGroup] = None,
) -> RecMetricList:
    rec_metrics = []
    for metric_enum, metric_def in metrics_config.rec_metrics.items():

        kwargs: Dict[str, Any] = {}
        if metric_def and metric_def.arguments is not None:
            kwargs = metric_def.arguments

        rec_tasks: List[RecTaskInfo] = []
        if metric_def.rec_tasks and metric_def.rec_task_indices:
            raise ValueError(
                "Only one of RecMetricDef.rec_tasks and RecMetricDef.rec_task_indices "
                "should be specified."
            )
        if metric_def.rec_tasks:
            rec_tasks = metric_def.rec_tasks
        elif metric_def.rec_task_indices:
            rec_tasks = [
                metrics_config.rec_tasks[idx] for idx in metric_def.rec_task_indices
            ]
        else:
            raise ValueError(
                "One of RecMetricDef.rec_tasks and RecMetricDef.rec_task_indices "
                "should be a non-empty list"
            )

        rec_metrics.append(
            REC_METRICS_MAPPING[metric_enum](
                world_size=world_size,
                my_rank=my_rank,
                batch_size=batch_size,
                tasks=rec_tasks,
                compute_mode=metrics_config.rec_compute_mode,
                window_size=metric_def.window_size,
                fused_update_limit=metrics_config.fused_update_limit,
                compute_on_all_ranks=metrics_config.compute_on_all_ranks,
                should_validate_update=metrics_config.should_validate_update,
                process_group=process_group,
                **kwargs,
            )
        )
    return RecMetricList(rec_metrics)


STATE_METRICS_NAMESPACE_MAPPING: Dict[StateMetricEnum, MetricNamespace] = {
    StateMetricEnum.OPTIMIZERS: MetricNamespace.OPTIMIZERS,
    StateMetricEnum.MODEL_CONFIGURATOR: MetricNamespace.MODEL_CONFIGURATOR,
}


def _generate_state_metrics(
    metrics_config: MetricsConfig,
    state_metrics_mapping: Dict[StateMetricEnum, StateMetric],
) -> Dict[str, StateMetric]:
    state_metrics: Dict[str, StateMetric] = {}
    for metric_enum in metrics_config.state_metrics:
        metric_namespace: Optional[
            MetricNamespace
        ] = STATE_METRICS_NAMESPACE_MAPPING.get(metric_enum, None)
        if metric_namespace is None:
            raise ValueError(f"Unknown StateMetrics {metric_enum}")
        full_namespace = compose_metric_namespace(
            metric_namespace, str(metric_namespace)
        )
        state_metrics[full_namespace] = state_metrics_mapping[metric_enum]
    return state_metrics


def generate_metric_module(
    metric_class: Type[RecMetricModule],
    metrics_config: MetricsConfig,
    batch_size: int,
    world_size: int,
    my_rank: int,
    state_metrics_mapping: Dict[StateMetricEnum, StateMetric],
    device: torch.device,
    process_group: Optional[dist.ProcessGroup] = None,
) -> RecMetricModule:
    rec_metrics = _generate_rec_metrics(
        metrics_config, world_size, my_rank, batch_size, process_group
    )
    if metrics_config.throughput_metric:
        throughput_metric = ThroughputMetric(
            batch_size=batch_size,
            world_size=world_size,
            window_seconds=metrics_config.throughput_metric.window_size,
        )
    else:
        throughput_metric = None
    state_metrics = _generate_state_metrics(metrics_config, state_metrics_mapping)
    metrics = metric_class(
        batch_size=batch_size,
        world_size=world_size,
        rec_tasks=metrics_config.rec_tasks,
        rec_metrics=rec_metrics,
        throughput_metric=throughput_metric,
        state_metrics=state_metrics,
        compute_interval_steps=metrics_config.compute_interval_steps,
        min_compute_interval=metrics_config.min_compute_interval,
        max_compute_interval=metrics_config.max_compute_interval,
    )
    metrics.to(device)
    return metrics
