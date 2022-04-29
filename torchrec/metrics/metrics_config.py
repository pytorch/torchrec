#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from torchrec.metrics.metrics_namespace import StrValueMixin


class RecMetricEnumBase(StrValueMixin, Enum):
    pass


class RecMetricEnum(RecMetricEnumBase):
    NE = "ne"
    CTR = "ctr"
    AUC = "auc"
    CALIBRATION = "calibration"
    MSE = "mse"


@dataclass(unsafe_hash=True, eq=True)
class RecTaskInfo:
    name: str = "DefaultTask"
    label_name: str = "label"
    prediction_name: str = "prediction"
    weight_name: str = "weight"


class RecComputeMode(Enum):
    """This Enum lists the supported computation modes for RecMetrics.

    FUSED_TASKS_COMPUTATION indicates that RecMetrics will fuse the computation
    for multiple tasks of the same metric. This can be used by modules where the
    outputs of all the tasks are vectorized.
    """

    FUSED_TASKS_COMPUTATION = 1
    UNFUSED_TASKS_COMPUTATION = 2


_DEFAULT_WINDOW_SIZE = 10_000_000
_DEFAULT_QPS_WINDOW_SECONDS = 100


@dataclass
class RecMetricDef:
    """The dataclass that defines a RecMetric.

    Args:
        rec_tasks (List[RecTaskInfo]): this and next fields specify the RecTask
            information. ``rec_tasks`` specifies the RecTask information while
            ``rec_task_indices`` represents the indices that point to the
            RecTask information stored in the parent ``MetricsConfig``. Only one
            of the two fields should be specified.
        rec_task_indices (List[int]): see the doscstring of ``rec_tasks``.
        window_size (int): the window size for this metric.
        arguments (Optional[Dict[str, Any]]): any propritary arguments to be used
            by this Metric.
    """

    rec_tasks: List[RecTaskInfo] = field(default_factory=list)
    rec_task_indices: List[int] = field(default_factory=list)
    window_size: int = _DEFAULT_WINDOW_SIZE
    arguments: Optional[Dict[str, Any]] = None


class StateMetricEnum(StrValueMixin, Enum):
    OPTIMIZERS = "optimizers"
    MODEL_CONFIGURATOR = "model_configurator"


@dataclass
class QPSDef:
    window_size: int = _DEFAULT_QPS_WINDOW_SECONDS


@dataclass
class MetricsConfig:
    """The dataclass that lists all the configurations to be used by the
    MetricModule.

    Args:
        rec_tasks (List[RecTaskInfo]): the list of RecTasks that will be shared
            by all the metrics.
        rec_metrics (Dict[RecMetricEnum, RecMetricDef]): the confiurations of
            the RecMetric objects.
        qps_metric: (Optional[QPSDef]): the configurations of the QPSMetric
            object.
        rec_compute_mode (RecComputeMode): the computation mode for the
            RecMetric objects. This will be applied to all the RecMetric
            objects defined by ``rec_metrics``.
        fused_update_limit (int): the maximum updates that can be fused. The
            default is 0 which means no fusion. Setting this field to 1 is
            logically identical to 0.  If this field ii larger than 1,
            RecMetrics will perform the actual update every ``update()`` calls.
        state_metrics (List[StateMetricEnum]): indicates what state_metrics
            will be enabled.
        compute_on_all_ranks (bool): whether to compute rec metrics on all ranks.
            If False, only compute on rank 0.
    """

    rec_tasks: List[RecTaskInfo] = field(default_factory=list)
    rec_metrics: Dict[RecMetricEnum, RecMetricDef] = field(default_factory=dict)
    qps_metric: Optional[QPSDef] = None
    rec_compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION
    fused_update_limit: int = 0
    state_metrics: List[StateMetricEnum] = field(default_factory=list)
    # Computing metrics every step can be expsensive. This field is used to
    # specify the computation interval. MetricModule does not use this field but
    # the end users need to use this field to decide whether to call
    # ``compute()``.
    compute_interval_steps: int = 100
    compute_on_all_ranks: bool = False


DefaultTaskInfo = RecTaskInfo(
    name="DefaultTask",
    label_name="label",
    prediction_name="prediction",
    weight_name="weight",
)


DefaultMetricsConfig = MetricsConfig(
    rec_tasks=[DefaultTaskInfo],
    rec_metrics={
        RecMetricEnum.NE: RecMetricDef(
            rec_tasks=[DefaultTaskInfo], window_size=_DEFAULT_WINDOW_SIZE
        ),
    },
    qps_metric=QPSDef(),
    state_metrics=[],
)

# Explicitly specifying the empty fields to avoid any mistakes cased by simply
# relying on the Python default values, e.g., MetricConfig().
EmptyMetricsConfig = MetricsConfig(
    rec_tasks=[],
    rec_metrics={},
    qps_metric=None,
    state_metrics=[],
)
