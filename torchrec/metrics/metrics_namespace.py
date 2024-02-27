#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The namespace definition and genration APIs for rec metrics.

The key is defined as the following format:
    f"{namespace}-{task_name}|{metric_key}"
    1. namespace should be one of MetricNamespace.
    2. task_name is user-defined value. If there is no task_name, it should be
       the same as namespace.
    3. metric_key can be:
        a.) One of MetricName.
        b.) One of MetricPrefix + One of MetricName.
        c.) Any string. This arbitrary case should happen only for reporting the
            internal state metrics.
"""

from enum import Enum
from typing import Optional


class StrValueMixin:
    def __str__(self) -> str:
        # pyre-fixme[16]: `StrValueMixin` has no attribute `value`.
        return self.value


class MetricNameBase(StrValueMixin, Enum):
    pass


class MetricName(MetricNameBase):
    DEFAULT = ""

    NE = "ne"
    SEGMENTED_NE = "segmented_ne"
    LOG_LOSS = "logloss"
    THROUGHPUT = "throughput"
    TOTAL_EXAMPLES = "total_examples"
    CTR = "ctr"
    CALIBRATION = "calibration"
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    AUC = "auc"
    AUPRC = "auprc"
    RAUC = "rauc"
    GROUPED_AUC = "grouped_auc"
    GROUPED_AUPRC = "grouped_auprc"
    GROUPED_RAUC = "grouped_rauc"
    RECALL_SESSION_LEVEL = "recall_session_level"
    MULTICLASS_RECALL = "multiclass_recall"
    WEIGHTED_AVG = "weighted_avg"
    TOWER_QPS = "qps"
    ACCURACY = "accuracy"
    NDCG = "ndcg"
    XAUC = "xauc"
    SCALAR = "scalar"


class MetricNamespaceBase(StrValueMixin, Enum):
    pass


class MetricNamespace(MetricNamespaceBase):
    DEFAULT = ""

    NE = "ne"
    SEGMENTED_NE = "segmented_ne"
    THROUGHPUT = "throughput"
    CTR = "ctr"
    CALIBRATION = "calibration"
    MSE = "mse"
    AUC = "auc"
    AUPRC = "auprc"
    RAUC = "rauc"
    MAE = "mae"
    ACCURACY = "accuracy"

    OPTIMIZERS = "optimizers"
    MODEL_CONFIGURATOR = "model_configurator"

    MULTICLASS_RECALL = "multiclass_recall"

    WEIGHTED_AVG = "weighted_avg"
    RECALL_SESSION_LEVEL = "recall_session_level"

    TOWER_QPS = "qps"
    NDCG = "ndcg"
    XAUC = "xauc"

    SCALAR = "scalar"


class MetricPrefix(StrValueMixin, Enum):
    DEFAULT = ""
    LIFETIME = "lifetime_"
    WINDOW = "window_"


def task_wildcard_metrics_pattern(
    namespace: MetricNamespaceBase,
    metric_name: MetricNameBase,
    metric_prefix: MetricPrefix = MetricPrefix.DEFAULT,
) -> str:
    r"""Get the re (regular expression) pattern to find a set of metrics
    regardless task names. The motivation to have this API is from the past
    bugs which tools hard-code the patterns but the naming change, causing
    some testing issues.
    """
    return rf"{namespace}-.+\|{metric_prefix}{metric_name}"


def compose_metric_namespace(
    namespace: MetricNamespaceBase,
    task_name: str,
) -> str:
    r"""Get the full namespace of a metric based on the input parameters"""
    return f"{namespace}-{task_name}"


def compose_customized_metric_key(
    namespace: str,
    metric_name: str,
    description: Optional[str] = None,
) -> str:
    r"""Get the metric key. The input are unrestricted (string) namespace and
    metric_name. This API should only be used by compose_metric_key() and
    state metrics as the keys of state metrics are unknown.
    """
    return f"{namespace}|{metric_name}{description or ''}"


def compose_metric_key(
    namespace: MetricNamespaceBase,
    task_name: str,
    metric_name: MetricNameBase,
    metric_prefix: MetricPrefix = MetricPrefix.DEFAULT,
    description: Optional[str] = None,
) -> str:
    r"""Get the metric key based on the input parameters"""
    return compose_customized_metric_key(
        compose_metric_namespace(namespace, task_name),
        f"{metric_prefix}{metric_name}",
        description,
    )
