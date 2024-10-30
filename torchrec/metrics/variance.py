#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, List, Optional, Type

import torch

from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
)
from torchrec.pt2.utils import pt2_compile_callable


class VarianceMetricComputation(RecMetricComputation):
    """
    Metric that logs the variance and mean of the given "label".
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            "sum_deviation_squared",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "sum_values",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "num_samples",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )

    @pt2_compile_callable
    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        batch_size = labels.shape[-1]
        current_mean = (self.sum_values / self.num_samples.clamp(min=1)).unsqueeze(1)
        delta = labels - current_mean
        delta_2 = delta * (
            self.num_samples / (self.num_samples + batch_size).clamp(min=1)
        ).unsqueeze(1)

        num_samples_update = torch.zeros_like(labels[:, 0]).fill_(batch_size)
        values_update = labels.sum(dim=-1)
        m2_update = (delta_2 * delta).sum(dim=-1)

        for state_name, state_value in zip(
            ["num_samples", "sum_values", "sum_deviation_squared"],
            [num_samples_update, values_update, m2_update],
        ):
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, batch_size)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.VARIANCE,
                metric_prefix=MetricPrefix.LIFETIME,
                value=self.sum_deviation_squared / (self.num_samples - 1).clamp(min=1),
            ),
            MetricComputationReport(
                name=MetricName.VARIANCE,
                metric_prefix=MetricPrefix.WINDOW,
                value=self.get_window_state("sum_deviation_squared")
                / (self.get_window_state("num_samples") - 1).clamp(min=1),
            ),
            MetricComputationReport(
                name=MetricName.SCALAR,
                metric_prefix=MetricPrefix.LIFETIME,
                value=self.sum_values / self.num_samples.clamp(min=1),
            ),
            MetricComputationReport(
                name=MetricName.SCALAR,
                metric_prefix=MetricPrefix.WINDOW,
                value=self.get_window_state("sum_values")
                / self.get_window_state("num_samples").clamp(min=1),
            ),
        ]


class VarianceMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.VARIANCE
    _computation_class: Type[RecMetricComputation] = VarianceMetricComputation
