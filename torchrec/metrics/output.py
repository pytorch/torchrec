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
    RecMetricException,
)


class OutputMetricComputation(RecMetricComputation):
    """
    Metric that logs whatever model outputs are given in kwargs
    TODO - make this generic metric that can be used for any model output tensor
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            "latest_imp",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=False,
            dist_reduce_fx="sum",
            persistent=False,
        )
        self._add_state(
            "total_latest_imp",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=False,
            dist_reduce_fx="sum",
            persistent=False,
        )

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        if "latest_imp" and "total_latest_imp" not in kwargs:
            raise RecMetricException(
                "OutputMetricComputation requires 'latest_imp' and 'total_latest_imp' in kwargs"
            )
        states = {
            "latest_imp": kwargs["latest_imp"],
            "total_latest_imp": kwargs["total_latest_imp"],
        }

        for state_name, state_value in states.items():
            setattr(self, state_name, state_value)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.OUTPUT,
                metric_prefix=MetricPrefix.LIFETIME,
                value=self.latest_imp,
                description="latest_imp",
            ),
            MetricComputationReport(
                name=MetricName.OUTPUT,
                metric_prefix=MetricPrefix.LIFETIME,
                value=self.total_latest_imp,
                description="total_latest_imp",
            ),
        ]


class OutputMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.OUTPUT
    _computation_class: Type[RecMetricComputation] = OutputMetricComputation
