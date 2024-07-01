#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, cast, Dict, List, Optional, Type

import torch
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)


ERROR_SUM = "error_sum"
WEIGHTED_NUM_SAMPES = "weighted_num_samples"


def compute_mae(
    error_sum: torch.Tensor, weighted_num_samples: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        weighted_num_samples == 0.0, 0.0, error_sum / weighted_num_samples
    ).double()


def compute_error_sum(
    labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    predictions = predictions.double()
    return torch.sum(weights * torch.abs(labels - predictions), dim=-1)


def get_mae_states(
    labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:

    return torch.stack(
        [
            compute_error_sum(labels, predictions, weights),  # error sum
            torch.sum(weights, dim=-1),  # weighted_num_samples
        ]
    )


class MAEMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for MAE, i.e. Mean Absolute Error.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.state_names = ["error_sum", "weighted_num_samples"]
        self._add_state(
            self.state_names,
            torch.zeros((len(self.state_names), self._n_tasks), dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )

    # pyre-fixme[14]: `update` overrides method defined in `RecMetricComputation`
    #  inconsistently.
    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> None:
        if predictions is None or weights is None:
            raise RecMetricException(
                "Inputs 'predictions' and 'weights' should not be None for MAEMetricComputation update"
            )
        states = get_mae_states(labels, predictions, weights)
        num_samples = predictions.shape[-1]
        state = getattr(self, self._fused_name)
        state += states
        self._aggregate_window_state(self._fused_name, states, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.MAE,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_mae(
                    self.get_state(ERROR_SUM),
                    self.get_state(WEIGHTED_NUM_SAMPES),
                ),
            ),
            MetricComputationReport(
                name=MetricName.MAE,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_mae(
                    self.get_window_state(ERROR_SUM),
                    self.get_window_state(WEIGHTED_NUM_SAMPES),
                ),
            ),
        ]


class MAEMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.MAE
    _computation_class: Type[RecMetricComputation] = MAEMetricComputation
