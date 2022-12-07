#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, Dict, List, Optional, Type

import torch
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)

CALIBRATION_NUM = "calibration_num"
CALIBRATION_DENOM = "calibration_denom"


def compute_calibration(
    calibration_num: torch.Tensor, calibration_denom: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        calibration_denom <= 0.0, 0.0, calibration_num / calibration_denom
    ).double()


def get_calibration_states(
    labels: torch.Tensor, predictions: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return torch.stack(
        [
            # state "calibration_num"
            torch.sum(predictions * weights, dim=-1),
            # state "calibration_denom"
            torch.sum(labels * weights, dim=-1),
        ]
    )


class CalibrationMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for Calibration, which is the
    ratio between the prediction and the labels (conversions).

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        state_names = [
            CALIBRATION_NUM,
            CALIBRATION_DENOM,
        ]
        self._add_state(
            state_names,
            torch.zeros((len(state_names), self._n_tasks), dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        if predictions is None or weights is None:
            raise RecMetricException(
                "Inputs 'predictions' and 'weights' should not be None for CalibrationMetricComputation update"
            )
        num_samples = predictions.shape[-1]

        states = get_calibration_states(labels, predictions, weights)
        state = getattr(self, self._fused_name)
        state += states
        self._aggregate_window_state(self._fused_name, states, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.CALIBRATION,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_calibration(
                    self.get_state(CALIBRATION_NUM),
                    self.get_state(CALIBRATION_DENOM),
                ),
            ),
            MetricComputationReport(
                name=MetricName.CALIBRATION,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_calibration(
                    self.get_window_state(CALIBRATION_NUM),
                    self.get_window_state(CALIBRATION_DENOM),
                ),
            ),
        ]


class CalibrationMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.CALIBRATION
    _computation_class: Type[RecMetricComputation] = CalibrationMetricComputation
