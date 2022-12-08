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

PREDICTIONS = "predictions"
LABELS = "labels"
WEIGHTS = "weights"


def compute_auc(
    n_tasks: int, predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    # The return values are sorted_predictions, sorted_index but only
    # sorted_predictions is needed.
    _, sorted_indices = torch.sort(predictions, descending=True, dim=-1)
    aucs = []
    for sorted_indices_i, labels_i, weights_i in zip(sorted_indices, labels, weights):
        sorted_labels = torch.index_select(labels_i, dim=0, index=sorted_indices_i)
        sorted_weights = torch.index_select(weights_i, dim=0, index=sorted_indices_i)
        cum_fp = torch.cumsum(sorted_weights * (1.0 - sorted_labels), dim=0)
        cum_tp = torch.cumsum(sorted_weights * sorted_labels, dim=0)
        auc = torch.where(
            cum_fp[-1] * cum_tp[-1] == 0,
            0.5,  # 0.5 is the no-signal default value for auc.
            torch.trapz(cum_tp, cum_fp) / cum_fp[-1] / cum_tp[-1],
        )
        aucs.append(auc.view(1))
    return torch.cat(aucs)


def _state_reduction(state: torch.Tensor) -> torch.Tensor:
    return torch.cat(list(state), dim=-1)


class AUCMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for AUC, i.e. Area Under the Curve.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.state_names: List[str] = [
            PREDICTIONS,
            LABELS,
            WEIGHTS,
        ]
        self._add_state(
            self.state_names,
            torch.zeros(
                (len(self.state_names), self._n_tasks, 1),
                dtype=torch.double,
                device=self.device,
            ),
            add_window_state=False,
            dist_reduce_fx=_state_reduction,
            persistent=False,
        )

    def _init_states(self) -> None:
        state = getattr(self, self._fused_name)
        state = torch.zeros(
            (len(self.state_names), self._n_tasks, 1),
            dtype=torch.double,
            device=self.device,
        )
        setattr(self, self._fused_name, state)

        for name, _ in self._fused_map.items():
            setattr(
                self,
                name,
                torch.zeros((self._n_tasks, 1), dtype=torch.double, device=self.device),
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
                "Inputs 'predictions' and 'weights' should not be None for AUCMetricComputation update"
            )
        predictions = predictions.double()
        labels = labels.double()
        weights = weights.double()
        state = getattr(self, self._fused_name)
        num_samples = state.size(-1)
        batch_size = predictions.size(-1)
        start_index = max(num_samples + batch_size - self._window_size, 0)

        states = torch.stack([predictions, labels, weights])
        state = torch.cat([state[:, :, start_index:], states], dim=-1)
        setattr(self, self._fused_name, state)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.AUC,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_auc(
                    self._n_tasks,
                    self.get_state(PREDICTIONS),
                    self.get_state(LABELS),
                    self.get_state(WEIGHTS),
                ),
            )
        ]

    def reset(self) -> None:
        super().reset()
        self._init_states()


class AUCMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.AUC
    _computation_class: Type[RecMetricComputation] = AUCMetricComputation
