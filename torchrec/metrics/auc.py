#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, List, Optional, Type

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


def _state_reduction(state: List[torch.Tensor]) -> List[torch.Tensor]:
    return [torch.cat(state, dim=1)]


class AUCMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for AUC, i.e. Area Under the Curve.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._add_state(
            PREDICTIONS,
            [],
            add_window_state=False,
            dist_reduce_fx=_state_reduction,
            persistent=False,
        )
        self._add_state(
            LABELS,
            [],
            add_window_state=False,
            dist_reduce_fx=_state_reduction,
            persistent=False,
        )
        self._add_state(
            WEIGHTS,
            [],
            add_window_state=False,
            dist_reduce_fx=_state_reduction,
            persistent=False,
        )
        self._init_states()

    # The states values are set to empty lists in __init__() and reset(), and then we
    # add a size (self._n_tasks, 1) tensor to each of the list as the initial values
    # This is to bypass the limitation of state aggregation in TorchMetrics sync() when
    # we try to checkpoint the states before update()
    # The reason for using lists here is to avoid automatically stacking the tensors from
    # all the trainers into one tensor in sync()
    # The reason for using non-empty tensors as the first elements is to avoid the
    # floating point exception thrown in sync() for aggregating empty tensors
    def _init_states(self) -> None:
        if len(getattr(self, PREDICTIONS)) > 0:
            return

        getattr(self, PREDICTIONS).append(
            torch.zeros((self._n_tasks, 1), dtype=torch.double, device=self.device)
        )
        getattr(self, LABELS).append(
            torch.zeros((self._n_tasks, 1), dtype=torch.double, device=self.device)
        )
        getattr(self, WEIGHTS).append(
            torch.zeros((self._n_tasks, 1), dtype=torch.double, device=self.device)
        )

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> None:
        if predictions is None or weights is None:
            raise RecMetricException(
                "Inputs 'predictions' and 'weights' should not be None for AUCMetricComputation update"
            )
        predictions = predictions.double()
        labels = labels.double()
        weights = weights.double()
        num_samples = getattr(self, PREDICTIONS)[0].size(-1)
        batch_size = predictions.size(-1)
        start_index = max(num_samples + batch_size - self._window_size, 0)
        # Using `self.predictions =` will cause Pyre errors.
        getattr(self, PREDICTIONS)[0] = torch.cat(
            [
                cast(torch.Tensor, getattr(self, PREDICTIONS)[0])[:, start_index:],
                predictions,
            ],
            dim=-1,
        )
        getattr(self, LABELS)[0] = torch.cat(
            [cast(torch.Tensor, getattr(self, LABELS)[0])[:, start_index:], labels],
            dim=-1,
        )
        getattr(self, WEIGHTS)[0] = torch.cat(
            [cast(torch.Tensor, getattr(self, WEIGHTS)[0])[:, start_index:], weights],
            dim=-1,
        )

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.AUC,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_auc(
                    self._n_tasks,
                    cast(torch.Tensor, getattr(self, PREDICTIONS)[0]),
                    cast(torch.Tensor, getattr(self, LABELS)[0]),
                    cast(torch.Tensor, getattr(self, WEIGHTS)[0]),
                ),
            )
        ]

    def reset(self) -> None:
        super().reset()
        self._init_states()


class AUCMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.AUC
    _computation_class: Type[RecMetricComputation] = AUCMetricComputation
