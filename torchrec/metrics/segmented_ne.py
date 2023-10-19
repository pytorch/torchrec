#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
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
SEGMENTS = "segments"


def compute_cross_entropy(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    predictions = predictions.double()
    predictions.clamp_(min=eta, max=1 - eta)
    cross_entropy = -weights * labels * torch.log2(predictions) - weights * (
        1.0 - labels
    ) * torch.log2(1.0 - predictions)
    return cross_entropy


def _compute_cross_entropy_norm(
    mean_label: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    mean_label = mean_label.double()
    mean_label.clamp_(min=eta, max=1 - eta)
    return -pos_labels * torch.log2(mean_label) - neg_labels * torch.log2(
        1.0 - mean_label
    )


def compute_ne_helper(
    ce_sum: torch.Tensor,
    weighted_num_samples: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    mean_label = pos_labels / weighted_num_samples
    ce_norm = _compute_cross_entropy_norm(mean_label, pos_labels, neg_labels, eta)
    return ce_sum / ce_norm


def compute_logloss(
    ce_sum: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    # we utilize tensor broadcasting for operations
    labels_sum = pos_labels + neg_labels
    labels_sum.clamp_(min=eta)
    return ce_sum / labels_sum


def compute_ne(
    ce_sum: torch.Tensor,
    weighted_num_samples: torch.Tensor,
    pos_labels: torch.Tensor,
    neg_labels: torch.Tensor,
    num_groups: int,
    eta: float,
) -> torch.Tensor:
    # size should be (num_groups)
    result_ne = torch.zeros(num_groups)
    for group in range(num_groups):
        mean_label = pos_labels[group] / weighted_num_samples[group]
        ce_norm = _compute_cross_entropy_norm(
            mean_label, pos_labels[group], neg_labels[group], eta
        )
        ne = ce_sum[group] / ce_norm
        result_ne[group] = ne

    # ne indexed by group - tensor size (num_groups)
    return result_ne


def get_segemented_ne_states(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
    grouping_keys: torch.Tensor,
    eta: float,
    num_groups: int,
) -> Dict[str, torch.Tensor]:
    groups = torch.unique(grouping_keys)
    cross_entropy, weighted_num_samples, pos_labels, neg_labels = (
        torch.zeros(num_groups),
        torch.zeros(num_groups),
        torch.zeros(num_groups),
        torch.zeros(num_groups),
    )
    for group in groups:
        group_mask = grouping_keys == group

        group_labels = labels[group_mask]
        group_predictions = predictions[group_mask]
        group_weights = weights[group_mask]

        ce_sum_group = torch.sum(
            compute_cross_entropy(
                labels=group_labels,
                predictions=group_predictions,
                weights=group_weights,
                eta=eta,
            ),
            dim=-1,
        )

        weighted_num_samples_group = torch.sum(group_weights, dim=-1)
        pos_labels_group = torch.sum(group_weights * group_labels, dim=-1)
        neg_labels_group = torch.sum(group_weights * (1.0 - group_labels), dim=-1)

        cross_entropy[group] = ce_sum_group.item()
        weighted_num_samples[group] = weighted_num_samples_group.item()
        pos_labels[group] = pos_labels_group.item()
        neg_labels[group] = neg_labels_group.item()

    # tensor size for each value is (num_groups)
    return {
        "cross_entropy_sum": cross_entropy,
        "weighted_num_samples": weighted_num_samples,
        "pos_labels": pos_labels,
        "neg_labels": neg_labels,
    }


def _state_reduction_sum(state: torch.Tensor) -> torch.Tensor:
    return state.sum(dim=0)


class SegmentedNEMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for Segmented NE, i.e. Normalized Entropy - for boolean labels.

    Only binary labels are currently supported (0s, 1s), NE is computed for each label, NE across the whole model output
    can be done through the normal NE metric.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.

    Args:
        include_logloss (bool): return vanilla logloss as one of metrics results, on top of segmented NE.
    """

    def __init__(
        self,
        *args: Any,
        include_logloss: bool = False,  # TODO - include
        num_groups: int = 1,
        **kwargs: Any,
    ) -> None:
        self._include_logloss: bool = include_logloss
        super().__init__(*args, **kwargs)
        self._num_groups = num_groups  # would there be checkpointing issues with this? maybe make this state
        self._add_state(
            "cross_entropy_sum",
            torch.zeros((self._n_tasks, num_groups), dtype=torch.double),
            add_window_state=False,
            dist_reduce_fx=_state_reduction_sum,
            persistent=True,
        )
        self._add_state(
            "weighted_num_samples",
            torch.zeros((self._n_tasks, num_groups), dtype=torch.double),
            add_window_state=False,
            dist_reduce_fx=_state_reduction_sum,
            persistent=True,
        )
        self._add_state(
            "pos_labels",
            torch.zeros((self._n_tasks, num_groups), dtype=torch.double),
            add_window_state=False,
            dist_reduce_fx=_state_reduction_sum,
            persistent=True,
        )
        self._add_state(
            "neg_labels",
            torch.zeros((self._n_tasks, num_groups), dtype=torch.double),
            add_window_state=False,
            dist_reduce_fx=_state_reduction_sum,
            persistent=True,
        )
        self.eta = 1e-12

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
                "Inputs 'predictions' and 'weights' and 'grouping_keys' should not be None for NEMetricComputation update"
            )
        elif (
            "required_inputs" not in kwargs
            or kwargs["required_inputs"].get("grouping_keys") is None
        ):
            raise RecMetricException(
                f"Required inputs for SegmentedNEMetricComputation update should contain 'grouping_keys', got kwargs: {kwargs}"
            )
        elif kwargs["required_inputs"]["grouping_keys"].dtype != torch.int64:
            raise RecMetricException(
                f"Grouping keys must have type torch.int64, got {kwargs['required_inputs']['grouping_keys'].dtype}."
            )

        grouping_keys = kwargs["required_inputs"]["grouping_keys"]
        states = get_segemented_ne_states(
            labels,
            predictions,
            weights,
            grouping_keys,
            eta=self.eta,
            num_groups=self._num_groups,
        )

        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value

    def _compute(self) -> List[MetricComputationReport]:
        reports = []
        computed_ne = compute_ne(
            self.cross_entropy_sum[0],
            self.weighted_num_samples[0],
            self.pos_labels[0],
            self.neg_labels[0],
            num_groups=self._num_groups,
            eta=self.eta,
        )

        for group in range(self._num_groups):
            reports.append(
                MetricComputationReport(
                    name=MetricName.SEGMENTED_NE,
                    metric_prefix=MetricPrefix.LIFETIME,
                    value=computed_ne[group],
                    description="_" + str(group),
                ),
            )

        if self._include_logloss:
            log_loss_groups = compute_logloss(
                self.cross_entropy_sum[0],
                self.pos_labels[0],
                self.neg_labels[0],
                eta=self.eta,
            )

            for group in range(self._num_groups):
                reports.append(
                    MetricComputationReport(
                        name=MetricName.LOG_LOSS,
                        metric_prefix=MetricPrefix.LIFETIME,
                        value=log_loss_groups[group],
                        description="_" + str(group),
                    )
                )

        return reports


class SegmentedNEMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.SEGMENTED_NE
    _computation_class: Type[RecMetricComputation] = SegmentedNEMetricComputation
