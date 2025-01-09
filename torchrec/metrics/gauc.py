#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, cast, Dict, List, Optional, Type

import torch

from torch.autograd.profiler import record_function
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)


def compute_gauc_3d(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_candidates: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Both predictions and labels are 3-d tensors in shape [n_task, n_group, n_sample]."""

    n_task, n_group, n_sample = predictions.size()
    max_len = max(n_task, n_group, n_sample)
    # Pre-register an arange to avoid multiple cpu=>gpu assignment.
    pre_arange = torch.arange(max_len, device=predictions.device)

    with record_function("## gauc_argsort ##"):
        sorted_indices = torch.argsort(predictions, descending=True, dim=-1)
    task_indices = (
        pre_arange[:n_task][:, None, None]
        .expand(n_task, n_group, n_sample)
        .contiguous()
        .view(-1)
    )
    group_indices = (
        pre_arange[:n_group][None, :, None]
        .expand(n_task, n_group, n_sample)
        .contiguous()
        .view(-1)
    )
    sample_indices = sorted_indices.contiguous().view(-1)
    sorted_labels = labels[task_indices, group_indices, sample_indices].view(
        n_task, n_group, n_sample
    )

    with record_function("## gauc_calculation ##"):
        num_sample = num_candidates[None, :].expand(n_task, n_group)
        # Count number of padding zeros.
        num_zeros = (n_sample - num_candidates)[None, :, None].expand(
            n_task, n_group, n_sample
        )  # [n_task, n_group, n_sample]
        # This assumes the labels are binary.
        num_zeros = (sorted_labels != 0) * num_zeros
        rank = torch.flip(pre_arange[:n_sample] + 1, [0])[None, None, :].expand(
            n_task, n_group, n_sample
        )
        positive_rank = sorted_labels * rank - num_zeros  # [n_task, n_group, n_sample]
        num_positive = sorted_labels.sum(-1)  # [n_task, n_group]

        # AUC is calcuated as (sum{positive_ranks} - num{positive_pairs}) /
        # (num_positive * num_negative).
        numerator = torch.sum(positive_rank, -1) - (
            num_positive * (num_positive + 1) / 2
        )
        denominator = num_positive * (num_sample - num_positive)
        auc = numerator / (denominator + 1e-10)  # [n_task, n_group]

    # Skip identical prediction sessions.
    identical_prediction_mask = ~(
        torch.all(
            torch.logical_or(
                predictions == predictions[:, :, 0:1],
                predictions == 0,  # avoid padding zeros.
            ),
            dim=-1,
        )
    )
    # Skip identical label(all 0s/1s) sessions.
    identical_label_mask = (num_positive >= 1) * (num_positive < num_sample)
    auc_mask = identical_label_mask * identical_prediction_mask
    auc *= auc_mask
    num_effective_samples = auc_mask.sum(-1)  # [n_task]
    auc = auc.sum(-1)  # [n_task]
    return {"auc_sum": auc, "num_samples": num_effective_samples}


def to_3d(
    tensor_2d: torch.Tensor, seq_lengths: torch.Tensor, max_length: int
) -> torch.Tensor:
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths)
    return torch.ops.fbgemm.jagged_2d_to_dense(tensor_2d, offsets, max_length)


def get_auc_states(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: Optional[torch.Tensor],
    num_candidates: torch.Tensor,
) -> Dict[str, torch.Tensor]:

    # predictions, labels: [n_task, n_sample]
    max_length = int(num_candidates.max().item())
    predictions_perm = predictions.permute(1, 0)
    labels_perm = labels.permute(1, 0)
    predictions_3d = to_3d(predictions_perm, num_candidates, max_length).permute(
        2, 0, 1
    )
    labels_3d = to_3d(labels_perm, num_candidates, max_length).permute(2, 0, 1)

    return compute_gauc_3d(
        predictions_3d,
        labels_3d,
        num_candidates,
    )


@torch.fx.wrap
def compute_window_auc(
    auc: torch.Tensor,
    num_samples: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    # [n_task]
    return {
        "gauc": (auc + 1e-9) / (num_samples + 2e-9),
        "num_samples": num_samples,
    }


class GAUCMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for GAUC, i.e. Session AUC.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._add_state(
            "auc_sum",
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

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        num_candidates: torch.Tensor,
        **kwargs: Dict[str, Any],
    ) -> None:
        if predictions is None or labels is None:
            raise RecMetricException(
                "Inputs 'predictions' and 'labels' should not be None for GAUCMetricComputation update"
            )

        states = get_auc_states(labels, predictions, weights, num_candidates)
        num_samples = predictions.shape[-1]

        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        reports = [
            MetricComputationReport(
                name=MetricName.GAUC,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_window_auc(
                    cast(torch.Tensor, self.auc_sum),
                    cast(torch.Tensor, self.num_samples),
                )["gauc"],
            ),
            MetricComputationReport(
                name=MetricName.GAUC,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_window_auc(
                    self.get_window_state("auc_sum"),
                    self.get_window_state("num_samples"),
                )["gauc"],
            ),
            MetricComputationReport(
                name=MetricName.GAUC_NUM_SAMPLES,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_window_auc(
                    cast(torch.Tensor, self.auc_sum),
                    cast(torch.Tensor, self.num_samples),
                )["num_samples"],
            ),
            MetricComputationReport(
                name=MetricName.GAUC_NUM_SAMPLES,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_window_auc(
                    self.get_window_state("auc_sum"),
                    self.get_window_state("num_samples"),
                )["num_samples"],
            ),
        ]

        return reports


class GAUCMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.GAUC
    _computation_class: Type[RecMetricComputation] = GAUCMetricComputation
