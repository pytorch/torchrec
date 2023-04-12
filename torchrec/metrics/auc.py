#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, cast, Dict, List, Optional, Type

import torch
import torch.distributed as dist
from torchrec.metrics.metrics_config import RecComputeMode, RecTaskInfo
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
GROUPING_KEYS = "grouping_keys"
REQUIRED_INPUTS = "required_inputs"


def _compute_auc_helper(
    predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    _, sorted_indices = torch.sort(predictions, descending=True, dim=-1)
    sorted_labels = torch.index_select(labels, dim=0, index=sorted_indices)
    sorted_weights = torch.index_select(weights, dim=0, index=sorted_indices)
    cum_fp = torch.cumsum(sorted_weights * (1.0 - sorted_labels), dim=0)
    cum_tp = torch.cumsum(sorted_weights * sorted_labels, dim=0)
    auc = torch.where(
        cum_fp[-1] * cum_tp[-1] == 0,
        0.5,  # 0.5 is the no-signal default value for auc.
        torch.trapz(cum_tp, cum_fp) / cum_fp[-1] / cum_tp[-1],
    )
    return auc


def compute_auc(
    n_tasks: int, predictions: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    Computes AUC (Area Under the Curve) for binary classification.

    Args:
        n_tasks (int): number of tasks.
        predictions (torch.Tensor): tensor of size (n_tasks, n_examples).
        labels (torch.Tensor): tensor of size (n_tasks, n_examples).
        weights (torch.Tensor): tensor of size (n_tasks, n_examples).
    """
    # The return values are sorted_predictions, sorted_index but only
    # sorted_predictions is needed.
    _, sorted_indices = torch.sort(predictions, descending=True, dim=-1)
    aucs = []
    for predictions_i, labels_i, weights_i in zip(predictions, labels, weights):
        auc = _compute_auc_helper(predictions_i, labels_i, weights_i)
        aucs.append(auc.view(1))
    return torch.cat(aucs)


def compute_auc_per_group(
    n_tasks: int,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    grouping_keys: torch.Tensor,
) -> torch.Tensor:
    """
    Computes AUC (Area Under the Curve) for binary classification for groups of predictions/labels.
    Args:
        n_tasks (int): number of tasks
        predictions (torch.Tensor): tensor of size (n_tasks, n_examples)
        labels (torch.Tensor): tensor of size (n_tasks, n_examples)
        weights (torch.Tensor): tensor of size (n_tasks, n_examples)
        grouping_keys (torch.Tensor): tensor of size (n_examples,)

    Returns:
        torch.Tensor: tensor of size (n_tasks,), average of AUCs per group.
    """
    aucs = []
    if grouping_keys.numel() != 0 and grouping_keys[0] == -1:
        # we added padding  as the first elements during init to avoid floating point exception in sync()
        # removing the paddings to avoid numerical errors.
        grouping_keys = grouping_keys[1:]
        predictions = predictions[:, 1:]
        labels = labels[:, 1:]
        weights = weights[:, 1:]

    # get unique group indices
    group_indices = torch.unique(grouping_keys)

    for (predictions_i, labels_i, weights_i) in zip(predictions, labels, weights):
        # Loop over each group
        auc_groups_sum = torch.tensor([0], dtype=torch.float32)
        for group_idx in group_indices:
            # get predictions, labels, and weights for this group
            group_mask = grouping_keys == group_idx
            grouped_predictions = predictions_i[group_mask]
            grouped_labels = labels_i[group_mask]
            grouped_weights = weights_i[group_mask]

            auc = _compute_auc_helper(
                grouped_predictions, grouped_labels, grouped_weights
            )
            auc_groups_sum = auc_groups_sum.to(auc.device)
            auc_groups_sum += auc.view(1)
        avg_auc = (
            auc_groups_sum / len(group_indices)
            if len(group_indices) > 0
            else torch.tensor([0.5], dtype=torch.float32)
        )
        aucs.append(avg_auc)
    return torch.cat(aucs)


def _state_reduction(state: List[torch.Tensor], dim: int = 1) -> List[torch.Tensor]:
    return [torch.cat(state, dim=dim)]


# pyre-ignore
_grouping_keys_state_reduction = partial(_state_reduction, dim=0)


class AUCMetricComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for AUC, i.e. Area Under the Curve.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.
    Args:
        grouped_auc (bool): If True, computes AUC per group and returns average AUC across all groups.
            The `grouping_keys` is provided during state updates along with predictions, labels, weights.
            This feature is currently not enabled for `fused_update_limit`.
    """

    def __init__(
        self,
        *args: Any,
        grouped_auc: bool = False,
        fused_update_limit: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if grouped_auc and fused_update_limit > 0:
            raise RecMetricException(
                "Grouped AUC and Fused Update Limit cannot be enabled together yet."
            )

        self._grouped_auc: bool = grouped_auc
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
        if self._grouped_auc:
            self._add_state(
                GROUPING_KEYS,
                [],
                add_window_state=False,
                dist_reduce_fx=_grouping_keys_state_reduction,
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
        if self._grouped_auc:
            getattr(self, GROUPING_KEYS).append(torch.tensor([-1], device=self.device))

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Args:
            predictions (torch.Tensor): tensor of size (n_task, n_examples)
            labels (torch.Tensor): tensor of size (n_task, n_examples)
            weights (torch.Tensor): tensor of size (n_task, n_examples)
            grouping_key (torch.Tensor): Optional tensor of size (1, n_examples) that specifies the groups of
                    predictions/labels per batch. If provided, the AUC metric also
                    computes AUC per group and returns the average AUC across all groups.
        """
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
        if self._grouped_auc:
            if REQUIRED_INPUTS not in kwargs or (
                (grouping_keys := kwargs[REQUIRED_INPUTS].get(GROUPING_KEYS)) is None
            ):
                raise RecMetricException(
                    f"Input '{GROUPING_KEYS}' are required for AUCMetricComputation grouped update"
                )
            getattr(self, GROUPING_KEYS)[0] = torch.cat(
                [
                    cast(torch.Tensor, getattr(self, GROUPING_KEYS)[0])[start_index:],
                    grouping_keys.squeeze(),
                ],
                dim=0,
            )

    def _compute(self) -> List[MetricComputationReport]:
        reports = [
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
        if self._grouped_auc:
            reports.append(
                MetricComputationReport(
                    name=MetricName.GROUPED_AUC,
                    metric_prefix=MetricPrefix.WINDOW,
                    value=compute_auc_per_group(
                        self._n_tasks,
                        cast(torch.Tensor, getattr(self, PREDICTIONS)[0]),
                        cast(torch.Tensor, getattr(self, LABELS)[0]),
                        cast(torch.Tensor, getattr(self, WEIGHTS)[0]),
                        cast(torch.Tensor, getattr(self, GROUPING_KEYS)[0]),
                    ),
                )
            )
        return reports

    def reset(self) -> None:
        super().reset()
        self._init_states()


class AUCMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.AUC
    _computation_class: Type[RecMetricComputation] = AUCMetricComputation

    def __init__(
        self,
        world_size: int,
        my_rank: int,
        batch_size: int,
        tasks: List[RecTaskInfo],
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        window_size: int = 100,
        fused_update_limit: int = 0,
        compute_on_all_ranks: bool = False,
        should_validate_update: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            world_size=world_size,
            my_rank=my_rank,
            batch_size=batch_size,
            tasks=tasks,
            compute_mode=compute_mode,
            window_size=window_size,
            fused_update_limit=fused_update_limit,
            compute_on_all_ranks=compute_on_all_ranks,
            should_validate_update=should_validate_update,
            process_group=process_group,
            **kwargs,
        )
        if kwargs.get("grouped_auc"):
            self._required_inputs.add(GROUPING_KEYS)
