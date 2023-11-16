#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, Dict, List, Optional, Type

import torch
from torch import distributed as dist
from torchrec.metrics.metrics_config import RecComputeMode, RecTaskInfo
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)

SUM_NDCG = "sum_ndcg"
NUM_SESSIONS = "num_sessions"
REQUIRED_INPUTS = "required_inputs"
SESSION_KEY = "session_id"


def session_ids_to_lengths(
    session_ids: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Convert session_ids to lengths tensor. It is used in all session-wise loss
    computations.

    Args:
        session_ids: a tensor of session_ids,
            e.g., ["1", "2", "2"]
        device: the device to put session_ids into

    Returns:
        session_lengths(torch.Tensor): a tensor of session lengths, e.g., tensor([1, 2])
    """

    session_lengths: List[int] = []
    if len(session_ids) == 0:
        return torch.zeros(1)
    length = 1
    for i in range(len(session_ids) - 1):
        if session_ids[i] == session_ids[i + 1]:
            length += 1
        else:
            session_lengths.append(length)
            length = 1
    session_lengths.append(length)

    return torch.tensor(session_lengths, dtype=torch.int, device=device)


def compute_lambda_ndcg(
    prediction: torch.Tensor,
    label: torch.Tensor,
    weight: torch.Tensor,
    session_lengths: torch.Tensor,
    use_exp_gain: bool,
) -> torch.Tensor:
    """
    Compute the sum lambda NDCG loss from a group of sessions.

    Args:
        prediction(torch.Tensor): a tensor of predicted scores
        label(torch.Tensor): a tensor of labels
        weight(torch.Tensor): a tensor of weights
        session_lengths(torch.Tensor): a tensor of session lengths converted from
            session_ids
        use_exp_gain(bool): whether to use exponential gain or not

    Returns:
        sum_loss(torch.Tensor): a tensor of the sum of the ndcg loss
    """

    loss = torch.zeros_like(session_lengths, dtype=torch.double)

    cur_index = int(0)
    for i, session_length in enumerate(session_lengths):
        data_indexes = torch.arange(
            cur_index,
            cur_index + int(session_length),
            dtype=torch.long,
            device=prediction.device,
        )
        session_loss = compute_lambda_ndcg_by_session(
            prediction=torch.take(prediction, data_indexes),
            label=torch.take(label, data_indexes),
            use_exp_gain=use_exp_gain,
        )
        loss[i] = session_loss * torch.max(torch.take(weight, data_indexes))
        cur_index += session_length

    return torch.sum(loss)


def compute_lambda_ndcg_by_session(
    prediction: torch.Tensor,
    label: torch.Tensor,
    use_exp_gain: bool,
) -> torch.Tensor:
    """
    Compute the lambda NDCG loss for one session.

    Args:
        prediction(torch.Tensor): a tensor of predicted scores
        label(torch.Tensor): a tensor of labels
        use_exp_gain(bool): whether to use exponential gain or not

    Returns:
        loss(torch.Tensor): a tensor of approximate ndcg loss
    """

    gain = torch.exp2(label).sub(1.0) if use_exp_gain else label
    discounts = get_position_discounts(prediction)

    idcg = gain @ get_position_discounts(label)
    idcg = torch.max(idcg, torch.tensor(1e-6))
    dcg = gain @ discounts

    return 1 - dcg / idcg


def get_position_discounts(t: torch.Tensor) -> torch.Tensor:
    orders = torch.argsort(torch.argsort(t, descending=True))
    return torch.reciprocal(torch.log2(orders.add(2.0))).type(torch.double)


def _validate_model_outputs(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    session_ids: torch.Tensor,
) -> None:
    assert predictions.shape == labels.shape == weights.shape == session_ids.shape
    assert (
        predictions.dim() == 2 and predictions.shape[0] > 0 and predictions.shape[1] > 0
    )


def get_ndcg_states(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
    session_ids: torch.Tensor,
    exponential_gain: bool,
) -> Dict[str, torch.Tensor]:
    n_tasks = labels.shape[0]
    sum_ndcg = torch.zeros(n_tasks, dtype=torch.double)

    for i in range(n_tasks):
        session_lengths = session_ids_to_lengths(
            session_ids=session_ids[i], device=predictions.device
        )

        sum_ndcg[i] = compute_lambda_ndcg(
            prediction=predictions[i],
            label=labels[i],
            weight=weights[i],
            session_lengths=session_lengths,
            use_exp_gain=exponential_gain,
        )

    return {SUM_NDCG: sum_ndcg, NUM_SESSIONS: session_lengths.shape[-1]}


def compute_ndcg(sum_ndcg: torch.Tensor, num_sessions: torch.Tensor) -> torch.Tensor:
    return sum_ndcg / num_sessions


class NDCGComputation(RecMetricComputation):
    r"""
    This class implements the RecMetricComputation for NDCG, i.e. Normalized Discounted Cumulative Gain.

    The constructor arguments are defined in RecMetricComputation.
    See the docstring of RecMetricComputation for more detail.

    """

    def __init__(
        self,
        *args: Any,
        exponential_gain: bool = False,
        session_key: str = SESSION_KEY,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._exponential_gain: bool = exponential_gain
        self._session_key: str = session_key

        self._add_state(
            SUM_NDCG,
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            NUM_SESSIONS,
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
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Args:
            predictions (torch.Tensor): tensor of size (n_task, n_examples)
            labels (torch.Tensor): tensor of size (n_task, n_examples)
            weights (torch.Tensor): tensor of size (n_task, n_examples)
        """
        if (
            REQUIRED_INPUTS not in kwargs
            or self._session_key not in kwargs[REQUIRED_INPUTS]
        ):
            raise RecMetricException(
                f"{self._session_key=} {kwargs=} input is required to calculate NDCG"
            )

        session_ids = kwargs[REQUIRED_INPUTS][self._session_key]

        if predictions is None or weights is None or session_ids is None:
            raise RecMetricException(
                "Inputs 'predictions', 'weights' and 'session_ids' should not be None for NDCGMetricComputation update"
            )

        _validate_model_outputs(predictions, labels, weights, session_ids)

        predictions = predictions.double()
        labels = labels.double()
        weights = weights.double()

        states = get_ndcg_states(
            labels=labels,
            predictions=predictions,
            weights=weights,
            session_ids=session_ids,
            exponential_gain=self._exponential_gain,
        )
        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, predictions.shape[-1])

    def _compute(self) -> List[MetricComputationReport]:

        return [
            MetricComputationReport(
                name=MetricName.NDCG,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_ndcg(
                    sum_ndcg=cast(torch.Tensor, getattr(self, SUM_NDCG)),
                    num_sessions=cast(torch.Tensor, getattr(self, NUM_SESSIONS)),
                ),
            ),
            MetricComputationReport(
                name=MetricName.NDCG,
                metric_prefix=MetricPrefix.WINDOW,
                value=compute_ndcg(
                    sum_ndcg=self.get_window_state(SUM_NDCG),
                    num_sessions=self.get_window_state(NUM_SESSIONS),
                ),
            ),
        ]


class NDCGMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.NDCG
    _computation_class: Type[RecMetricComputation] = NDCGComputation

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
        self._required_inputs.add("session_id")
