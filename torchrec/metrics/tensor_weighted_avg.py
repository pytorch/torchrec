#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, cast, Dict, List, Optional, Set, Type, Union

import torch
from torchrec.metrics.metrics_config import RecTaskInfo
from torchrec.metrics.metrics_namespace import MetricName, MetricNamespace, MetricPrefix
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetric,
    RecMetricComputation,
    RecMetricException,
)


def get_mean(value_sum: torch.Tensor, num_samples: torch.Tensor) -> torch.Tensor:
    return value_sum / num_samples


class TensorWeightedAvgMetricComputation(RecMetricComputation):
    def __init__(
        self,
        *args: Any,
        tensor_name: Optional[str] = None,
        weighted: bool = True,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if tensor_name is None:
            raise RecMetricException(
                f"TensorWeightedAvgMetricComputation expects tensor_name to not be None got {tensor_name}"
            )
        self.tensor_name: str = tensor_name
        self.weighted: bool = weighted
        self._add_state(
            "weighted_sum",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "weighted_num_samples",
            torch.zeros(self._n_tasks, dtype=torch.double),
            add_window_state=True,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._description = description

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        if (
            "required_inputs" not in kwargs
            or self.tensor_name not in kwargs["required_inputs"]
        ):
            raise RecMetricException(
                f"TensorWeightedAvgMetricComputation expects {self.tensor_name} in the required_inputs"
            )
        num_samples = labels.shape[0]
        target_tensor = cast(torch.Tensor, kwargs["required_inputs"][self.tensor_name])
        weights = cast(torch.Tensor, weights)
        states = {
            "weighted_sum": (
                target_tensor * weights if self.weighted else target_tensor
            ).sum(dim=-1),
            "weighted_num_samples": (
                weights.sum(dim=-1)
                if self.weighted
                else torch.ones(weights.shape).sum(dim=-1).to(device=weights.device)
            ),
        }
        for state_name, state_value in states.items():
            state = getattr(self, state_name)
            state += state_value
            self._aggregate_window_state(state_name, state_value, num_samples)

    def _compute(self) -> List[MetricComputationReport]:
        return [
            MetricComputationReport(
                name=MetricName.WEIGHTED_AVG,
                metric_prefix=MetricPrefix.LIFETIME,
                value=get_mean(
                    cast(torch.Tensor, self.weighted_sum),
                    cast(torch.Tensor, self.weighted_num_samples),
                ),
                description=self._description,
            ),
            MetricComputationReport(
                name=MetricName.WEIGHTED_AVG,
                metric_prefix=MetricPrefix.WINDOW,
                value=get_mean(
                    self.get_window_state("weighted_sum"),
                    self.get_window_state("weighted_num_samples"),
                ),
                description=self._description,
            ),
        ]


class TensorWeightedAvgMetric(RecMetric):
    _namespace: MetricNamespace = MetricNamespace.WEIGHTED_AVG
    _computation_class: Type[RecMetricComputation] = TensorWeightedAvgMetricComputation

    def __init__(
        self,
        # pyre-ignore Missing parameter annotation [2]
        *args,
        **kwargs: Dict[str, Any],
    ) -> None:

        super().__init__(*args, **kwargs)

    def _get_task_kwargs(
        self, task_config: Union[RecTaskInfo, List[RecTaskInfo]]
    ) -> Dict[str, Any]:
        if not isinstance(task_config, RecTaskInfo):
            raise RecMetricException(
                f"TensorWeightedAvgMetric expects task_config to be RecTaskInfo not {type(task_config)}. Check the FUSED_TASKS_COMPUTATION settings."
            )
        return {
            "tensor_name": task_config.tensor_name,
            "weighted": task_config.weighted,
        }

    def _get_task_required_inputs(
        self, task_config: Union[RecTaskInfo, List[RecTaskInfo]]
    ) -> Set[str]:
        if not isinstance(task_config, RecTaskInfo):
            raise RecMetricException(
                f"TensorWeightedAvgMetric expects task_config to be RecTaskInfo not {type(task_config)}. Check the FUSED_TASKS_COMPUTATION settings."
            )
        required_inputs = set()
        if task_config.tensor_name is not None:
            required_inputs.add(task_config.tensor_name)
        return required_inputs
