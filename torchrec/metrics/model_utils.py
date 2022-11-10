#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
from torchrec.metrics.rec_metric import RecTaskInfo


def is_empty_signals(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    weights: torch.Tensor,
) -> bool:
    return (
        torch.numel(labels) <= 0
        and torch.numel(predictions) <= 0
        and torch.numel(weights) <= 0
    )


def parse_model_outputs(
    label_name: str,
    prediction_name: str,
    weight_name: str,
    model_out: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    labels = model_out[label_name].squeeze()
    if not prediction_name:
        assert not weight_name, "weight name must be empty if prediction name is empty"
        return (labels, None, None)
    assert isinstance(labels, torch.Tensor)
    predictions = model_out[prediction_name].squeeze()
    assert isinstance(predictions, torch.Tensor)
    weights = model_out[weight_name].squeeze()
    assert isinstance(weights, torch.Tensor)

    if not is_empty_signals(labels, predictions, weights):
        if labels.dim() == predictions.dim():
            assert (torch.numel(labels) == torch.numel(predictions)) and (
                torch.numel(labels) == torch.numel(weights)
            ), (
                "Expect the same number of elements in labels, predictions, and weights. "
                f"Instead got {torch.numel(labels)}, {torch.numel(predictions)}, "
                f"{torch.numel(weights)}"
            )
        else:  # For multiclass models, labels.size() = (batch_size), and predictions.size() = (batch_size, number_of_classes)
            assert torch.numel(labels) == torch.numel(predictions) / predictions.size()[
                -1
            ] and torch.numel(labels) == torch.numel(weights)

        # non-empty tensors need to have rank 1
        if len(labels.size()) == 0:
            labels = labels.unsqueeze(0)
            predictions = predictions.unsqueeze(0)
            weights = weights.unsqueeze(0)

    return labels, predictions, weights


def parse_task_model_outputs(
    tasks: List[RecTaskInfo], model_out: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    all_labels: Dict[str, torch.Tensor] = {}
    all_predictions: Dict[str, torch.Tensor] = {}
    all_weights: Dict[str, torch.Tensor] = {}
    for task in tasks:
        labels, predictions, weights = parse_model_outputs(
            task.label_name, task.prediction_name, task.weight_name, model_out
        )
        if predictions is not None and weights is not None:
            if not is_empty_signals(labels, predictions, weights):
                all_labels[task.name] = labels
                all_predictions[task.name] = predictions
                all_weights[task.name] = weights
        else:
            if torch.numel(labels) > 0:
                all_labels[task.name] = labels

    return all_labels, all_predictions, all_weights
