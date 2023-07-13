#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict

import torch
from torchrec.metrics.metrics_config import DefaultTaskInfo

from torchrec.metrics.ndcg import NDCGMetric, SESSION_KEY


WORLD_SIZE = 4
BATCH_SIZE = 10


def generate_model_output_single_session() -> Dict[str, torch._tensor.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        "session_ids": torch.tensor([[1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[0.0, 1.0, 0.0, 0.0, 2.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0]]),
        "expected_ndcg_exp": torch.tensor([0.1103]),
        "expected_ndcg_non_exp": torch.tensor([0.1522]),
    }


def generate_model_output_multiple_sessions() -> Dict[str, torch._tensor.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3]]),
        "session_ids": torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[0.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0]]),
        "expected_ndcg_exp": torch.tensor([0.6748]),
        "expected_ndcg_non_exp": torch.tensor([0.6463]),
    }


def generate_model_output_negative_sessions() -> Dict[str, torch._tensor.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3]]),
        "session_ids": torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0]]),
        "expected_ndcg_exp": torch.tensor([2.5]),
        "expected_ndcg_non_exp": torch.tensor([2.5]),
    }


class NDCGMetricValueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.non_exponential_ndcg = NDCGMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            # pyre-ignore
            exponential_gain=False,  # exponential_gain is one of the kwargs
            # pyre-ignore
            session_key=SESSION_KEY,  # session_key is one of the kwargs
        )

        self.exponential_ndcg = NDCGMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            # pyre-ignore
            exponential_gain=True,  # exponential_gain is one of the kwargs
            # pyre-ignore
            session_key=SESSION_KEY,  # session_key is one of the kwargs
        )

    def test_single_session(self) -> None:
        """
        Test single session in a update
        """
        model_output = generate_model_output_multiple_sessions()
        self.non_exponential_ndcg.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        metric = self.non_exponential_ndcg.compute()
        actual_metric = metric[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_non_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

        self.exponential_ndcg.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        metric = self.exponential_ndcg.compute()
        actual_metric = metric[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_multiple_sessions(self) -> None:
        """
        Test multiple sessions in a single update
        """
        model_output = generate_model_output_single_session()
        self.non_exponential_ndcg.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        metric = self.non_exponential_ndcg.compute()
        actual_metric = metric[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_non_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

        self.exponential_ndcg.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        metric = self.exponential_ndcg.compute()
        actual_metric = metric[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

    def test_negtive_sessions(self) -> None:
        """
        Test negative sessions which all labels are 0
        """
        model_output = generate_model_output_negative_sessions()
        self.non_exponential_ndcg.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        metric = self.non_exponential_ndcg.compute()
        actual_metric = metric[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_non_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )

        self.exponential_ndcg.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        metric = self.exponential_ndcg.compute()
        actual_metric = metric[f"ndcg-{DefaultTaskInfo.name}|lifetime_ndcg"]
        expected_metric = model_output["expected_ndcg_exp"]

        torch.testing.assert_close(
            actual_metric,
            expected_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {expected_metric}",
        )
