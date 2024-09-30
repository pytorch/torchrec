#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from dataclasses import replace
from typing import Dict

import torch
from torchrec.metrics.metrics_config import DefaultTaskInfo

from torchrec.metrics.ndcg import NDCGMetric, SESSION_KEY


WORLD_SIZE = 4
BATCH_SIZE = 10


def get_test_case_single_session_within_batch() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
        "session_ids": torch.tensor([[1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[0.0, 1.0, 0.0, 0.0, 2.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0]]),
        "expected_ndcg_exp": torch.tensor([0.1103]),
        "expected_ndcg_non_exp": torch.tensor([0.1522]),
    }


def get_test_case_multiple_sessions_within_batch() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3]]),
        "session_ids": torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[0.0, 1.0, 0.0, 0.0, 2.0, 2.0, 1.0, 0.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0]]),
        "expected_ndcg_exp": torch.tensor([0.6748]),
        "expected_ndcg_non_exp": torch.tensor([0.6463]),
    }


def get_test_case_all_labels_zero() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3]]),
        "session_ids": torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0]]),
        "expected_ndcg_exp": torch.tensor([2.5]),
        "expected_ndcg_non_exp": torch.tensor([2.5]),
    }


def get_test_case_another_multiple_sessions_within_batch() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.5, 0.3, 0.4, 0.2, 0.1]]),
        "session_ids": torch.tensor([[1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        "expected_ndcg_exp": torch.tensor([(0.3066 + 0.0803) / 2]),
        "expected_ndcg_non_exp": torch.tensor([(0.3066 + 0.0803) / 2]),
    }


def get_test_case_at_k() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.5, 0.3, 0.4, 0.2, 0.1]]),
        "session_ids": torch.tensor([[1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        "expected_ndcg_exp": torch.tensor([(0.6131 + 0.3869) / 2]),
        "expected_ndcg_non_exp": torch.tensor([(0.6131 + 0.3869) / 2]),
    }


def get_test_case_remove_single_length_sessions() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.5, 0.3, 0.4, 0.2, 0.1]]),
        "session_ids": torch.tensor([[1, 1, 1, 2, 3, 4]]),
        "labels": torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        "expected_ndcg_exp": torch.tensor([0.3066]),
        "expected_ndcg_non_exp": torch.tensor([0.3066]),
    }


def get_test_case_negative_task() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.9, 0.5, 0.7, 0.6, 0.8, 0.9]]),
        "session_ids": torch.tensor([[1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]),
        "weights": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        "expected_ndcg_exp": torch.tensor([(0.3066 + 0.0803) / 2]),
        "expected_ndcg_non_exp": torch.tensor([(0.3066 + 0.0803) / 2]),
    }


"""
predictions = [0.1, 0.5, 0.3, 0.4, 0.2, 0.1] * weights = [1, 0, 0, 1, 0, 0] => [0.1, 0, 0, 0.4, 0.0, 0.0]
labels = [1, 0, 1, 1, 0, 1] * weights = [1, 0, 1, 1, 0, 1] => [1, 0, 0, 1, 0, 0]
    => NDCG going to be perfect for both sessions (trivially).
"""


def get_test_case_scale_by_weights_tensor() -> Dict[str, torch.Tensor]:
    return {
        "predictions": torch.tensor([[0.1, 0.5, 0.3, 0.4, 0.2, 0.1]]),
        "session_ids": torch.tensor([[1, 1, 1, 2, 2, 2]]),
        "labels": torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]),
        "weights": torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]),
        "expected_ndcg_exp": torch.tensor([(1.0 + 1.0) / 2]),
        "expected_ndcg_non_exp": torch.tensor([(1.0 + 1.0) / 2]),
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

        self.ndcg_at_k = NDCGMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            # pyre-ignore
            exponential_gain=False,  # exponential_gain is one of the kwargs
            # pyre-ignore
            session_key=SESSION_KEY,  # session_key is one of the kwargs
            # pyre-ignore[6]: In call `NDCGMetric.__init__`, for argument `k`, expected `Dict[str, typing.Any]` but got `int`
            k=2,
        )

        self.ndcg_remove_single_length_sessions = NDCGMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            # pyre-ignore
            exponential_gain=False,  # exponential_gain is one of the kwargs
            # pyre-ignore
            session_key=SESSION_KEY,  # session_key is one of the kwargs
            # pyre-ignore[6]: In call `NDCGMetric.__init__`, for argument `remove_single_length_sessions`, expected `Dict[str, typing.Any]` but got `bool`
            remove_single_length_sessions=True,
        )

        TempTaskInfo = replace(DefaultTaskInfo, is_negative_task=True)
        self.ndcg_apply_negative_task_mask = NDCGMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[TempTaskInfo],
            # pyre-ignore
            exponential_gain=False,  # exponential_gain is one of the kwargs
            # pyre-ignore
            session_key=SESSION_KEY,  # session_key is one of the kwargs
        )

        self.ndcg_report_as_increasing_and_scale_by_weights_tensor = NDCGMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
            # pyre-ignore
            exponential_gain=False,  # exponential_gain is one of the kwargs
            # pyre-ignore
            session_key=SESSION_KEY,  # session_key is one of the kwargs
            # pyre-ignore[6]: In call `NDCGMetric.__init__`, for argument `remove_single_length_sessions`, expected `Dict[str, typing.Any]` but got `bool`
            remove_single_length_sessions=True,
            # pyre-ignore[6]: In call `NDCGMetric.__init__`, for argument `scale_by_weights_tensor`, expected `Dict[str, typing.Any]` but got `bool`
            scale_by_weights_tensor=True,
            # pyre-ignore[6]: In call `NDCGMetric.__init__`, for argument `report_ndcg_as_decreasing_curve`, expected `Dict[str, typing.Any]` but got `bool`
            report_ndcg_as_decreasing_curve=False,
        )

    def test_single_session(self) -> None:
        """
        Test single session in a update.
        """
        model_output = get_test_case_multiple_sessions_within_batch()
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
        Test multiple sessions in a single update.
        """
        model_output = get_test_case_multiple_sessions_within_batch()
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

    def test_negative_sessions(self) -> None:
        """
        Test sessions where all labels are 0.
        """
        model_output = get_test_case_all_labels_zero()
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

    def test_another_multiple_sessions(self) -> None:
        """
        Test another multiple sessions in a single update.
        """
        model_output = get_test_case_another_multiple_sessions_within_batch()
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

    def test_at_k(self) -> None:
        """
        Test NDCG @ K.
        """
        model_output = get_test_case_at_k()
        self.ndcg_at_k.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        metric = self.ndcg_at_k.compute()
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

    def test_remove_single_length_sessions(self) -> None:
        """
        Test NDCG with removing single length sessions.
        """
        model_output = get_test_case_remove_single_length_sessions()
        self.ndcg_remove_single_length_sessions.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )
        metric = self.ndcg_remove_single_length_sessions.compute()
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

    def test_apply_negative_task_mask(self) -> None:
        """
        Test NDCG with apply negative task mask.
        """
        model_output = get_test_case_negative_task()
        self.ndcg_apply_negative_task_mask.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )

        metric = self.ndcg_apply_negative_task_mask.compute()
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

    def test_case_report_as_increasing_ndcg_and_scale_by_weights_tensor(self) -> None:
        """
        Test NDCG with reporting as increasing NDCG and scaling by weights tensor correctly.
        """
        model_output = get_test_case_scale_by_weights_tensor()
        self.ndcg_report_as_increasing_and_scale_by_weights_tensor.update(
            predictions={DefaultTaskInfo.name: model_output["predictions"][0]},
            labels={DefaultTaskInfo.name: model_output["labels"][0]},
            weights={DefaultTaskInfo.name: model_output["weights"][0]},
            required_inputs={SESSION_KEY: model_output["session_ids"][0]},
        )

        metric = self.ndcg_report_as_increasing_and_scale_by_weights_tensor.compute()
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
