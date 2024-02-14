#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchrec.metrics.metrics_config import DefaultTaskInfo
from torchrec.metrics.scalar import ScalarMetric


WORLD_SIZE = 4
BATCH_SIZE = 10


class ScalarMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scalar = ScalarMetric(
            world_size=WORLD_SIZE,
            my_rank=0,
            batch_size=BATCH_SIZE,
            tasks=[DefaultTaskInfo],
        )

    def test_scalar(self) -> None:
        """
        Test scalar metric passes through each tensor as is
        """
        metric_to_log = torch.tensor([0.1])

        self.scalar.update(
            labels={DefaultTaskInfo.name: metric_to_log},
            predictions={DefaultTaskInfo.name: metric_to_log},
            weights={DefaultTaskInfo.name: metric_to_log},
        )
        metric = self.scalar.compute()
        actual_metric = metric[f"scalar-{DefaultTaskInfo.name}|lifetime_scalar"]

        torch.testing.assert_close(
            actual_metric,
            metric_to_log,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {metric_to_log}",
        )

        # Pass through second tensor with different value
        # check we get the value back with no averaging or any differences

        metric_to_log = torch.tensor([0.9])

        self.scalar.update(
            labels={DefaultTaskInfo.name: metric_to_log},
            predictions={DefaultTaskInfo.name: metric_to_log},
            weights={DefaultTaskInfo.name: metric_to_log},
        )
        metric = self.scalar.compute()
        actual_metric = metric[f"scalar-{DefaultTaskInfo.name}|lifetime_scalar"]

        torch.testing.assert_close(
            actual_metric,
            metric_to_log,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_metric}, Expected: {metric_to_log}",
        )

    def test_scalar_window(self) -> None:
        """
        Test windowing of scalar metric gives average of previously reported values.
        """
        metric_to_log = torch.tensor([0.1])

        self.scalar.update(
            labels={DefaultTaskInfo.name: metric_to_log},
            predictions={DefaultTaskInfo.name: metric_to_log},
            weights={DefaultTaskInfo.name: metric_to_log},
        )

        metric_to_log = torch.tensor([0.9])

        self.scalar.update(
            labels={DefaultTaskInfo.name: metric_to_log},
            predictions={DefaultTaskInfo.name: metric_to_log},
            weights={DefaultTaskInfo.name: metric_to_log},
        )

        metric = self.scalar.compute()

        actual_window_metric = metric[f"scalar-{DefaultTaskInfo.name}|window_scalar"]

        expected_window_metric = torch.tensor([0.5])

        torch.testing.assert_close(
            actual_window_metric,
            expected_window_metric,
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
            equal_nan=True,
            msg=f"Actual: {actual_window_metric}, Expected: {expected_window_metric}",
        )
