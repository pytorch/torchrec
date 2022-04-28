#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import unittest

from torchrec.metrics.metrics_namespace import (
    compose_metric_key,
    task_wildcard_metrics_pattern,
    MetricName,
    MetricNamespace,
    MetricPrefix,
)


class MetricNamespaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task = "abc"

    def test_compose_metric_key(self) -> None:
        self.assertEqual(
            compose_metric_key(
                MetricNamespace.QPS,
                self.task,
                MetricName.QPS,
                MetricPrefix.LIFETIME,
            ),
            f"qps-{self.task}|lifetime_qps",
        )
        self.assertEqual(
            compose_metric_key(
                MetricNamespace.QPS,
                self.task,
                MetricName.QPS,
                MetricPrefix.WINDOW,
            ),
            f"qps-{self.task}|window_qps",
        )
        self.assertEqual(
            compose_metric_key(
                MetricNamespace.QPS,
                self.task,
                MetricName.QPS,
            ),
            f"qps-{self.task}|qps",
        )

    def test_task_wildcard_metric_pattern(self) -> None:
        pattern = task_wildcard_metrics_pattern(
            MetricNamespace.QPS,
            MetricName.QPS,
            MetricPrefix.LIFETIME,
        )
        key1 = compose_metric_key(
            MetricNamespace.QPS,
            "t1",
            MetricName.QPS,
            MetricPrefix.LIFETIME,
        )
        key2 = compose_metric_key(
            MetricNamespace.QPS,
            "t2",
            MetricName.QPS,
            MetricPrefix.LIFETIME,
        )
        key3 = compose_metric_key(
            MetricNamespace.QPS,
            "t3",
            MetricName.QPS,
            MetricPrefix.WINDOW,
        )
        self.assertTrue(re.search(pattern, key1))
        self.assertTrue(re.search(pattern, key2))
        self.assertFalse(re.search(pattern, key3))
