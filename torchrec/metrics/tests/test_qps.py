#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import unittest
from unittest.mock import Mock, patch

from torchrec.metrics.qps import (
    QPSMetric,
)


QPS_PATH = "torchrec.metrics.qps"


class QPSMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world_size = 64
        self.batch_size = 256

    @patch(QPS_PATH + ".time.monotonic")
    def test_no_batches(self, time_mock: Mock) -> None:
        time_mock.return_value = 1
        qps_metric = QPSMetric(
            batch_size=self.batch_size, world_size=self.world_size, window_seconds=100
        )
        self.assertEqual(qps_metric.compute(), {"qps-qps|total_examples": 0})

    @patch(QPS_PATH + ".time.monotonic")
    def test_one_batch(self, time_mock: Mock) -> None:
        time_mock.return_value = 1
        qps_metric = QPSMetric(
            batch_size=self.batch_size, world_size=self.world_size, window_seconds=100
        )
        qps_metric.update()
        self.assertEqual(
            qps_metric.compute(),
            {"qps-qps|total_examples": self.batch_size * self.world_size},
        )

    @patch(QPS_PATH + ".time.monotonic")
    def _test_qps(self, time_mock: Mock, warmup_steps: int) -> None:
        update_timestamps = [10, 11, 12, 14, 15, 17, 18, 20, 21, 22, 25, 29, 30]
        update_timestamps.sort()
        qps_metric = QPSMetric(
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_seconds=10,
            warmup_steps=warmup_steps,
        )

        window_time_lapse_buffer = []
        window_time_lapse = 0
        for i in range(len(update_timestamps)):
            time_mock.return_value = update_timestamps[i]
            qps_metric.update()
            if i >= warmup_steps:
                window_time_lapse_buffer.append(
                    update_timestamps[i] - update_timestamps[i - 1]
                )
                window_time_lapse += update_timestamps[i] - update_timestamps[i - 1]
            ret = qps_metric.compute()

            total_examples = self.world_size * self.batch_size * (i + 1)
            if i < warmup_steps:
                self.assertEqual(ret, {"qps-qps|total_examples": total_examples})
                continue

            lifetime_examples = total_examples - (
                self.world_size * self.batch_size * warmup_steps
            )
            lifetime_qps = lifetime_examples / (
                update_timestamps[i] - update_timestamps[warmup_steps - 1]
            )

            while window_time_lapse > 10:
                window_time_lapse -= window_time_lapse_buffer.pop(0)
            window_qps = (
                len(window_time_lapse_buffer)
                * self.world_size
                * self.batch_size
                / window_time_lapse
            )

            self.assertEqual(ret["qps-qps|lifetime_qps"], lifetime_qps)
            self.assertEqual(ret["qps-qps|window_qps"], window_qps)
            self.assertEqual(ret["qps-qps|total_examples"], total_examples)

    def test_qps_warmup_steps_0(self) -> None:
        with self.assertRaises(ValueError):
            self._test_qps(warmup_steps=0)

    def test_qps_warmup_steps_1(self) -> None:
        self._test_qps(warmup_steps=1)

    def test_qps_warmup_steps_2(self) -> None:
        self._test_qps(warmup_steps=2)

    def test_qps_warmup_steps_10(self) -> None:
        self._test_qps(warmup_steps=10)

    def test_warmup_checkpointing(self) -> None:
        warmup_steps = 5
        extra_steps = 2
        qps_metric = QPSMetric(
            batch_size=self.batch_size,
            world_size=self.world_size,
            window_seconds=10,
            warmup_steps=warmup_steps,
        )
        for i in range(5):
            for _ in range(warmup_steps + extra_steps):
                qps_metric.update()
            self.assertEqual(
                qps_metric.warmup_examples.item(),
                warmup_steps * (i + 1) * self.batch_size * self.world_size,
            )
            self.assertEqual(
                qps_metric.total_examples.item(),
                (warmup_steps + extra_steps)
                * (i + 1)
                * self.batch_size
                * self.world_size,
            )

            # Mimic trainer crashing and loading a checkpoint
            qps_metric._steps = 0
