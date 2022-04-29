#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import math
import time
from collections import deque
from typing import Deque, Dict

import torch
import torch.nn as nn
from torchrec.metrics.metrics_namespace import (
    compose_metric_key,
    MetricName,
    MetricNamespace,
    MetricPrefix,
)

MAX_WINDOW_TS = 1_000_000


class QPSMetric(nn.Module):
    """
    The module to calculate QPS.

    Args:
        batch_size (int): batch size for the trainer
        world_size (int): the number of trainers
        window_seconds (int): QPS use time-based window for window_qps. This
                              argument specify the window size in seconds.
        warmup_steps (int): the number of warmup batches. No QPS will be calculated
                            before the warmup batches count reached.

    Call Args:
        Not supported.

    Returns:
        Not supported.

    Example::

        qps = QPSMetric(
                  batch_size=128,
                  world_size=4,
                  window_seconds=100,
                  warmup_steps=100
              )
    """

    _namespace: MetricNamespace
    _metric_name: MetricName
    _batch_examples: int
    _window_seconds: int
    _warmup_steps: int
    _window_time_lapse_buffer: Deque[float]
    _window_time_lapse: float
    _previous_ts: float
    _lifetime_qps_key: str
    _window_qps_key: str
    _total_examples_key: str
    _steps: int

    def __init__(
        self,
        *,
        batch_size: int,
        world_size: int,
        window_seconds: int,
        warmup_steps: int = 100,
    ) -> None:
        super().__init__()
        self._namespace = MetricNamespace.QPS
        self._metric_name = MetricName.QPS
        if window_seconds < 1:
            raise ValueError(
                "window_seconds must be at least 1 to give window QPS the minimum time window"
            )
        if warmup_steps < 1:
            raise ValueError(
                "warmup_steps must be at least 1 to give QPS a reasonable begin time."
            )

        self._batch_examples = batch_size * world_size
        self._window_seconds = window_seconds
        self._warmup_steps = warmup_steps

        self.register_buffer("total_examples", torch.tensor(0, dtype=torch.long))
        self.register_buffer("warmup_examples", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "time_lapse_after_warmup", torch.tensor(0, dtype=torch.double)
        )

        self._window_time_lapse_buffer = deque(maxlen=MAX_WINDOW_TS)
        self._window_time_lapse = 0
        self._previous_ts = 0

        self._lifetime_qps_key = compose_metric_key(
            self._namespace,
            str(self._namespace),
            self._metric_name,
            MetricPrefix.LIFETIME,
        )
        self._window_qps_key = compose_metric_key(
            self._namespace,
            str(self._namespace),
            self._metric_name,
            MetricPrefix.WINDOW,
        )
        self._total_examples_key = compose_metric_key(
            self._namespace,
            str(self._namespace),
            MetricName.TOTAL_EXAMPLES,
        )

        self._steps = 0

    def _check_window(self) -> None:
        while self._window_time_lapse > self._window_seconds:
            self._window_time_lapse -= self._window_time_lapse_buffer.popleft()

    def update(self) -> None:
        ts = time.monotonic()
        self._steps += 1
        self.total_examples += self._batch_examples

        if self._steps <= self._warmup_steps:
            self.warmup_examples += self._batch_examples
            if self._steps == self._warmup_steps:
                self._previous_ts = ts
        else:
            time_lapse = ts - self._previous_ts
            self.time_lapse_after_warmup += time_lapse
            self._window_time_lapse += time_lapse
            self._window_time_lapse_buffer.append(time_lapse)
            self._check_window()
            self._previous_ts = ts

    def compute(self) -> Dict[str, torch.Tensor]:
        ret = {self._total_examples_key: self.total_examples}
        if self._steps > self._warmup_steps and (
            not math.isclose(self.time_lapse_after_warmup.item(), 0)
        ):
            lifetime_qps = (
                self.total_examples - self.warmup_examples
            ) / self.time_lapse_after_warmup
            if not math.isclose(self._window_time_lapse, 0):
                window_qps = (
                    len(self._window_time_lapse_buffer)
                    * self._batch_examples
                    / self._window_time_lapse
                )
            else:
                window_qps = 0.0
            if not math.isclose(lifetime_qps.item(), 0):
                ret.update(
                    {
                        self._lifetime_qps_key: torch.tensor(
                            lifetime_qps, dtype=torch.double
                        ),
                        self._window_qps_key: torch.tensor(
                            window_qps, dtype=torch.double
                        ),
                    }
                )
        return ret
