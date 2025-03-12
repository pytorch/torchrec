#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from logging import getLogger, Logger
from typing import Callable, Dict, List, Optional, Tuple

import torch
from tensorboard.adhoc import Adhoc

from torch import distributed as dist

from torchrec.modules.mc_modules import MCHEvictionPolicy, MCHManagedCollisionModule
from torchrec.sparse.jagged_tensor import JaggedTensor

logger: Logger = getLogger(__name__)


class ScalarLogger(torch.nn.Module):
    """
    A logger to report various metrics related to ZCH.
    This module is adapted from ScalarLogger for multi-probe ZCH.

    Args:
        name (str): Name of the embedding table.
        frequency (int): Frequency of reporting metrics in number of iterations.

    Example::
        scalar_logger = ScalarLogger(
            name=name,
            frequency=tb_logging_frequency,
        )
    """

    def __init__(
        self,
        name: str,
        frequency: int,
    ) -> None:
        """
        Initializes the logger.

        Args:
            name (str): Name of the embedding table.
            frequency (int): Frequency of reporting metrics in number of iterations.

        Returns:
            None
        """
        super().__init__()
        self._name: str = name
        self._frequency: int = frequency

        # size related metrics
        self._unused_size: int = 0
        self._active_size: int = 0
        self._total_size: int = 0

        # scalar step, buffer to avoid TB logging issues during training job preemption
        self.register_buffer("_scalar_logger_steps", torch.tensor(0, dtype=torch.int64))
        self._steps_cpu: int = -1

    def _maybe_align_step_counter(self) -> None:
        """
        Align the step counter if not.
        """
        if self._steps_cpu == -1:
            self._steps_cpu = int(self._scalar_logger_steps.item())

    def _step(self) -> None:
        """
        Increments the step counter.
        """
        self._scalar_logger_steps += 1
        self._steps_cpu += 1

    def should_report(self) -> bool:
        """
        Returns whether the logger should report metrics.
        This function only returns True for rank 0 and every self._frequency steps.
        """
        if self._steps_cpu % self._frequency != 0:
            return False

        rank: int = -1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        return rank == 0

    def build_metric_name(
        self,
        metric: str,
        run_type: str,
    ) -> str:
        """
        Builds the metric name for reporting.

        Args:
            metric (str): Name of the metric.
            run_type (str): Run type of the model, e.g. train, eval, etc.

        Returns:
            str: Metric name.
        """
        return f"mc_zch_stats/{self._name}/{metric}/{run_type}"

    def update_size(
        self,
        counts: torch.Tensor,
    ) -> None:
        """
        Updates the size related metrics.

        Args:
            counts (torch.Tensor): Counts of each id in the embedding table.

        Returns:
            None
        """
        zero_counts = counts == 0
        self._unused_size = int(torch.sum(zero_counts).item())

        self._total_size = counts.shape[0]
        self._active_size = self._total_size - self._unused_size

    def _report(
        self,
        run_type: str,
    ) -> None:
        """
        Reports various metrics related to ZCH.

        Args:
            run_type (str): Run type of the model, e.g. train, eval, etc.

        Returns:
            None
        """
        if self.should_report():
            total_size = self._total_size + 0.001
            usused_ratio = round(self._unused_size / total_size, 3)
            active_ratio = round(self._active_size / total_size, 3)

            Adhoc.writer().add_scalar(
                self.build_metric_name("unused_size", run_type),
                self._unused_size,
                self._steps_cpu,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("usused_ratio", run_type),
                usused_ratio,
                self._steps_cpu,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("active_size", run_type),
                self._active_size,
                self._steps_cpu,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("active_ratio", run_type),
                active_ratio,
                self._steps_cpu,
            )

            logger.info(f"{self._name=}, {run_type=}")
            logger.info(f"{self._total_size=}")
            logger.info(f"{self._unused_size=}, {usused_ratio=}")
            logger.info(f"{self._active_size=}, {active_ratio=}")
            logger.info(f"{self._steps_cpu=}")

            # reset after reporting
            self._unused_size = 0
            self._active_size = 0
            self._total_size = 0

    def forward(
        self,
        run_type: str,
    ) -> None:
        """
        Reports various metrics related to ZCH.

        Args:
            run_type (str): Run type of the model, e.g. train, eval, etc.

        Returns:
            None
        """
        self._maybe_align_step_counter()

        self._report(run_type=run_type)

        self._step()


class MCHTBManagedCollisionModule(MCHManagedCollisionModule):
    def __init__(
        self,
        zch_size: int,
        device: torch.device,
        eviction_policy: MCHEvictionPolicy,
        eviction_interval: int,
        input_hash_size: int = (2**63) - 1,
        input_hash_func: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        mch_size: Optional[int] = None,
        mch_hash_func: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        name: Optional[str] = None,
        output_global_offset: int = 0,  # typically not provided by user
        output_segments: Optional[List[int]] = None,  # typically not provided by user
        buckets: int = 1,
        tb_logging_frequency: int = 0,
    ) -> None:
        super().__init__(
            zch_size=zch_size,
            device=device,
            eviction_policy=eviction_policy,
            eviction_interval=eviction_interval,
            input_hash_size=input_hash_size,
            input_hash_func=input_hash_func,
            mch_size=mch_size,
            mch_hash_func=mch_hash_func,
            name=name,
            output_global_offset=output_global_offset,
            output_segments=output_segments,
            buckets=buckets,
        )

        ## ------ logging ------
        self._tb_logging_frequency = tb_logging_frequency
        self._scalar_logger: Optional[ScalarLogger] = None
        if self._tb_logging_frequency > 0:
            assert self._name is not None, "name must be provided for logging"
            self._scalar_logger = ScalarLogger(
                name=self._name,
                frequency=self._tb_logging_frequency,
            )

    def profile(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        features = super().profile(features)

        if self._scalar_logger is not None:
            self._scalar_logger.update_size(counts=self._mch_metadata["counts"])

        return features

    def remap(self, features: Dict[str, JaggedTensor]) -> Dict[str, JaggedTensor]:
        remapped_features = super().remap(features)

        if self._scalar_logger is not None:
            self._scalar_logger(
                run_type="train" if self.training else "eval",
            )

        return remapped_features

    def rebuild_with_output_id_range(
        self,
        output_id_range: Tuple[int, int],
        output_segments: List[int],
        device: Optional[torch.device] = None,
    ) -> "MCHTBManagedCollisionModule":

        new_zch_size = output_id_range[1] - output_id_range[0]

        return type(self)(
            name=self._name,
            zch_size=new_zch_size,
            device=device or self.device,
            eviction_policy=self._eviction_policy,
            eviction_interval=self._eviction_interval,
            input_hash_size=self._input_hash_size,
            input_hash_func=self._input_hash_func,
            output_global_offset=output_id_range[0],
            output_segments=output_segments,
            buckets=len(output_segments) - 1,
            tb_logging_frequency=self._tb_logging_frequency,
        )
