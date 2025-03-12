#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import logging
import time
from typing import Optional

import torch

from tensorboard.adhoc import Adhoc
from torchrec.modules.hash_mc_evictions import HashZchEvictionConfig

logger: logging.Logger = logging.getLogger(__name__)


class ScalarLogger(torch.nn.Module):
    """
    A logger to report various metrics related to multi-probe ZCH.

    Args:
        name: name of the embedding table.
        zch_size: size of the sharded embedding table.
        frequency: frequency of reporting metrics.


    Example::
        logger = ScalarLogger(...)
        logger(run_type, identities)
    """

    STEPS_BUFFER: str = "_scalar_logger_steps"
    SECONDS_IN_HOUR: int = 3600
    MAX_HOURS: int = 2**31 - 1

    def __init__(
        self,
        name: str,
        zch_size: int,
        frequency: int,
    ) -> None:
        super().__init__()

        self.register_buffer(
            ScalarLogger.STEPS_BUFFER, torch.tensor(1, dtype=torch.int64)
        )

        self._name: str = name
        self._zch_size: int = zch_size
        self._frequency: int = frequency
        self._dtype_checked: bool = False
        self._total_cnt: int = 0
        self._hit_cnt: int = 0
        self._collision_cnt: int = 0
        self._eviction_cnt: int = 0
        self._opt_in_id_cnt: int = 0
        self._sum_eviction_age: float = 0.0
        self._unoccupied_cnt: int = 0

    def build_metric_name(
        self,
        metric: str,
        run_type: str,
    ) -> str:
        return f"mpzch/{self._name}/{metric}/{run_type}"

    def update(
        self,
        identities_0: torch.Tensor,
        values: torch.Tensor,
        remapped_ids: torch.Tensor,
        evicted_emb_indices: Optional[torch.Tensor],
        metadata: Optional[torch.Tensor],
        num_reserved_slots: int,
        expire_hr: int,
        eviction_config: Optional[HashZchEvictionConfig] = None,
    ) -> None:
        if not self._dtype_checked:
            assert (
                identities_0.dtype == values.dtype
            ), "identity type and feature type must match for meaningful metrics collection."
            self._dtype_checked = True

        remapped_identities_0 = torch.index_select(identities_0, 0, remapped_ids)[:, 0]
        unoccupied = remapped_identities_0 == -1
        self._unoccupied_cnt += int(torch.sum(unoccupied).item())

        self._total_cnt += values.numel()
        hits = torch.eq(remapped_identities_0, values)
        self._hit_cnt += int(torch.sum(hits).item())
        non_collisions = torch.logical_or(unoccupied, hits)
        self._collision_cnt += values.numel() - int(torch.sum(non_collisions).item())

        in_zch_range = self._zch_size - num_reserved_slots
        ids_in_zch_range = torch.lt(remapped_ids, in_zch_range)
        self._opt_in_id_cnt += int(torch.sum(ids_in_zch_range).item())

        if evicted_emb_indices is not None and evicted_emb_indices.numel() > 0:
            deduped_evicted_indices = torch.unique(evicted_emb_indices)
            self._eviction_cnt += deduped_evicted_indices.numel()

            assert (
                metadata is not None
            ), "metadata cannot be None when evicted_emb_indices has values"
            now_c = int(time.time())
            hours = now_c / ScalarLogger.SECONDS_IN_HOUR
            cur_hour = hours % ScalarLogger.MAX_HOURS
            shape = (len(deduped_evicted_indices),)
            if expire_hr > 0:
                self._sum_eviction_age += int(
                    torch.sum(
                        torch.full(shape, cur_hour, device=metadata.device)
                        - metadata[deduped_evicted_indices, 0]
                    ).item()
                )
            elif eviction_config is not None and eviction_config.single_ttl is not None:
                ttl = (
                    eviction_config.single_ttl
                    if eviction_config.single_ttl is not None
                    else 0
                )
                self._sum_eviction_age += int(
                    torch.sum(
                        torch.full(shape, cur_hour, device=metadata.device)
                        - metadata[deduped_evicted_indices, 0]
                        + torch.full(shape, ttl, device=metadata.device)
                    ).item()
                )

    def forward(
        self,
        run_type: str,
        identities: torch.Tensor,
    ) -> None:
        """
        Args:
            run_type: type of the run (train, eval, etc).
            identities: the identities tensor for metrics computation.

        Returns:
            None
        """

        # pyre-ignore[29]: `Union[(self: TensorBase, other: Any) -> Tensor, Tensor, Module]` is not a function.
        if self._scalar_logger_steps % self._frequency == 0:
            hit_rate = round(self._hit_cnt / self._total_cnt, 3)
            collision_rate = round(self._collision_cnt / self._total_cnt, 3)
            total_unused_slots = int(torch.sum(identities[:, 0] == -1).item())
            usage_ratio = round(
                (self._zch_size - total_unused_slots) / self._zch_size, 3
            )
            opt_in_ratio = (
                round(self._opt_in_id_cnt / self._total_cnt, 3)
                if self._total_cnt > 0
                else 0
            )
            avg_eviction_age = (
                round(self._sum_eviction_age / self._eviction_cnt, 3)
                if self._eviction_cnt > 0
                else 0
            )

            Adhoc.writer().add_scalar(
                self.build_metric_name("total_unused_slots", run_type),
                total_unused_slots,
                self._scalar_logger_steps,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("usage_ratio", run_type),
                usage_ratio,
                self._scalar_logger_steps,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("hit_cnt", run_type),
                self._hit_cnt,
                self._scalar_logger_steps,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("hit_rate", run_type),
                hit_rate,
                self._scalar_logger_steps,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("collision_cnt", run_type),
                self._collision_cnt,
                self._scalar_logger_steps,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("collision_rate", run_type),
                collision_rate,
                self._scalar_logger_steps,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("eviction_cnt", run_type),
                self._eviction_cnt,
                self._scalar_logger_steps,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("opt_in_id_cnt", run_type),
                self._opt_in_id_cnt,
                self._scalar_logger_steps,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("opt_in_ratio", run_type),
                opt_in_ratio,
                self._scalar_logger_steps,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("avg_eviction_age", run_type),
                avg_eviction_age,
                self._scalar_logger_steps,
            )
            Adhoc.writer().add_scalar(
                self.build_metric_name("unoccupied_cnt", run_type),
                self._unoccupied_cnt,
                self._scalar_logger_steps,
            )

            logger.info(
                f"{self._name=}, {run_type=}, "
                f"{self._total_cnt=}, {self._unoccupied_cnt=}, {total_unused_slots=}, {usage_ratio=}, "
                f"{self._hit_cnt=}, {hit_rate=}, {self._opt_in_id_cnt=}, {opt_in_ratio=}, "
                f"{self._collision_cnt=}, {collision_rate=}, "
                f"{self._eviction_cnt=}, {avg_eviction_age=}"
            )

            # reset the counter after reporting
            self._total_cnt = 0
            self._hit_cnt = 0
            self._collision_cnt = 0
            self._eviction_cnt = 0
            self._opt_in_id_cnt = 0
            self._sum_eviction_age = 0.0
            self._unoccupied_cnt = 0

        # pyre-ignore[16]: `ScalarLogger` has no attribute `_scalar_logger_steps`.
        # pyre-ignore[29]: `Union[(self: TensorBase, other: Any) -> Tensor, Tensor, Module]` is not a function.
        self._scalar_logger_steps += 1
