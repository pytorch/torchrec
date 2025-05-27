#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from typing import Mapping, Optional, Tuple, Union

logger: logging.Logger = logging.getLogger(__name__)


class PruningLogger(ABC):
    @abstractmethod
    def log_table_eviction_info(
        self,
        iteration: Optional[Union[bool, float, int]],
        rank: Optional[int],
        table_to_sizes_mapping: Mapping[str, Tuple[int, int]],
        eviction_tables: Mapping[str, float],
    ) -> None:
        pass

    @abstractmethod
    def log_run_info(
        self,
    ) -> None:
        pass


class PruningLoggerDefault(PruningLogger):
    """
    noop logger as a default
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialize PruningScubaLogger.
        """
        pass

    def log_table_eviction_info(
        self,
        iteration: Optional[Union[bool, float, int]],
        rank: Optional[int],
        table_to_sizes_mapping: Mapping[str, Tuple[int, int]],
        eviction_tables: Mapping[str, float],
    ) -> None:
        logger.info(
            f"iteration={iteration}, rank={rank}, table_to_sizes_mapping={table_to_sizes_mapping}, eviction_tables={eviction_tables}"
        )

    def log_run_info(
        self,
    ) -> None:
        pass
