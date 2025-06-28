#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

from torchrec.distributed.logging_handlers import _log_handlers

TORCHREC_LOGGER_NAME = "torchrec"

_log_handlers.update({TORCHREC_LOGGER_NAME: logging.NullHandler()})
