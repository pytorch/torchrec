#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import logging

import torch

from torchrec.distributed.quantized_comms.types import (
    TORCH_BFLOAT16_MAX,
    TORCH_BFLOAT16_MIN,
    TORCH_HALF_MAX,
    TORCH_HALF_MIN,
)

logger: logging.Logger = logging.getLogger()


def fp32_to_fp16_with_clamp(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor, TORCH_HALF_MIN, TORCH_HALF_MAX).half()


def fp32_to_bf16_with_clamp(tensor: torch.Tensor) -> torch.Tensor:
    return (
        torch.clamp(tensor, TORCH_BFLOAT16_MIN, TORCH_BFLOAT16_MAX)
        .bfloat16()
        .view(torch.float16)
    )


def measure_fp16_quant_error(input_tensor: torch.Tensor) -> None:
    num_nan_fp32_tensor = torch.numel(input_tensor[torch.isnan(input_tensor)])
    logger.info(
        "num NaN in fp32 tensor: {}, ratio: {}.".format(
            num_nan_fp32_tensor, num_nan_fp32_tensor / torch.numel(input_tensor)
        )
    )

    logger.info(
        "fp32 tensor profile: min: {}, max: {}, min abs:{}, max abs:{}.".format(
            torch.min(input_tensor),
            torch.max(input_tensor),
            torch.min(torch.abs(input_tensor)),
            torch.max(torch.abs(input_tensor)),
        )
    )

    fp16_tensor = fp32_to_fp16_with_clamp(input_tensor)
    num_nan_fp16_tensor = torch.numel(fp16_tensor[torch.isnan(fp16_tensor)])

    logger.info(
        "num NaN in fp16 tensor: {}, ratio: {}.".format(
            num_nan_fp16_tensor, num_nan_fp16_tensor / torch.numel(input_tensor)
        )
    )

    diff = torch.abs(input_tensor - fp16_tensor.float())
    rel_diff = diff / torch.abs(input_tensor)
    logger.info(
        "fp32_to_fp16 abs error: min={}, max={}, avg={}.".format(
            torch.min(diff), torch.max(diff), torch.mean(diff)
        )
    )

    rel_diff_not_nan = rel_diff[torch.logical_not(torch.isnan(rel_diff))]
    logger.info(
        "fp32_to_fp16 rel error: min={}, max={}, avg={}.".format(
            torch.min(rel_diff_not_nan),
            torch.max(rel_diff_not_nan),
            torch.mean(rel_diff_not_nan),
        )
    )

    rel_diff_1_idx = torch.where(rel_diff == 1.0)
    fp32_rel_err_1_vals = input_tensor[rel_diff_1_idx]
    if torch.numel(fp32_rel_err_1_vals) > 0:
        fp32_rel_err_1_vals = torch.abs(fp32_rel_err_1_vals)
        logger.info(
            "fp32_to_fp16 rel error == 1: fp32 min:{}, fp32 max:{}, fp32 avg:{}.".format(
                torch.min(fp32_rel_err_1_vals),
                torch.max(fp32_rel_err_1_vals),
                torch.mean(fp32_rel_err_1_vals),
            )
        )

        subrange_ratio = torch.numel(fp16_tensor[rel_diff_1_idx]) / torch.numel(
            fp16_tensor
        )
        logger.info("sub fp16 range ratio: {}".format(subrange_ratio))
