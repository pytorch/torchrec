# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import struct
import math
from typing import Callable, List, Tuple

import hypothesis.strategies as st
import numpy as np
import torch

# Eigen/Python round 0.5 away from 0, Numpy rounds to even
round_to_nearest: Callable[[np.ndarray], np.ndarray] = np.vectorize(round)


def bytes_to_floats(byte_matrix: np.ndarray) -> np.ndarray:
    floats = np.empty([np.shape(byte_matrix)[0], 1], dtype=np.float32)
    for i, byte_values in enumerate(byte_matrix):
        (floats[i],) = struct.unpack("f", bytearray(byte_values))
    return floats


def floats_to_bytes(floats: np.ndarray) -> np.ndarray:
    byte_matrix = np.empty([np.shape(floats)[0], 4], dtype=np.uint8)
    for i, value in enumerate(floats):
        assert isinstance(value, np.float32), (value, floats)
        as_bytes = struct.pack("f", value)
        # In Python3 bytes will be a list of int, in Python2 a list of string
        if isinstance(as_bytes[0], int):
            byte_matrix[i] = list(as_bytes)
        else:
            byte_matrix[i] = list(map(ord, as_bytes))
    return byte_matrix


def bytes_to_half_floats(byte_matrix: np.ndarray) -> np.ndarray:
    floats = np.empty([np.shape(byte_matrix)[0], 1], dtype=np.float16)
    for i, byte_values in enumerate(byte_matrix):
        (floats[i],) = np.frombuffer(
            memoryview(byte_values).tobytes(), dtype=np.float16
        )
    return floats


def half_floats_to_bytes(floats: np.ndarray) -> np.ndarray:
    byte_matrix = np.empty([np.shape(floats)[0], 2], dtype=np.uint8)
    for i, value in enumerate(floats):
        assert isinstance(value, np.float16), (value, floats)
        byte_matrix[i] = np.frombuffer(
            memoryview(value.tobytes()).tobytes(), dtype=np.uint8
        )
    return byte_matrix


def fused_rowwise_8bit_quantize_reference(data: np.ndarray) -> np.ndarray:
    minimum = np.min(data, axis=-1, keepdims=True)
    maximum = np.max(data, axis=-1, keepdims=True)
    span = maximum - minimum
    bias = minimum
    scale = span / 255.0
    inverse_scale = 255.0 / (span + 1e-8)
    quantized_data = round_to_nearest((data - bias) * inverse_scale)
    scale_bytes = floats_to_bytes(scale.reshape(-1))
    scale_bytes = scale_bytes.reshape(data.shape[:-1] + (scale_bytes.shape[-1],))
    bias_bytes = floats_to_bytes(bias.reshape(-1))
    bias_bytes = bias_bytes.reshape(data.shape[:-1] + (bias_bytes.shape[-1],))
    return np.concatenate([quantized_data, scale_bytes, bias_bytes], axis=-1)


def fused_rowwise_8bit_dequantize_reference(fused_quantized: np.ndarray) -> np.ndarray:
    scale = bytes_to_floats(fused_quantized[..., -8:-4].astype(np.uint8).reshape(-1, 4))
    scale = scale.reshape(fused_quantized.shape[:-1] + (scale.shape[-1],))
    bias = bytes_to_floats(fused_quantized[..., -4:].astype(np.uint8).reshape(-1, 4))
    bias = bias.reshape(fused_quantized.shape[:-1] + (bias.shape[-1],))
    quantized_data = fused_quantized[..., :-8]
    return quantized_data * scale + bias


def fused_rowwise_nbit_quantize_reference(data: np.ndarray, bit: int) -> np.ndarray:
    minimum = np.min(data, axis=1).astype(np.float16).astype(np.float32)
    maximum = np.max(data, axis=1)
    span = maximum - minimum
    qmax = (1 << bit) - 1
    scale = (span / qmax).astype(np.float16).astype(np.float32)
    bias = np.zeros(data.shape[0])
    quantized_data = np.zeros(data.shape).astype(np.uint8)

    for i in range(data.shape[0]):
        bias[i] = minimum[i]
        inverse_scale = 1.0 if scale[i] == 0.0 else 1 / scale[i]
        if scale[i] == 0.0 or math.isinf(inverse_scale):
            scale[i] = 1.0
            inverse_scale = 1.0
        quantized_data[i] = np.clip(
            np.round((data[i, :] - minimum[i]) * inverse_scale), 0, qmax
        )

    # pack
    assert 8 % bit == 0
    num_elem_per_byte = 8 // bit
    packed_dim = (data.shape[1] + num_elem_per_byte - 1) // num_elem_per_byte
    packed_data = np.zeros([data.shape[0], packed_dim]).astype(np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if j % num_elem_per_byte == 0:
                packed_data[i, j // num_elem_per_byte] = quantized_data[i, j]
            else:
                packed_data[i, j // num_elem_per_byte] += quantized_data[i, j] << (
                    (j % num_elem_per_byte) * bit
                )

    scale_bytes = half_floats_to_bytes(scale.astype(np.float16))
    bias_bytes = half_floats_to_bytes(bias.astype(np.float16))
    return np.concatenate([packed_data, scale_bytes, bias_bytes], axis=1)


def fused_rowwise_nbit_quantize_dequantize_reference(data: np.ndarray, bit: int) -> np.ndarray:
    fused_quantized = fused_rowwise_nbit_quantize_reference(data, bit)
    scale = bytes_to_half_floats(fused_quantized[:, -4:-2].astype(np.uint8)).astype(
        np.float32
    )
    bias = bytes_to_half_floats(fused_quantized[:, -2:].astype(np.uint8)).astype(
        np.float32
    )
    quantized_data = fused_quantized[:, :-4]

    # unpack
    packed_dim = fused_quantized.shape[1] - 4
    assert 8 % bit == 0
    num_elem_per_byte = 8 // bit
    assert packed_dim == ((data.shape[1] + num_elem_per_byte - 1) // num_elem_per_byte)
    unpacked_data = np.zeros(data.shape).astype(np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            unpacked_data[i, j] = (
                quantized_data[i, j // num_elem_per_byte]
                >> ((j % num_elem_per_byte) * bit)
            ) & ((1 << bit) - 1)

    return scale * unpacked_data + bias


# Used for `@unittest.skipIf`
gpu_unavailable: Tuple[bool, str] = (
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    "CUDA is not available or no GPUs detected",
)
# Used for `if` statements inside tests
gpu_available: bool = not gpu_unavailable[0]


def cpu_and_maybe_gpu() -> st.SearchStrategy[List[torch.device]]:
    gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    # st.sampled_from is not guaranteed to test all the values passed to it.
    # However, Hypothesis, by default, generates 100 test cases from the specified strategies.
    # If st.sampled_from contains >100 items or if it's used in conjunction with other strategies
    # then it may not test all values; however, for smaller tests it may work fine.
    # This is still a stopgap solution until we figure out a way to parameterize UnitTestCase.
    return st.sampled_from(
        [torch.device("cpu")] + ([torch.device("cuda")] if gpu_available else [])
    )


def cpu_only() -> st.SearchStrategy[List[torch.device]]:
    return st.sampled_from([torch.device("cpu")])
