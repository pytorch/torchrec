#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

INPUT_PATH="$1"
BASE_OUTPUT_PATH="$2"
BATCH_SIZE="$3"
TEMP_PATH=""$BASE_OUTPUT_PATH"temp/"
SRC_DIR=""$BASE_OUTPUT_PATH"criteo_preproc/"
BINARY_OUTPUT_PATH=""$BASE_OUTPUT_PATH"criteo_binary/"
FINAL_OUTPUT_PATH=""$BINARY_OUTPUT_PATH"split"

python convert_tsv_to_parquet.py -i "$INPUT_PATH" -o "$BASE_OUTPUT_PATH"
python process_criteo_parquet.py -b "$BASE_OUTPUT_PATH"
python convert_parquet_to_binary.py --src_dir "$SRC_DIR" \
                                --intermediate_dir  "$TEMP_PATH" \
                                --dst_dir "$BINARY_OUTPUT_PATH"
python split_binary_dataset.py --input_path "$BINARY_OUTPUT_PATH" --output_path "$FINAL_OUTPUT_PATH" --batch_size "$BATCH_SIZE"
