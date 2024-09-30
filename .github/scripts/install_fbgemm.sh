#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

echo "CU_VERSION"
echo "$CU_VERSION"

echo "CHANNEL"
echo "$CHANNEL"

if [ "$CHANNEL" = "nightly" ]; then
    ${CONDA_RUN} pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/"$CU_VERSION"
elif [ "$CHANNEL" = "test" ]; then
    ${CONDA_RUN} pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/test/"$CU_VERSION"
fi
