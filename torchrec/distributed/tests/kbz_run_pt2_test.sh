#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

WORKDIR="/var/tmp/torchrec-pt2"
mkdir -p "$WORKDIR"

LOGFILE="$WORKDIR/log-test-sigmoid-$(date +"%Y%m%dT%H%M").log"

pushd "$HOME/fbsource/fbcode" || return

TORCH_LOGS="+dynamo,output_code,graph_code,bytecode,dynamic" \
TORCHDYNAMO_VERBOSE=1 \
TORCHINDUCTOR_MAX_AUTOTUNE=1 \
TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 \
TORCH_COMPILE_DEBUG=1 \
TORCH_SHOW_DISPATCH_TRACE=1 \
buck2 run @mode/dev-nosan torchrec/distributed/tests:test_pt2 "$@" 2>&1 | tee "$LOGFILE"

popd || return

echo "LOGFILE=$LOGFILE"
