#!/usr/bin/env bash
set -xe
WHEEL_FILE=$1
DEST_DIR=$2

echo 'torchrec_dynamic_embedding-0.0.1-cp37-cp37m-linux_x86_64.whl'

CUDA_SUFFIX=cu$(echo "$CUDA_VERSION" | tr '.' '_')
WHEEL_FILENAME=$(basename "${WHEEL_FILE}")
DEST_FILENAME=$(echo "${WHEEL_FILENAME}" | sed "s#torchrec_dynamic_embedding#torchrec_dynamic_embedding${CUDA_SUFFIX}#g")
mv "${DEST_DIR}/${DEST_FILENAME}" "${WHEEL_FILE}"
