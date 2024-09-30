#!/usr/bin/env bash
set -xe
WHEEL_FILE=$1
DEST_DIR=$2

CUDA_SUFFIX=cu$(echo "$CUDA_VERSION" | sed 's#\.##g')
WHEEL_FILENAME=$(basename "${WHEEL_FILE}")
DEST_FILENAME=$(echo "${WHEEL_FILENAME}" | sed -r 's#(torchrec_dynamic_embedding-[0-9]+\.[0-9]+\.[0-9]+)#\1'"+${CUDA_SUFFIX}#g")
mv  "${WHEEL_FILE}" "${DEST_DIR}/${DEST_FILENAME}"
