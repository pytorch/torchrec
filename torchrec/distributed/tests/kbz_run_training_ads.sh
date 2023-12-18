#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

WORKDIR="/var/tmp/torchrec-pt2"
mkdir -p "$WORKDIR"

LOGFILE="$WORKDIR/log-torchrec-training-ads-$(date +"%Y%m%dT%H%M").log"

pushd "$HOME/fbsource/fbcode" || return

buck2 run @//mode/opt //aps_models/ads/icvr:icvr_launcher \
  -- \
  mode=mast_ctr_cvr_cmf_rep launcher.fbl_entitlement=ads_global_qps \
  features=ctr_cvr_conso_cmf_pipeline_features_455876776_3teach \
  model=ctr_cvr_cmf_when_rep_config_msmn_3teach \
  model_name=ctr_cvr_when model.when_arch.layer_configurations=[1,1,1,1,1,1,1,1,1,1,1,1,1] \
  model.when_arch.use_extended_residual_contexts=True \
  optimizers.dense_default.lr_schedule.0.max_iters=20000 \
  training.planner.storage_reservation_policy=FixedPercentage training.planner.storage_reservation_percentage=0.72 \
  data_loader.dataset.batch_size=2048 \
  trainer.garbage_collection.garbage_collection_interval=100 \
  model.when_arch.layer_norm_init_weight=0.3 \
  optimizers.dense_default.lr_schedule.0.value=0.001 \
  model.when_arch.customized_mlp_init_scale=0.3 \
  launcher.num_workers=128 \
  launcher.max_retries=10 \
  launcher.data_project=oncall_ads_model_platform \
  launcher.hardware=ZIONEX_80G \
  data_loader.dataset.table_ds=[2023-10-01,2023-10-02,2023-10-03,2023-10-04,2023-10-05]


popd || return

echo "LOGFILE=$LOGFILE"
