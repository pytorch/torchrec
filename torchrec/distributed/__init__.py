#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchrec.distributed.comm import (  # noqa
    get_local_rank,
    get_local_size,
)
from torchrec.distributed.model_parallel import DistributedModelParallel  # noqa
from torchrec.distributed.train_pipeline import (  # noqa
    TrainPipeline,
    TrainPipelineBase,
    TrainPipelineSparseDist,
)
from torchrec.distributed.types import (  # noqa
    Awaitable,
    NoWait,
    ParameterSharding,
    ModuleSharder,
    ShardingPlanner,
    ShardedModule,
    ShardedTensor,
    ShardingEnv,
)
from torchrec.distributed.utils import (  # noqa
    get_unsharded_module_names,
    sharded_model_copy,
)
