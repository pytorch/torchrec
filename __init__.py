#!/usr/bin/env python3

import torchrec.distributed  # noqa
from torchrec.fx import tracer  # noqa
from torchrec.modules.embedding_configs import (  # noqa
    EmbeddingBagConfig,
    EmbeddingConfig,
    DataType,
    PoolingType,
)
from torchrec.modules.embedding_modules import (  # noqa
    EmbeddingBagCollection,
    EmbeddingCollection,
)  # noqa
from torchrec.modules.score_learning import PositionWeightsAttacher  # noqa
from torchrec.modules.sparsenn import SimpleSparseNN, SparseArch  # noqa
from torchrec.sparse.jagged_tensor import (  # noqa
    JaggedTensor,
    KeyedJaggedTensor,
    KeyedTensor,
)
