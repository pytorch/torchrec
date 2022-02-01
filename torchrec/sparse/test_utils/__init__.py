#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def keyed_jagged_tensor_equals(
    kjt1: Optional[KeyedJaggedTensor], kjt2: Optional[KeyedJaggedTensor]
) -> bool:
    def _tensor_eq_or_none(
        t1: Optional[torch.Tensor], t2: Optional[torch.Tensor]
    ) -> bool:
        if t1 is None and t2 is None:
            return True
        elif t1 is None and t2 is not None:
            return False
        elif t1 is not None and t2 is None:
            return False
        else:
            assert t1 is not None
            assert t2 is not None
            return torch.equal(t1, t2) and t1.dtype == t2.dtype

    if kjt1 is None and kjt2 is None:
        return True
    elif kjt1 is None and kjt2 is not None:
        return False
    elif kjt1 is not None and kjt2 is None:
        return False
    else:
        assert kjt1 is not None
        assert kjt2 is not None
        return (
            kjt1.keys() == kjt2.keys()
            and _tensor_eq_or_none(kjt1.lengths(), kjt2.lengths())
            and _tensor_eq_or_none(kjt1.values(), kjt2.values())
            and _tensor_eq_or_none(kjt1._weights, kjt2._weights)
        )
