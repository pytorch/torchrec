# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List

import torch

try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "tde_cpp.so"))
except Exception as ex:
    print(f"File tde_cpp.so not found {ex}")


__all__ = []


class TensorList:
    def __init__(self, tensors: List[torch.Tensor]):
        self.tensor_list = torch.classes.tde.TensorList()
        for tensor in tensors:
            # tensor.data will allow inplace ops during autograd.
            # https://discuss.pytorch.org/t/disable-in-place-correctness-version-check-any-other-workaround/90738/2
            self.tensor_list.append(tensor.data)

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, i):
        return self.tensor_list[i]
