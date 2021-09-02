#!/usr/bin/env python3

from typing import List, Union

import torch
import torch.fx
from torchrec.modules.lazy_extension import LazyModuleExtensionMixin


class PadCat(LazyModuleExtensionMixin, torch.nn.Module):
    """
    PadCat first zero-pads a specified dimension of all input tensors so that
    all tensors have the same size in that dimension, and then it concatenates
    all tensors along a different specified dimension.

    For example:

    1 x 5, 1 x 3, 1 x 2 (shapes of input tensors)
             |
             |          (zero-pad on 2nd dim)
             v
    1 x 5, 1 x 5, 1 x 5
             |
             |          (concat on 1st dim)
             v
           3 x 5

    Constructor Args:
        cat_dim (int): the dimension along which we concatenate all input tensors.
        pad_dim (int): the dimension along which we zero-pad all input tensors so that
            they have the same size in that dimension.

    Call Args:
        tensors (List[torch.Tensor]): the input tensors.

    Returns:
        output_tensor (torch.Tensor): result of zero-padding and concatenating the input tensors.

    Example:
        >>> input_tensors: List[torch.Tensor] = [
        >>>     torch.tensor([[1, 2, 3, 4, 5]]),
        >>>     torch.tensor([[1, 2, 3]]),
        >>>     torch.tensor([[1, 2]]),
        >>> ]
        >>> concated_tensor = PadCat(cat_dim=0, pad_dim=1)(input_tensors)
        >>> assert torch.allclose(concated_tensor[0], torch.tensor([1, 2, 3, 4, 5]))
        >>> assert torch.allclose(concated_tensor[1], torch.tensor([1, 2, 3, 0, 0]))
        >>> assert torch.allclose(concated_tensor[2], torch.tensor([1, 2, 0, 0, 0]))
    """

    def __init__(
        self,
        cat_dim: int,
        pad_dim: int,
    ) -> None:
        super().__init__()
        assert cat_dim != pad_dim, "`cat_dim` and `pad_dim` can't be equal."
        self._cat_dim: int = cat_dim
        self._pad_dim: int = pad_dim
        self._initialized: bool = False
        self._num_tensors: int = -1
        self._pad_dim_sizes: List[int] = []
        self._cat_dim_sizes: List[int] = []
        self._cat_dim_sizes_cumsum: torch.Tensor = torch.empty(0)
        self._output_tensor_shape: List[int] = []

    # pyre-ignore[14]: `torchrec.modules.concat.PadCat.initialize_parameters`
    # overrides method defined in `LazyModuleExtensionMixin` inconsistently.
    def initialize_parameters(self, tensors: List[torch.Tensor]) -> None:
        if self._initialized:
            return

        if self._num_tensors == -1:
            self._num_tensors = len(tensors)

        for i in range(self._num_tensors):
            self._pad_dim_sizes.append(tensors[i].size(self._pad_dim))
            self._cat_dim_sizes.append(tensors[i].size(self._cat_dim))
        self._cat_dim_sizes_cumsum = torch.cumsum(
            torch.tensor(self._cat_dim_sizes), dim=0
        )
        self._output_tensor_shape = list(tensors[0].shape)
        self._output_tensor_shape[self._pad_dim] = max(self._pad_dim_sizes)
        self._output_tensor_shape[self._cat_dim] = self._cat_dim_sizes_cumsum[-1].item()
        self._initialized = True

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        output_tensor: torch.Tensor = (
            torch.empty_like(tensors[0]).resize_(self._output_tensor_shape).fill_(0)
        )
        for i in range(self._num_tensors):
            # pyre-ignore[16]: `torch.Tensor` has no attribute `narrow`.
            output_tensor.narrow(
                self._cat_dim,
                self._cat_dim_sizes_cumsum[i - 1].item() if i > 0 else 0,
                self._cat_dim_sizes[i],
            ).narrow(self._pad_dim, 0, self._pad_dim_sizes[i]).copy_(
                tensors[i].narrow(self._pad_dim, 0, self._pad_dim_sizes[i])
            )
        return output_tensor


class Split(torch.nn.Module):
    """
    Splits the tensor into chunks. Each chunk is a view of the original tensor.

    For more details, please see documentation of `torch.split`:
    https://pytorch.org/docs/stable/generated/torch.split.html#torch-split

    Constructor Args:
        split_size_or_sections (Union[int, List[int]]): size of a single chunk or
            list of sizes for each chunk.
        dim (int): dimension along which to split the tensor. Defaults: 0.

    Call Args:
        input (torch.Tensor): tensor to split.

    Returns:
        output_tensors (List[torch.Tensor]): result of splitting the input tensor.

    Example:
        >>> input_tensors: List[torch.Tensor] = [
        >>>     torch.tensor([[1, 2, 3, 4, 5]]),
        >>>     torch.tensor([[1, 2, 3]]),
        >>>     torch.tensor([[1, 2]]),
        >>> ]
        >>> cat_dim = 0
        >>> pad_dim = 1
        >>>
        >>> concated_tensor = PadCat(cat_dim=cat_dim, pad_dim=pad_dim)(input_tensors)
        >>> output_tensors = Split([t.size(cat_dim) for t in input_tensors], dim=cat_dim)(
        >>>     concated_tensor
        >>> )
        >>> assert torch.allclose(output_tensors[0], torch.tensor([1, 2, 3, 4, 5]))
        >>> assert torch.allclose(output_tensors[1], torch.tensor([1, 2, 3, 0, 0]))
        >>> assert torch.allclose(output_tensors[2], torch.tensor([1, 2, 0, 0, 0]))
    """

    def __init__(
        self,
        split_size_or_sections: Union[int, List[int]],
        dim: int = 0,
    ) -> None:
        super().__init__()
        self._split_size_or_sections: Union[int, List[int]] = split_size_or_sections
        self._dim: int = dim

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        return torch.split(input, self._split_size_or_sections, dim=self._dim)
