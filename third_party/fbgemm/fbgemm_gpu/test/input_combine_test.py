#!/usr/bin/env python3

# pyre-unsafe

from typing import List, Optional, Tuple
import unittest

import torch

try:
    torch.ops.load_library("fbgemm_gpu_py.so")
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:input_combine_cpu")


class TBEInputPrepareReference(torch.nn.Module):
    def __init__(self, include_last_offsets: List[bool]):
        super().__init__()
        self.include_last_offsets = include_last_offsets

    def forward(
        self,
        indices_list: List[torch.Tensor],
        offsets_list: List[torch.Tensor],
        per_sample_weights_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        size = 0
        assert len(indices_list) > 0
        assert len(indices_list) == len(offsets_list)
        assert len(indices_list) == len(per_sample_weights_list)
        assert len(indices_list) == len(self.include_last_offsets)
        for i in range(len(self.include_last_offsets)):
            size += indices_list[i].size(0)
            assert indices_list[i].dim() == 1
            assert offsets_list[i].dim() == 1
            if per_sample_weights_list[i].numel() > 0:
                assert per_sample_weights_list[i].dim() == 1
                assert indices_list[i].numel() == per_sample_weights_list[i].numel()
        combined_indices = torch.empty(
            size,
            dtype=torch.int32,
            device=indices_list[0].device,
        )
        torch.cat(indices_list, out=combined_indices)
        offsets_starts = torch.zeros(
            [len(offsets_list) + 1],
            dtype=offsets_list[0].dtype,
            device=offsets_list[0].device,
        )
        offsets_accs = torch.zeros(
            [len(offsets_list) + 1],
            dtype=offsets_list[0].dtype,
            device=offsets_list[0].device,
        )

        for i, include_last_offset in enumerate(self.include_last_offsets):
            if include_last_offset:
                offsets_starts[i + 1] = offsets_starts[i] + offsets_list[i].size(0) - 1
            else:
                offsets_starts[i + 1] = offsets_starts[i] + offsets_list[i].size(0)
            offsets_accs[i + 1] = offsets_accs[i] + indices_list[i].size(0)

        assert offsets_accs[-1] == combined_indices.size(0)
        combined_offsets_size: List[int] = [int(offsets_starts[-1].item()) + 1]
        combined_offsets = torch.zeros(
            combined_offsets_size,
            dtype=torch.int32,
            device=offsets_list[0].device,
        )
        for i in range(len(self.include_last_offsets)):
            combined_offsets[offsets_starts[i] : offsets_starts[i + 1]] = (
                offsets_list[i][: offsets_starts[i + 1] - offsets_starts[i]]
                + offsets_accs[i]
            )
        combined_offsets[-1] = offsets_accs[-1]
        per_sample_weights: Optional[torch.Tensor] = None
        for i in range(len(self.include_last_offsets)):
            if per_sample_weights_list[i].size(0) > 0:
                per_sample_weights = torch.ones(
                    combined_indices.size(0),
                    dtype=per_sample_weights_list[i].dtype,
                    device=per_sample_weights_list[i].device,
                )
                break
        if per_sample_weights is not None:
            for i in range(len(self.include_last_offsets)):
                if per_sample_weights_list[i].size(0) > 0:
                    per_sample_weights[
                        offsets_accs[i] : offsets_accs[i + 1]
                    ] = per_sample_weights_list[i][:]

        # indices and offsets are required to be int32 for TBE
        return combined_indices, combined_offsets, per_sample_weights


class InputCombineTest(unittest.TestCase):
    def _get_inputs(self, dtypes):
        indices_list = [
            torch.tensor([1, 2, 3], dtype=dtypes[0]),
            torch.tensor([1, 2, 3, 4], dtype=dtypes[1]),
        ]
        offsets_list = [
            torch.tensor([0, 2], dtype=dtypes[0]),
            torch.tensor([0, 1, 4], dtype=dtypes[1]),
        ]
        include_last_offsets = [False, True]
        per_sample_weights = [
            torch.tensor([1, 2, 1], dtype=torch.float),
            torch.tensor([1, 2, 1, 3], dtype=torch.float),
        ]
        empty_per_sample_weights = [
            torch.tensor([], dtype=torch.float),
            torch.tensor([], dtype=torch.float),
        ]
        return indices_list, offsets_list, per_sample_weights, empty_per_sample_weights, include_last_offsets

    def _run_test(self, dtypes) -> None:
        indices_list, offsets_list, per_sample_weights, empty_per_sample_weights, include_last_offsets = self._get_inputs(dtypes)
        ref_mod = TBEInputPrepareReference(include_last_offsets)

        outputs = torch.ops.fbgemm.tbe_input_combine(indices_list, offsets_list, per_sample_weights, torch.BoolTensor(include_last_offsets))
        ref_outputs = ref_mod(indices_list, offsets_list, per_sample_weights)
        for i, j in zip(outputs, ref_outputs):
            torch.testing.assert_allclose(i, j)
        self.assertTrue(outputs[0].dtype == torch.int32)
        self.assertTrue(outputs[1].dtype == torch.int32)


        outputs = torch.ops.fbgemm.tbe_input_combine(indices_list, offsets_list, empty_per_sample_weights, torch.BoolTensor(include_last_offsets))
        ref_outputs = ref_mod(indices_list, offsets_list, empty_per_sample_weights)
        for i, j in zip(outputs[:-1], ref_outputs[:-1]):
            torch.testing.assert_allclose(i, j)
            self.assertTrue(j.dtype == torch.int32)

        self.assertTrue(outputs[0].dtype == torch.int32)
        self.assertTrue(outputs[1].dtype == torch.int32)
        self.assertTrue(outputs[-1].size(0) == 0)

    def test_input_combine_int64(self):
        self._run_test((torch.int64, torch.int64))

    def test_input_combine_int32(self):
        self._run_test((torch.int64, torch.int64))

    def test_input_combined_mix(self):
        self._run_test((torch.int64, torch.int32))


if __name__ == "__main__":
    unittest.main()
