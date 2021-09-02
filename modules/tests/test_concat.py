#!/usr/bin/env python3

import unittest
from typing import List

import torch
from torchrec.fx import symbolic_trace
from torchrec.modules.concat import PadCat, Split


class TestConcat(unittest.TestCase):
    def test_concat_and_split_2d_cat_dim_size_1(self) -> None:
        """
        1 x 5, 1 x 3, 1 x 2 (`input_tensors`)
                 |
                 |          (pad on 2nd dim)
                 v
        1 x 5, 1 x 5, 1 x 5
                 |
                 |          (concat on 1st dim)
                 v
               3 x 5
                 |
                 |          (split on 1st dim)
                 v
        1 x 5, 1 x 5, 1 x 5
        """

        input_tensors: List[torch.Tensor] = [
            torch.tensor([[1, 2, 3, 4, 5]]),
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[1, 2]]),
        ]
        cat_dim = 0
        pad_dim = 1

        concated_tensor = PadCat(cat_dim=cat_dim, pad_dim=pad_dim)(input_tensors)
        self.assertTrue(
            torch.allclose(concated_tensor[0], torch.tensor([1, 2, 3, 4, 5])),
        )
        self.assertTrue(
            torch.allclose(concated_tensor[1], torch.tensor([1, 2, 3, 0, 0])),
        )
        self.assertTrue(
            torch.allclose(concated_tensor[2], torch.tensor([1, 2, 0, 0, 0])),
        )

        output_tensors = Split([t.size(cat_dim) for t in input_tensors], dim=cat_dim)(
            concated_tensor
        )
        self.assertTrue(len(output_tensors) == len(input_tensors))

        self.assertTrue(
            torch.allclose(output_tensors[0], torch.tensor([1, 2, 3, 4, 5]))
        )
        self.assertTrue(
            torch.allclose(output_tensors[1], torch.tensor([1, 2, 3, 0, 0]))
        )
        self.assertTrue(
            torch.allclose(output_tensors[2], torch.tensor([1, 2, 0, 0, 0]))
        )

    def test_concat_and_split_2d_cat_dim_size_2(self) -> None:
        """
        2 x 5, 2 x 3, 2 x 2 (`input_tensors`)
                 |
                 |          (pad on 2nd dim)
                 v
        2 x 5, 2 x 5, 2 x 5
                 |
                 |          (concat on 1st dim)
                 v
               6 x 5
                 |
                 |          (split on 1st dim)
                 v
        2 x 5, 2 x 5, 2 x 5
        """

        input_tensors: List[torch.Tensor] = [
            torch.arange(1, 2 * 5 + 1).view(2, 5),
            torch.arange(1, 2 * 3 + 1).view(2, 3),
            torch.arange(1, 2 * 2 + 1).view(2, 2),
        ]
        cat_dim = 0
        pad_dim = 1

        concated_tensor = PadCat(cat_dim=cat_dim, pad_dim=pad_dim)(input_tensors)
        self.assertTrue(
            torch.allclose(
                concated_tensor.narrow(cat_dim, 0, 2),
                torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            ),
        )
        self.assertTrue(
            torch.allclose(
                concated_tensor.narrow(cat_dim, 2, 2),
                torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]]),
            ),
        )
        self.assertTrue(
            torch.allclose(
                concated_tensor.narrow(cat_dim, 4, 2),
                torch.tensor([[1, 2, 0, 0, 0], [3, 4, 0, 0, 0]]),
            ),
        )

        output_tensors = Split(
            [t.size(cat_dim) for t in input_tensors],
            dim=cat_dim,
        )(concated_tensor)
        self.assertTrue(len(output_tensors) == len(input_tensors))

        self.assertTrue(
            torch.allclose(
                output_tensors[0], torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            ),
        )
        self.assertTrue(
            torch.allclose(
                output_tensors[1], torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]])
            ),
        )
        self.assertTrue(
            torch.allclose(
                output_tensors[2], torch.tensor([[1, 2, 0, 0, 0], [3, 4, 0, 0, 0]])
            ),
        )

    def test_concat_and_split_3d(self) -> None:
        """
        2 x 2 x 5, 2 x 2 x 3, 2 x 2 x 2 (`input_tensors`)
                       |
                       |                (pad on 3nd dim)
                       v
        2 x 2 x 5, 2 x 2 x 5, 2 x 2 x 5
                       |
                       |                (concat on 1st dim)
                       v
                   6 x 2 x 5
                       |
                       |                (split on 1st dim)
                       v
        2 x 2 x 5, 2 x 2 x 5, 2 x 2 x 5
        """

        input_tensors: List[torch.Tensor] = [
            torch.arange(1, 2 * 2 * 5 + 1).view(2, 2, 5),
            torch.arange(1, 2 * 2 * 3 + 1).view(2, 2, 3),
            torch.arange(1, 2 * 2 * 2 + 1).view(2, 2, 2),
        ]
        cat_dim = 0
        pad_dim = 2

        concated_tensor = PadCat(cat_dim=cat_dim, pad_dim=pad_dim)(input_tensors)
        self.assertTrue(
            torch.allclose(
                concated_tensor.narrow(cat_dim, 0, 2),
                torch.tensor(
                    [
                        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                        [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                    ]
                ),
            ),
            "tensor: {}".format(concated_tensor.narrow(cat_dim, 0, 2)),
        )
        self.assertTrue(
            torch.allclose(
                concated_tensor.narrow(cat_dim, 2, 2),
                torch.tensor(
                    [
                        [[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]],
                        [[7, 8, 9, 0, 0], [10, 11, 12, 0, 0]],
                    ]
                ),
            ),
            "tensor: {}".format(concated_tensor.narrow(cat_dim, 2, 2)),
        )
        self.assertTrue(
            torch.allclose(
                concated_tensor.narrow(cat_dim, 4, 2),
                torch.tensor(
                    [
                        [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0]],
                        [[5, 6, 0, 0, 0], [7, 8, 0, 0, 0]],
                    ]
                ),
            ),
            "tensor: {}".format(concated_tensor.narrow(cat_dim, 4, 2)),
        )

        output_tensors = Split(
            [t.size(cat_dim) for t in input_tensors],
            dim=cat_dim,
        )(concated_tensor)
        self.assertTrue(len(output_tensors) == len(input_tensors))

        self.assertTrue(
            torch.allclose(
                output_tensors[0],
                torch.tensor(
                    [
                        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                        [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                    ]
                ),
            ),
            "tensor: {}".format(output_tensors[0]),
        )
        self.assertTrue(
            torch.allclose(
                output_tensors[1],
                torch.tensor(
                    [
                        [[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]],
                        [[7, 8, 9, 0, 0], [10, 11, 12, 0, 0]],
                    ]
                ),
            ),
            "tensor: {}".format(output_tensors[1]),
        )
        self.assertTrue(
            torch.allclose(
                output_tensors[2],
                torch.tensor(
                    [
                        [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0]],
                        [[5, 6, 0, 0, 0], [7, 8, 0, 0, 0]],
                    ]
                ),
            ),
            "tensor: {}".format(output_tensors[2]),
        )

    def test_fx_script_PadCat(self) -> None:
        m = PadCat(0, 1)

        # Dry-run to initialize lazy module.
        m([torch.arange(1, 5 + 1).view(1, 5), torch.arange(1, 3 + 1).view(1, 3)])

        gm = symbolic_trace(m)
        torch.jit.script(gm)

    def test_fx_script_Split(self) -> None:
        m = Split(1)
        gm = symbolic_trace(m)
        torch.jit.script(gm)


if __name__ == "__main__":
    unittest.main()
