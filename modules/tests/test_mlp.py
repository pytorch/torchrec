#!/usr/bin/env python3

import unittest
from typing import Callable, List, Union

import hypothesis.strategies as st
import torch
from hypothesis import given, settings
from torch import nn
from torchrec.fx import symbolic_trace
from torchrec.modules.mlp import Perceptron, MCPerceptron, MLP, MCMLP
from torchrec.modules.normalization import LayerNorm
from torchrec.modules.utils import extract_module_or_tensor_callable


class TestMLP(unittest.TestCase):
    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    # to decorator factory `hypothesis.given`.
    @given(
        has_bias=st.booleans(),
        activation=st.sampled_from(
            [
                torch.relu,
                torch.tanh,
                torch.sigmoid,
                nn.SiLU(),
            ]
        ),
    )
    def test_perceptron_single_channel(
        self,
        has_bias: bool,
        activation: Union[
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        batch_size = 3

        input_tensors: List[torch.Tensor] = [
            torch.randn(batch_size, 40),  # Task 1
            torch.randn(batch_size, 30),  # Task 2
            torch.randn(batch_size, 20),  # Task 3
            torch.randn(batch_size, 10),  # Task 4
        ]

        perceptron_layer_size = 16
        num_tasks = 4

        perceptron_for_tasks = [
            Perceptron(perceptron_layer_size, bias=has_bias, activation=activation)
            for _ in range(num_tasks)
        ]

        # Dry-run with input of a different batch size
        dry_run_batch_size = 1
        assert dry_run_batch_size != batch_size
        for i in range(num_tasks):
            perceptron_for_tasks[i](
                torch.randn(dry_run_batch_size, input_tensors[i].shape[-1])
            )

        output_tensors = []
        expected_output_tensors = []
        for i in range(len(input_tensors)):
            output_tensors.append(perceptron_for_tasks[i](input_tensors[i]))
            expected_output_tensors.append(
                perceptron_for_tasks[i]._activation_fn(
                    perceptron_for_tasks[i]._linear(input_tensors[i])
                )
            )

        for i in range(len(output_tensors)):
            self.assertEqual(
                list(output_tensors[i].shape), [batch_size, perceptron_layer_size]
            )
            self.assertTrue(
                torch.allclose(output_tensors[i], expected_output_tensors[i])
            )

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    # to decorator factory `hypothesis.given`.
    @given(
        has_bias=st.booleans(),
        activation=st.sampled_from(
            [
                torch.relu,
                torch.tanh,
                torch.sigmoid,
                nn.SiLU(),
            ]
        ),
    )
    @settings(deadline=None)
    def test_perceptron_multi_channel_2d_input(
        self,
        has_bias: bool,
        activation: Union[
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        batch_size = 3
        input_last_dim_size = 40

        input_tensor: torch.Tensor = torch.randn(batch_size, input_last_dim_size)

        perceptron_layer_size = 16
        num_tasks = 4

        multi_channel_perceptron = MCPerceptron(
            perceptron_layer_size, num_tasks, bias=has_bias, activation=activation
        )

        # Dry-run with input of a different batch size
        dry_run_batch_size = 1
        assert dry_run_batch_size != batch_size
        multi_channel_perceptron(
            torch.randn(dry_run_batch_size, input_tensor.shape[-1])
        )

        output_tensor = multi_channel_perceptron(input_tensor)
        expected_output_tensor = multi_channel_perceptron._activation_fn(
            multi_channel_perceptron._linear(input_tensor)
        )
        for i in range(num_tasks):
            self.assertEqual(
                list(output_tensor[i, :, :].shape), [batch_size, perceptron_layer_size]
            )
        self.assertTrue(torch.allclose(output_tensor, expected_output_tensor))

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    # to decorator factory `hypothesis.given`.
    @given(
        has_bias=st.booleans(),
        activation=st.sampled_from(
            [
                torch.relu,
                torch.tanh,
                torch.sigmoid,
                nn.SiLU(),
            ]
        ),
    )
    @settings(deadline=None)
    def test_perceptron_multi_channel_3d_input(
        self,
        has_bias: bool,
        activation: Union[
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        batch_size = 3
        input_last_dim_size = 40
        num_tasks = 4
        input_tensor: torch.Tensor = torch.randn(
            num_tasks, batch_size, input_last_dim_size
        )
        perceptron_layer_size = 16

        multi_channel_perceptron = MCPerceptron(
            perceptron_layer_size, num_tasks, bias=has_bias, activation=activation
        )

        # Dry-run with input of a different batch size
        dry_run_batch_size = 1
        assert dry_run_batch_size != batch_size
        multi_channel_perceptron(
            torch.randn(num_tasks, dry_run_batch_size, input_tensor.shape[-1])
        )

        output_tensor = multi_channel_perceptron(input_tensor)
        expected_output_tensor = multi_channel_perceptron._activation_fn(
            multi_channel_perceptron._linear(input_tensor)
        )
        self.assertEqual(
            list(output_tensor.shape), [num_tasks, batch_size, perceptron_layer_size]
        )
        self.assertTrue(torch.allclose(output_tensor, expected_output_tensor))

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    # to decorator factory `hypothesis.given`.
    @given(
        has_bias=st.booleans(),
        activation=st.sampled_from(
            [
                torch.relu,
                torch.tanh,
                torch.sigmoid,
                nn.SiLU(),
            ]
        ),
    )
    def test_mlp_single_channel(
        self,
        has_bias: bool,
        activation: Union[
            Callable[[], torch.nn.Module],
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        batch_size = 3

        input_tensors: List[torch.Tensor] = [
            torch.randn(batch_size, 40),  # Task 1
            torch.randn(batch_size, 30),  # Task 2
            torch.randn(batch_size, 20),  # Task 3
            torch.randn(batch_size, 10),  # Task 4
        ]

        mlp_layer_sizes = [16, 8, 4]
        num_tasks = 4

        def run_mlp_and_check_output(
            mlp_for_tasks: List[torch.nn.Module],
            input_tensors: List[torch.Tensor],
            batch_size: int,
            last_layer_size: int,
        ) -> None:
            output_tensors = []
            for i in range(len(input_tensors)):
                output_tensors.append(mlp_for_tasks[i](input_tensors[i]))

            for i in range(len(output_tensors)):
                self.assertEqual(
                    list(output_tensors[i].shape), [batch_size, last_layer_size]
                )

        # Case 1: Manually use torchrec.Perceptron modules to build MLP module.
        mlp_for_tasks = [
            nn.Sequential(
                *[
                    Perceptron(
                        layer_size,
                        bias=has_bias,
                        activation=extract_module_or_tensor_callable(activation),
                    )
                    for layer_size in mlp_layer_sizes
                ]
            )
            for _ in range(num_tasks)
        ]
        run_mlp_and_check_output(
            mlp_for_tasks, input_tensors, batch_size, mlp_layer_sizes[-1]
        )

        # Case 2: Use torchrec.MLP module.
        mlp_for_tasks = [
            MLP(mlp_layer_sizes, bias=has_bias, activation=activation) for _ in range(4)
        ]
        run_mlp_and_check_output(
            # pyre-ignore[6]: Expected `List[nn.Module]` for 1st positional only parameter
            # to anonymous call but got `List[MLP]`.
            mlp_for_tasks,
            input_tensors,
            batch_size,
            mlp_layer_sizes[-1],
        )

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    # to decorator factory `hypothesis.given`.
    @given(
        has_bias=st.booleans(),
        activation=st.sampled_from(
            [
                torch.relu,
                torch.tanh,
                torch.sigmoid,
                nn.SiLU(),
            ]
        ),
    )
    @settings(deadline=None)
    def test_mlp_multi_channel_2d_input(
        self,
        has_bias: bool,
        activation: Union[
            Callable[[], torch.nn.Module],
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        batch_size = 3
        input_last_dim_size = 40

        input_tensors: torch.Tensor = torch.randn(batch_size, input_last_dim_size)

        mlp_layer_sizes = [16, 8, 4]
        num_tasks = 4

        def run_mlp_and_check_output(
            multi_channel_mlp: torch.nn.Module,
            input_tensor: torch.Tensor,
            num_tasks: int,
            batch_size: int,
            last_layer_size: int,
        ) -> None:
            output_tensor = multi_channel_mlp(input_tensor)
            for i in range(num_tasks):
                self.assertEqual(
                    list(output_tensor[i, :, :].shape), [batch_size, last_layer_size]
                )

        # Case 1: Manually use torchrec.MCPerceptron modules to build multi-channel MLP module.
        multi_channel_mlp = nn.Sequential(
            *[
                MCPerceptron(
                    layer_size,
                    num_tasks,
                    bias=has_bias,
                    activation=extract_module_or_tensor_callable(activation),
                )
                for layer_size in mlp_layer_sizes
            ],
        )
        run_mlp_and_check_output(
            multi_channel_mlp, input_tensors, num_tasks, batch_size, mlp_layer_sizes[-1]
        )

        # Case 2: Use torchrec.MCMLP module.
        multi_channel_mlp = MCMLP(
            mlp_layer_sizes, num_tasks, bias=has_bias, activation=activation
        )
        run_mlp_and_check_output(
            multi_channel_mlp, input_tensors, num_tasks, batch_size, mlp_layer_sizes[-1]
        )

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    # to decorator factory `hypothesis.given`.
    @given(
        has_bias=st.booleans(),
        activation=st.sampled_from(
            [
                torch.relu,
                torch.tanh,
                torch.sigmoid,
                nn.SiLU(),
            ]
        ),
    )
    @settings(deadline=None)
    def test_mlp_multi_channel_3d_input(
        self,
        has_bias: bool,
        activation: Union[
            Callable[[], torch.nn.Module],
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        batch_size = 3
        input: torch.Tensor = torch.randn(4, batch_size, 40)

        mlp_layer_sizes = [16, 8, 4]
        num_tasks = 4

        def run_mlp_and_check_output(
            multi_channel_mlp: torch.nn.Module,
            input: torch.Tensor,
            batch_size: int,
            last_layer_size: int,
        ) -> None:
            output_tensors = multi_channel_mlp(input)
            for i in range(len(output_tensors)):
                self.assertEqual(
                    list(output_tensors[i].shape), [batch_size, last_layer_size]
                )

        multi_channel_mlp = MCMLP(
            mlp_layer_sizes, num_tasks, bias=has_bias, activation=activation
        )
        run_mlp_and_check_output(
            multi_channel_mlp, input, batch_size, mlp_layer_sizes[-1]
        )

    def test_fx_script_Perceptron(self) -> None:
        batch_size = 1
        in_features = 3
        out_features = 5
        m = Perceptron(out_features)

        # Dry-run to initialize lazy module.
        m(torch.randn(batch_size, in_features))

        gm = symbolic_trace(m)
        torch.jit.script(gm)

    def test_fx_script_MCPerceptron(self) -> None:
        batch_size = 1
        in_features = 3
        out_features = 5
        num_channels = 4
        m = MCPerceptron(out_features, num_channels)

        # Dry-run to initialize lazy module.
        m(torch.randn(num_channels, batch_size, in_features))

        gm = symbolic_trace(m)
        torch.jit.script(gm)

    def test_fx_script_MLP(self) -> None:
        batch_size = 1
        in_features = 3
        layer_sizes = [16, 8, 4]
        m = MLP(layer_sizes)

        # Dry-run to initialize lazy module.
        m(torch.randn(batch_size, in_features))

        gm = symbolic_trace(m)
        torch.jit.script(gm)

    def test_fx_script_MCMLP(self) -> None:
        batch_size = 1
        in_features = 3
        layer_sizes = [16, 8, 4]
        num_channels = 4
        m = MCMLP(layer_sizes, num_channels)

        # Dry-run to initialize lazy module.
        m(torch.randn(num_channels, batch_size, in_features))

        gm = symbolic_trace(m)
        torch.jit.script(gm)


if __name__ == "__main__":
    unittest.main()
