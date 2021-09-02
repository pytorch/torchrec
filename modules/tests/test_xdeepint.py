#!/usr/bin/env python3
# pyre-strict

import unittest
from typing import Callable, Optional, Union

import hypothesis.strategies as st
import torch
from hypothesis import given
from torchrec.modules.xdeepint import PINLayer, XdeepInt


class TestXdeepInt(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.sampled_from([torch.nn.ReLU(), torch.tanh, torch.sigmoid,
    #  None])` to decorator factory `hypothesis.given`.
    @given(
        activation=st.sampled_from(
            [
                torch.nn.ReLU(),
                torch.tanh,
                torch.sigmoid,
                None,
            ]
        ),
        dropout=st.one_of(st.none(), st.floats(0.0, 1.0, exclude_max=True)),
    )
    def test_pinlayer(
        self,
        activation: Optional[
            Union[
                torch.nn.Module,
                Callable[[torch.Tensor], torch.Tensor],
            ]
        ],
        dropout: Optional[float],
    ) -> None:
        batch_size, num_features, embedding_dim = 2, 3, 4

        pin_layer = PINLayer(activation, dropout)

        input_tensor: torch.Tensor = torch.randn(
            batch_size, num_features, embedding_dim
        )
        output_tensor = pin_layer(input_tensor, input_tensor)

        self.assertEqual(
            list(output_tensor.shape), [batch_size, num_features, embedding_dim]
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `hypothesis.strategies.integers(0, 5)` to decorator factory `hypothesis.given`.
    @given(
        num_pin_layers=st.integers(0, 5),
        num_subspaces=st.sampled_from([1, 2]),
        activation=st.sampled_from(
            [
                torch.nn.ReLU(),
                torch.tanh,
                torch.sigmoid,
                None,
            ]
        ),
        dropout=st.one_of(st.none(), st.floats(0.0, 1.0, exclude_max=True)),
    )
    def test_xdeepint(
        self,
        num_pin_layers: int,
        num_subspaces: int,
        activation: Optional[
            Union[
                torch.nn.Module,
                Callable[[torch.Tensor], torch.Tensor],
            ]
        ],
        dropout: Optional[float],
    ) -> None:
        batch_size, num_features, embedding_dim = 2, 3, 4

        xdeepint_network = XdeepInt(
            num_pin_layers,
            num_subspaces,
            activation=activation,
            dropout=dropout,
        )

        input_tensor: torch.Tensor = torch.randn(
            batch_size, num_features, embedding_dim
        )
        output_tensor = xdeepint_network(input_tensor)

        self.assertEqual(
            list(output_tensor.shape),
            [batch_size, num_features, embedding_dim * (num_pin_layers + 1)],
        )
