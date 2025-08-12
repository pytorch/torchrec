#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, List, Union
from unittest.mock import MagicMock, patch

import torch
from torch.autograd import Variable
from torch.distributed import ProcessGroup
from torch.distributed.tensor import distribute_tensor, DTensor, init_device_mesh, Shard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchrec.optim.clipping import GradientClipping, GradientClippingOptimizer
from torchrec.optim.test_utils import DummyKeyedOptimizer


class TestGradientClippingOptimizer(unittest.TestCase):
    def test_clip_all_gradients_norm(self) -> None:
        # Clip all gradients to zero
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=0.0, clipping=GradientClipping.NORM
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([1.0, 2.0])
        gradient_clipping_optimizer.step()

        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.equal(param_1.grad, torch.tensor([0.0, 0.0])))

    def test_clip_no_gradients_norm(self) -> None:
        # gradients are too small to be clipped
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=1.0, clipping=GradientClipping.NORM
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([0.5, 0.5])
        gradient_clipping_optimizer.step()

        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.equal(param_1.grad, torch.tensor([0.5, 0.5])))

    def test_clip_partial_gradients_norm(self) -> None:
        # test partial clipping
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=1.0, clipping=GradientClipping.NORM
        )

        gradient_clipping_optimizer.zero_grad()

        param_1.grad = torch.tensor([2.0, 4.0])
        gradient_clipping_optimizer.step()

        norm = 2.0**2 + 4.0**2
        expected_grad = torch.tensor([2.0, 4.0]) * norm ** (-0.5)
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.allclose(param_1.grad, expected_grad))

    def test_clip_partial_gradients_norm_multi_params(self) -> None:
        # test partial clipping
        max_gradient = 2.0
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)
        param_2 = Variable(torch.tensor([2.0, 4.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1, "param_2": param_2},
            {},
            [{"params": [param_1]}, {"params": [param_2]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=max_gradient,
            clipping=GradientClipping.NORM,
        )

        gradient_clipping_optimizer.zero_grad()

        param_1.grad = torch.tensor([2.0, 4.0])
        param_2.grad = torch.tensor([4.0, 8.0])

        gradient_clipping_optimizer.step()

        print(param_1.grad, param_2.grad)

        norm = (2.0**2 + 4.0**2 + 4.0**2 + 8.0**2) ** (-0.5)
        expected_grad_1 = torch.tensor([2.0, 4.0]) * norm * max_gradient
        expected_grad_2 = torch.tensor([4.0, 8.0]) * norm * max_gradient

        print(param_1.grad, param_2.grad, expected_grad_1, expected_grad_2)

        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.allclose(param_1.grad, expected_grad_1))
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.allclose(param_2.grad, expected_grad_2))

    def test_clip_all_gradients_value(self) -> None:
        # Clip all gradients to zero
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=0, clipping=GradientClipping.VALUE
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([1.0, 2.0])
        gradient_clipping_optimizer.step()

        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.equal(param_1.grad, torch.tensor([0.0, 0.0])))

    def test_clip_no_gradients_value(self) -> None:
        # gradients are too small to be clipped
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=1.0, clipping=GradientClipping.VALUE
        )

        gradient_clipping_optimizer.zero_grad()
        param_1.grad = torch.tensor([0.5, 0.5])
        gradient_clipping_optimizer.step()

        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.equal(param_1.grad, torch.tensor([0.5, 0.5])))

    def test_clip_gradients_value(self) -> None:
        # test partial clipping
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=1, clipping=GradientClipping.VALUE
        )

        gradient_clipping_optimizer.zero_grad()

        param_1.grad = torch.tensor([2.0, 4.0])
        gradient_clipping_optimizer.step()

        expected_grad = torch.tensor([1.0, 1.0])

        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.allclose(param_1.grad, expected_grad))

    def test_clip_partial_gradients_value_multi_params(self) -> None:
        # test partial clipping
        max_gradient = 2.0
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)
        param_2 = Variable(torch.tensor([2.0, 4.0]), requires_grad=True)

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1, "param_2": param_2},
            {},
            [{"params": [param_1]}, {"params": [param_2]}],
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            max_gradient=max_gradient,
            clipping=GradientClipping.VALUE,
        )

        gradient_clipping_optimizer.zero_grad()

        param_1.grad = torch.tensor([2.0, 4.0])
        param_2.grad = torch.tensor([4.0, 8.0])

        gradient_clipping_optimizer.step()

        expected_grad_1 = torch.tensor([2.0, 2.0])
        expected_grad_2 = torch.tensor([2.0, 2.0])

        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.allclose(param_1.grad, expected_grad_1))
        # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.allclose(param_2.grad, expected_grad_2))

    @patch("torch.nn.utils.clip_grad_norm_")
    def test_clip_no_gradients_norm_meta_device(
        self, mock_clip_grad_norm: MagicMock
    ) -> None:
        # Clip all gradients to zero
        param_1 = Variable(
            torch.tensor([1.0, 2.0], device=torch.device("meta")), requires_grad=True
        )

        keyed_optimizer = DummyKeyedOptimizer(
            {"param_1": param_1}, {}, [{"params": [param_1]}]
        )

        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer, max_gradient=0.0, clipping=GradientClipping.NORM
        )

        gradient_clipping_optimizer.zero_grad()
        gradient_clipping_optimizer.step()

        mock_clip_grad_norm.assert_not_called()


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class TestGradientClippingDTensor(DTensorTestBase):
    """No tests for Replicated DTensors as handled prior to GradientClippingOptimizer"""

    def _get_params_to_pg(
        self, params: List[DTensor]
    ) -> Dict[DTensor, List[ProcessGroup]]:
        return {param: [param.device_mesh.get_group()] for param in params}

    @with_comms
    @parametrize("norm_type", ("inf", 1, 2))
    def test_tensor_and_sharded_dtensor_clip_all_gradients_norm(
        self, norm_type: Union[float, str]
    ) -> None:
        """
        Test to ensure that the gradient clipping optimizer clips gradients
        correctly with mixed sharded DTensor and tensor by comparing gradients to its
        torch.tensor counterpart.

        Note that clipping for DTensor may require communication.
        """

        # data for testing clipping
        data_1 = torch.tensor([1.0, 2.0, 3.0], device=self.device_type)
        data_2 = torch.tensor([4.0, 5.0, 6.0], device=self.device_type)
        data_1_grad = torch.tensor([12.0, 15.0, 18.0], device=self.device_type)
        data_2_grad = torch.tensor([20.0, 30.0, 15.0], device=self.device_type)

        # create gradient clipping optimizer containing no dtensor for reference
        ref_param_1 = torch.nn.Parameter(data_1.clone())
        ref_param_2 = torch.nn.Parameter(data_2.clone())
        ref_param_1.grad = data_1_grad.clone()
        ref_param_2.grad = data_2_grad.clone()
        ref_keyed_optimizer = DummyKeyedOptimizer(
            params={"param_1": ref_param_1, "param_2": ref_param_2},
            state={},
            param_groups=[{"params": [ref_param_1, ref_param_2]}],
        )
        ref_gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=ref_keyed_optimizer,
            clipping=GradientClipping.NORM,
            max_gradient=10.0,
            norm_type=norm_type,
        )
        ref_gradient_clipping_optimizer.step()

        # create gradient clipping optimizer containing a DTensor and a tensor
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        param_1 = distribute_tensor(
            tensor=torch.tensor(
                data_1.clone(), requires_grad=True, device=self.device_type
            ),
            device_mesh=device_mesh,
            placements=[Shard(0)],
        )
        param_2 = torch.tensor(
            data_2.clone(), requires_grad=True, device=self.device_type
        )
        param_1.grad = distribute_tensor(
            tensor=data_1_grad.clone(),
            device_mesh=device_mesh,
            placements=[Shard(0)],
        )
        param_2.grad = data_2_grad.clone()
        param_to_pgs = self._get_params_to_pg([param_1])
        keyed_optimizer = DummyKeyedOptimizer(
            params={"dtensor_param_1": param_1, "dtensor_param_2": param_2},
            state={},
            param_groups=[{"params": [param_1, param_2]}],
        )
        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            clipping=GradientClipping.NORM,
            max_gradient=10.0,
            norm_type=norm_type,
            enable_global_grad_clip=True,
            param_to_pgs=param_to_pgs,  # pyre-ignore[6]
        )
        gradient_clipping_optimizer.step()

        for param_group, ref_param_group in zip(
            gradient_clipping_optimizer.param_groups,
            ref_gradient_clipping_optimizer.param_groups,
            strict=True,
        ):
            for param, ref_param in zip(
                param_group["params"], ref_param_group["params"], strict=True
            ):
                param_grad = (
                    param.grad.full_tensor()  # pyre-ignore[16]
                    if isinstance(param, DTensor)
                    else param.grad
                )
                self.assertEqual(
                    param_grad,
                    ref_param.grad,
                    f"Expect gradient to be the same. However, found {param_grad=}, {ref_param.grad=}",
                )

    @with_comms
    @parametrize("norm_type", ("inf", 1, 2))
    def test_multiple_sharded_dtensors_clip_all_gradients_norm(
        self, norm_type: Union[float, str]
    ) -> None:
        """
        Test to ensure that the gradient clipping optimizer clips gradients
        correctly with multiple sharded DTensors by comparing gradients to their
        torch.tensor counterpart.

        Note that clipping for DTensor may require communication.
        """

        # data for testing clipping
        data_1 = torch.tensor([1.0, 2.0, 3.0], device=self.device_type)
        data_2 = torch.tensor([4.0, 5.0, 6.0], device=self.device_type)
        data_1_grad = torch.tensor([12.0, 15.0, 18.0], device=self.device_type)
        data_2_grad = torch.tensor([20.0, 30.0, 15.0], device=self.device_type)

        # create gradient clipping optimizer containing no dtensor for reference
        ref_param_1 = torch.nn.Parameter(data_1.clone())
        ref_param_2 = torch.nn.Parameter(data_2.clone())
        ref_param_1.grad = data_1_grad.clone()
        ref_param_2.grad = data_2_grad.clone()
        ref_keyed_optimizer = DummyKeyedOptimizer(
            params={"param_1": ref_param_1, "param_2": ref_param_2},
            state={},
            param_groups=[{"params": [ref_param_1, ref_param_2]}],
        )
        ref_gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=ref_keyed_optimizer,
            clipping=GradientClipping.NORM,
            max_gradient=10.0,
            norm_type=norm_type,
        )
        ref_gradient_clipping_optimizer.step()

        # create gradient clipping optimizer containing 2 DTensors
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        param_1 = distribute_tensor(
            tensor=torch.tensor(
                data_1.clone(), requires_grad=True, device=self.device_type
            ),
            device_mesh=device_mesh,
            placements=[Shard(0)],
        )
        param_2 = distribute_tensor(
            tensor=torch.tensor(
                data_2.clone(), requires_grad=True, device=self.device_type
            ),
            device_mesh=device_mesh,
            placements=[Shard(0)],
        )
        param_1.grad = distribute_tensor(
            tensor=data_1_grad.clone(),
            device_mesh=device_mesh,
            placements=[Shard(0)],
        )
        param_2.grad = distribute_tensor(
            tensor=data_2_grad.clone(),
            device_mesh=device_mesh,
            placements=[Shard(0)],
        )
        param_to_pgs = self._get_params_to_pg([param_1, param_2])
        keyed_optimizer = DummyKeyedOptimizer(
            params={"dtensor_param_1": param_1, "dtensor_param_2": param_2},
            state={},
            param_groups=[{"params": [param_1, param_2]}],
        )
        gradient_clipping_optimizer = GradientClippingOptimizer(
            optimizer=keyed_optimizer,
            clipping=GradientClipping.NORM,
            max_gradient=10.0,
            norm_type=norm_type,
            enable_global_grad_clip=True,
            param_to_pgs=param_to_pgs,  # pyre-ignore[6]
        )
        gradient_clipping_optimizer.step()

        for param_group, ref_param_group in zip(
            gradient_clipping_optimizer.param_groups,
            ref_gradient_clipping_optimizer.param_groups,
            strict=True,
        ):
            for param, ref_param in zip(
                param_group["params"], ref_param_group["params"], strict=True
            ):
                param_grad = (
                    param.grad.full_tensor()  # pyre-ignore[16]
                    if isinstance(param, DTensor)
                    else param.grad
                )
                self.assertEqual(
                    param_grad,
                    ref_param.grad,
                    f"Expect gradient to be the same. However, found {param_grad=}, {ref_param.grad=}",
                )
