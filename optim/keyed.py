#!/usr/bin/env python3

from copy import deepcopy
from typing import (
    Callable,
    List,
    Mapping,
    Dict,
    Any,
    Collection,
    Tuple,
    Union,
)

import torch
from torch import optim
from torchrec.distributed.types import ShardedTensor

OptimizerFactory = Callable[[List[torch.Tensor]], optim.Optimizer]


class KeyedOptimizer(optim.Optimizer):
    """
    Takes a dict of parameters and exposes state_dict by parameter key.
    """

    def __init__(
        self,
        params: Mapping[str, Union[torch.Tensor, ShardedTensor]],
        # pyre-ignore [2]
        state: Mapping[Any, Any],
        param_groups: Collection[Mapping[str, Any]],
    ) -> None:
        torch._C._log_api_usage_once(f"torchrec.optim.{self.__class__.__name__}")
        # pyre-ignore [4]
        self.state: Mapping[Any, Any] = state
        self.param_groups: Collection[Mapping[str, Any]] = param_groups
        self.params = params

        params_set = set(params.values())
        non_param_state_keys = [key for key in self.state if key not in params_set]
        if len(non_param_state_keys) > 0:
            raise ValueError(
                "All state keys must be params. The following keys are not: {}.".format(
                    non_param_state_keys
                )
            )

        # Same parameter can be associated w/ multiple keys,
        # it happens e.g. in case of BatchedDenseEmbeddingBag.
        self._param_to_keys: Dict[torch.Tensor, List[str]] = {}
        for key, p in params.items():
            tensor = p.local_shard if isinstance(p, ShardedTensor) else p
            if tensor not in self._param_to_keys:
                self._param_to_keys[tensor] = []
            self._param_to_keys[tensor].append(key)

    def state_dict(self) -> Dict[str, Any]:
        state = {}
        param_groups = []
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param in self.state:
                    keys = self._param_to_keys[param]
                    # TODO figure out how to split optimizer state if needed.
                    for key in keys:
                        state[key] = self.state[param]

            packed = {k: v for k, v in param_group.items() if k != "params"}
            param_keys = []
            for p in param_group["params"]:
                param_keys += self._param_to_keys[p]
            packed["params"] = param_keys
            param_groups.append(packed)
        return {"state": state, "param_groups": param_groups}

    # pyre-ignore [2]
    def add_param_group(self, param_group: Any) -> None:
        raise NotImplementedError()


class CombinedOptimizer(KeyedOptimizer):
    """
    Combines multiple optimizers into one.
    """

    def __init__(
        self, optims: List[Union[KeyedOptimizer, Tuple[str, KeyedOptimizer]]]
    ) -> None:
        super().__init__({}, {}, [])

        # Append empty optimizer key if not passed.
        self._optims: List[Tuple[str, KeyedOptimizer]] = []
        for key_value in optims:
            if isinstance(key_value, KeyedOptimizer):
                key_value = ("", key_value)
            self._optims.append(key_value)

        self._new_to_old_keys: List[Dict[str, str]] = [
            {
                CombinedOptimizer._prepend_opt_key(param, opt_key): param
                for param_group in opt.state_dict()["param_groups"]  # CombinedOptimizer
                # can have CombinedOptimizers as children, so directly accessing
                # opt.params is incorrect as CombinedOptimizers themselves do not hold
                # any params.
                for param in param_group["params"]
            }
            for opt_key, opt in self._optims
        ]

    def __repr__(self) -> str:
        ret = []
        for _, opt in self._optims:
            ret.append(opt.__repr__())
        return ",".join(ret)

    def zero_grad(self, set_to_none: bool = False) -> None:
        for _, opt in self._optims:
            # pyre-ignore [28]
            opt.zero_grad(set_to_none=set_to_none)

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        for _, opt in self._optims:
            opt.step(closure=closure)

    @property
    def optimizers(self) -> List[Tuple[str, KeyedOptimizer]]:
        return self._optims

    @staticmethod
    def _prepend_opt_key(name: str, opt_key: str) -> str:
        return opt_key + ("." if opt_key else "") + name

    def state_dict(self) -> Dict[str, Any]:
        # pyre-ignore[3]
        def update_params(
            param_group: Dict[Any, Any],  # pyre-ignore[2]
            opt_key: str,
        ) -> Dict[Any, Any]:
            params = param_group["params"]
            param_group["params"] = [
                CombinedOptimizer._prepend_opt_key(param, opt_key) for param in params
            ]
            return param_group

        state = {}
        param_groups = []

        for opt_key, opt in self._optims:
            opt_state_dict = opt.state_dict()

            opt_state = opt_state_dict["state"]
            opt_state = {
                CombinedOptimizer._prepend_opt_key(key, opt_key): value
                for (key, value) in opt_state.items()
            }
            state.update(opt_state)

            opt_param_groups = opt_state_dict["param_groups"]
            opt_param_groups = [
                update_params(param_group, opt_key) for param_group in opt_param_groups
            ]
            param_groups.extend(opt_param_groups)

        return {"state": state, "param_groups": param_groups}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        state_dict = deepcopy(state_dict)

        # pyre-ignore[3]
        def update_params(
            param_group: Dict[Any, Any],  # pyre-ignore[2]
            new_to_old_keys: Dict[str, str],
        ) -> Dict[Any, Any]:
            params = param_group["params"]
            param_group["params"] = [
                new_to_old_keys[param] for param in params if param in new_to_old_keys
            ]
            return param_group

        state = state_dict["state"]
        param_groups = state_dict["param_groups"]
        for (_, opt), new_to_old_keys in zip(self._optims, self._new_to_old_keys):
            opt_state = {
                new_to_old_keys[key]: value
                for (key, value) in state.items()
                if key in new_to_old_keys
            }
            opt_param_groups = [
                update_params(param_group, new_to_old_keys)
                for param_group in param_groups
                if all(p in new_to_old_keys for p in param_group["params"])
            ]
            opt_state_dict = {"state": opt_state, "param_groups": opt_param_groups}
            opt.load_state_dict(opt_state_dict)


class KeyedOptimizerWrapper(KeyedOptimizer):
    """
    Takes a dict of parameters and exposes state_dict by parameter key.
    """

    def __init__(
        self,
        params: Mapping[str, Union[torch.Tensor, ShardedTensor]],
        optim_factory: OptimizerFactory,
    ) -> None:
        # Get local shards and dedup.
        tensors = {
            (value.local_shard if isinstance(value, ShardedTensor) else value)
            for value in params.values()
        }
        self._optimizer: optim.Optimizer = optim_factory(list(tensors))
        super().__init__(params, self._optimizer.state, self._optimizer.param_groups)

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._optimizer.zero_grad()

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        self._optimizer.step(closure=closure)


class OptimizerWrapper(KeyedOptimizer):
    def __init__(self, optimizer: KeyedOptimizer) -> None:
        self._optimizer = optimizer
        self.params: Mapping[str, Union[torch.Tensor, ShardedTensor]] = optimizer.params
        # pyre-ignore [4]
        self.state: Mapping[Any, Any] = optimizer.state
        self.param_groups: Collection[Mapping[str, Any]] = optimizer.param_groups

    def __repr__(self) -> str:
        return self._optimizer.__repr__()

    def zero_grad(self, set_to_none: bool = False) -> None:
        # pyre-ignore [28]
        self._optimizer.zero_grad(set_to_none=set_to_none)

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        self._optimizer.step(closure=closure)

    # pyre-ignore [2]
    def add_param_group(self, param_group: Any) -> None:
        raise NotImplementedError()

    def state_dict(self) -> Dict[str, Any]:
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._optimizer.load_state_dict(state_dict)
        # Reassign references because self._optimizer receives new state and param_group
        # references after load_state_dict.
        self.state = self._optimizer.state
        self.param_groups = self._optimizer.param_groups
