#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


class BaseArgInfoStep(abc.ABC):
    @abc.abstractmethod
    # pyre-ignore
    def process(self, arg) -> Any:
        raise Exception("Not implemented in the BaseArgInfoStep")

    def __eq__(self, other: object) -> bool:
        """
        Some tests use the equality checks on the ArgInfo and/or CallArgs, so it's
        natural to use dataclasses for ArgInfoStep implementations. However
        Torchrec doesn't like dataclasses: https://github.com/pytorch/pytorch/issues/74909

        So, this class creates a makeshift generic implementation similar to dataclass, but without
        dataclass.
        """
        if not isinstance(other, type(self)):
            return False
        return all(
            getattr(self, field_name) == getattr(other, field_name)
            for field_name in self.__dict__.keys()
        )


@dataclass
class ArgInfo:
    """
    Representation of args from a node.

    Attributes:
        steps (List[ArgInfoStep]): sequence of transformations from input batch.
            Steps can be thought of consequtive transformations on the input, with
            output of previous step used as an input for the next. I.e. for 3 steps
            it is similar to step3(step2(step1(input)))
            See `BaseArgInfoStep` class hierearchy for supported transformations
    """

    steps: List[BaseArgInfoStep]

    def add_step(self, step: BaseArgInfoStep) -> "ArgInfo":
        self.steps.insert(0, step)
        return self

    def append_step(self, step: BaseArgInfoStep) -> "ArgInfo":
        self.steps.append(step)
        return self

    # pyre-ignore[3]
    def process_steps(
        self,
        arg: Any,  # pyre-ignore[2]
    ) -> Any:
        if not self.steps:
            return None
        for step in self.steps:
            arg = step.process(arg)

        return arg


@dataclass
class CallArgs:
    args: List[ArgInfo]
    kwargs: Dict[str, ArgInfo]

    # pyre-ignore[3]
    def build_args_kwargs(
        self, initial_input: Any  # pyre-ignore[2]
    ) -> Tuple[List[Any], Dict[str, Any]]:
        args = [arg.process_steps(initial_input) for arg in self.args]
        kwargs = {
            key: arg.process_steps(initial_input) for key, arg in self.kwargs.items()
        }
        return args, kwargs
