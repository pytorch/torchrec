#!/usr/bin/env python3

from typing import Any

from torchrec.optim.keyed import KeyedOptimizer


class DummyKeyedOptimizer(KeyedOptimizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # pyre-ignore[2]
    def step(self, closure: Any) -> None:
        pass  # Override NotImplementedError.
