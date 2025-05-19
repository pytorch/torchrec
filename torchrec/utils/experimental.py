#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, overload, ParamSpec, Type, TypeVar, Union

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


@overload
def experimental(
    obj: Callable[P, R],
    feature: str | None = None,
    since: str | None = None,
) -> Callable[P, R]: ...


@overload
def experimental(
    obj: Type[T],
    feature: str | None = None,
    since: str | None = None,
) -> Type[T]: ...


def experimental(
    obj: Union[Callable[P, R], Type[T]],
    feature: str | None = None,
    since: str | None = None,
) -> Union[Callable[P, R], Type[T]]:
    tag: str = feature or obj.__name__  # pyre-ignore[16]
    message_parts: list[str] = [
        f"`{tag}` is *experimental* and may change or be removed without notice."
    ]
    if since:
        message_parts.insert(0, f"[since {since}]")
    warning_message: str = " ".join(message_parts)

    @functools.lru_cache(maxsize=1)
    def _issue_warning() -> None:
        warnings.warn(warning_message, UserWarning, stacklevel=3)

    if isinstance(obj, type):
        orig_init: Callable[..., None] = obj.__init__

        @functools.wraps(orig_init)
        def new_init(self, *args: Any, **kwargs: Any) -> Any:  # pyre-ignore[3]
            _issue_warning()
            return orig_init(self, *args, **kwargs)

        obj.__init__ = new_init
        return obj
    else:

        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # pyre-ignore[3]
            _issue_warning()
            return obj(*args, **kwargs)  # pyre-ignore[29]

        return wrapper
