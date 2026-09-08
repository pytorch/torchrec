#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Marks a module's state_dict key set as a compatibility surface.

Adding, removing, or renaming persistent state changes the keys a module writes
to a checkpoint. Old checkpoints then fail to load. Decorating a class with
``@checkpoint_schema_stable("some_id")`` enrolls it in the golden-snapshot test,
so the next key-set change on it has to be acknowledged at diff submission time.

The decorator has no effect at load time. It records the class so tests can
find it.
"""

from typing import Callable, Dict, Type, TypeVar

_C = TypeVar("_C", bound=type)


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


# Stable id -> the class that claimed it.
_SCHEMA_STABLE_CLASSES: Dict[str, Type[object]] = {}


def checkpoint_schema_stable(stable_id: str) -> Callable[[_C], _C]:
    """Enroll a class in the golden-snapshot test under ``stable_id``.

    The id is written by hand rather than derived from the class, so moving a
    file or renaming a module does not silently change the golden key and
    abandon the entry it used to match. Two classes sharing a bare name pick
    different ids, e.g. ``embedding_bag_collection.float`` and
    ``embedding_bag_collection.quantized``.

    Two different classes claiming one id raises: the second would otherwise
    shadow the first and quietly take over its snapshot. Re-registering the same
    class is allowed, so a module reload does not become an import failure.

    Registration happens at import, so the registry only holds classes the
    importing target depends on. A decorated class in a module nothing imports
    stays invisible. The paired coverage test cross-checks the registry against
    its own case list, which catches divergence between those two, but neither
    can see a module that was never imported.
    """
    if not stable_id or stable_id != stable_id.strip():
        raise ValueError(f"stable_id must be non-empty and unpadded, got {stable_id!r}")

    def decorate(cls: _C) -> _C:
        existing = _SCHEMA_STABLE_CLASSES.get(stable_id)
        if existing is not None and _qualname(existing) != _qualname(cls):
            raise ValueError(
                f"{stable_id!r} is already registered to "
                f"{_qualname(existing)}; {_qualname(cls)} cannot reuse it."
            )
        _SCHEMA_STABLE_CLASSES[stable_id] = cls
        return cls

    return decorate


def registered_schema_stable_classes() -> Dict[str, Type[object]]:
    """Every registered class, keyed by stable id."""
    return dict(_SCHEMA_STABLE_CLASSES)
