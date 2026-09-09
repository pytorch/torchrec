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
to a checkpoint. Old checkpoints then fail to load. ``@checkpoint_schema_stable``
marks a class with its stable checkpoint-schema id. A matching ``_GoldenCase``
row is what enrolls the class in the golden-snapshot test.

The decorator has no effect at load time and registers nothing anywhere. It
records the id on the class, so the test can check that a row and the class it
names agree on which id they mean.
"""

import re
from typing import Callable, Optional, TypeVar

_C = TypeVar("_C", bound=type)

_STABLE_ID = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)*")

# Set by the decorator. Read with schema_id_of, never with getattr.
SCHEMA_ID_ATTRIBUTE = "_checkpoint_schema_id"


def checkpoint_schema_stable(stable_id: str) -> Callable[[_C], _C]:
    """Mark a class with the id its golden entry is filed under.

    The id is written by hand rather than derived from the class, so moving a
    file or renaming a class does not silently change the golden key and abandon
    the entry it used to match. That is the whole point of it: an id must stay
    put when its class is renamed.
    """
    if not stable_id or stable_id != stable_id.strip():
        raise ValueError(
            f"stable_id must be non-empty and unpadded, got {stable_id!r}."
        )
    if not _STABLE_ID.fullmatch(stable_id):
        raise ValueError(
            f"stable_id must be lowercase snake_case, got {stable_id!r}. A "
            "class-shaped id invites the next person to rename it along with "
            "the class, which abandons the golden entry it was filed under."
        )

    def decorate(cls: _C) -> _C:
        setattr(cls, SCHEMA_ID_ATTRIBUTE, stable_id)
        return cls

    return decorate


def schema_id_of(cls: type) -> Optional[str]:
    """The id this class was marked with, or None.

    Reads the class's own namespace. A subclass of a marked class inherits the
    attribute, so getattr would report the parent's id and hide a subclass that
    was never marked at all.
    """
    return vars(cls).get(SCHEMA_ID_ATTRIBUTE)
