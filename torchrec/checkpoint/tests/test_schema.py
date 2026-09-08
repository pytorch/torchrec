#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The registry's contracts, none of which importing a decorated class exercises.

A stable id is the name a golden entry is filed under. If one can be blank, or
carry stray whitespace, or be claimed by a second class, the entry it points at
stops describing the class that wrote it.
"""

import unittest
from typing import Type

from torchrec.checkpoint import schema
from torchrec.checkpoint.schema import (
    checkpoint_schema_stable,
    registered_schema_stable_classes,
)


def _make_class(qualname: str, module: str = "torchrec.fake") -> Type[object]:
    """A class with a controlled identity.

    Two calls with the same name give two distinct objects that look identical
    to the registry, which is what a module reload produces.
    """
    cls = type(qualname, (), {})
    cls.__module__ = module
    cls.__qualname__ = qualname
    return cls


class CheckpointSchemaStableTest(unittest.TestCase):
    def setUp(self) -> None:
        # The registry is module-level, so a test that registers anything would
        # otherwise leak into the next one and into the real entries.
        previous = dict(schema._SCHEMA_STABLE_CLASSES)
        self.addCleanup(schema._SCHEMA_STABLE_CLASSES.update, previous)
        self.addCleanup(schema._SCHEMA_STABLE_CLASSES.clear)

    def test_rejects_an_empty_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty and unpadded"):
            checkpoint_schema_stable("")

    def test_rejects_a_padded_id(self) -> None:
        for stable_id in (" widget", "widget ", "\twidget", "widget\n"):
            with self.subTest(stable_id=stable_id):
                with self.assertRaisesRegex(ValueError, "non-empty and unpadded"):
                    checkpoint_schema_stable(stable_id)

    def test_rejects_a_second_class_under_one_id(self) -> None:
        checkpoint_schema_stable("widget")(_make_class("Widget"))
        with self.assertRaisesRegex(ValueError, "cannot reuse it"):
            checkpoint_schema_stable("widget")(_make_class("Gadget"))

    def test_a_reloaded_class_may_reclaim_its_id(self) -> None:
        # Same module and qualname, different object — what reimporting a module
        # produces. Raising here would turn a reload into an import failure.
        first = _make_class("Widget")
        second = _make_class("Widget")
        self.assertIsNot(first, second)

        checkpoint_schema_stable("widget")(first)
        checkpoint_schema_stable("widget")(second)

        # The later registration wins, so the registry tracks the live class.
        self.assertIs(registered_schema_stable_classes()["widget"], second)

    def test_returns_the_class_unchanged(self) -> None:
        cls = _make_class("Widget")
        self.assertIs(checkpoint_schema_stable("widget")(cls), cls)

    def test_the_registry_view_is_a_copy(self) -> None:
        checkpoint_schema_stable("widget")(_make_class("Widget"))
        view = registered_schema_stable_classes()
        view.clear()
        self.assertIn("widget", registered_schema_stable_classes())
