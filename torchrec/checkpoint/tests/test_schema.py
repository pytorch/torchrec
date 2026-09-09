#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The decorator's contracts, none of which importing a marked class exercises.

A stable id is the name a golden entry is filed under. If one can be blank or
carry stray whitespace, the entry it points at stops describing the class that
wrote it. And because every enrolled module here subclasses another marked
class, reading the mark has to ignore inheritance or a subclass that was never
marked reports its parent's id.
"""

import unittest
from typing import Type

from torchrec.checkpoint.schema import (
    checkpoint_schema_stable,
    SCHEMA_ID_ATTRIBUTE,
    schema_id_of,
)


def _make_class(qualname: str, module: str = "torchrec.fake") -> Type[object]:
    """A class with a controlled identity."""
    cls = type(qualname, (), {})
    cls.__module__ = module
    cls.__qualname__ = qualname
    return cls


class CheckpointSchemaStableTest(unittest.TestCase):
    def test_rejects_an_empty_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty and unpadded"):
            checkpoint_schema_stable("")

    def test_rejects_a_padded_id(self) -> None:
        for stable_id in (" widget", "widget ", "\twidget", "widget\n"):
            with self.subTest(stable_id=stable_id):
                with self.assertRaisesRegex(ValueError, "non-empty and unpadded"):
                    checkpoint_schema_stable(stable_id)

    def test_rejects_ids_that_are_not_snake_case(self) -> None:
        for stable_id in (
            "RecMetricModule",
            "Widget",
            "wid-get",
            "wid get",
            "widget__gadget",
            "_widget",
            "widget_",
            "widget2X",
        ):
            with self.subTest(stable_id=stable_id):
                with self.assertRaisesRegex(ValueError, "lowercase snake_case"):
                    checkpoint_schema_stable(stable_id)

    def test_accepts_the_ids_in_use(self) -> None:
        for stable_id in (
            "tensor_pool",
            "cpu_offloaded_rec_metric_module",
            "hash_zch_managed_collision_module",
            "widget2",
        ):
            with self.subTest(stable_id=stable_id):
                cls = _make_class("Widget")
                checkpoint_schema_stable(stable_id)(cls)
                self.assertEqual(schema_id_of(cls), stable_id)

    def test_marks_the_class_with_its_id(self) -> None:
        cls = _make_class("Widget")
        checkpoint_schema_stable("widget")(cls)
        self.assertEqual(schema_id_of(cls), "widget")

    def test_returns_the_class_unchanged(self) -> None:
        cls = _make_class("Widget")
        self.assertIs(checkpoint_schema_stable("widget")(cls), cls)

    def test_an_unmarked_class_has_no_id(self) -> None:
        self.assertIsNone(schema_id_of(_make_class("Widget")))

    def test_a_subclass_does_not_inherit_its_parent_id(self) -> None:
        # The real subject: CPUCommsRecMetricModule and friends all subclass
        # RecMetricModule, which is marked. Reading with getattr would report
        # the parent's id and hide a subclass nobody marked.
        parent = _make_class("Widget")
        checkpoint_schema_stable("widget")(parent)
        child = type("Gadget", (parent,), {})

        self.assertEqual(getattr(child, SCHEMA_ID_ATTRIBUTE), "widget")
        self.assertIsNone(schema_id_of(child))

    def test_a_marked_subclass_reports_its_own_id(self) -> None:
        parent = _make_class("Widget")
        checkpoint_schema_stable("widget")(parent)
        child = checkpoint_schema_stable("gadget")(type("Gadget", (parent,), {}))

        self.assertEqual(schema_id_of(child), "gadget")
        self.assertEqual(schema_id_of(parent), "widget")
