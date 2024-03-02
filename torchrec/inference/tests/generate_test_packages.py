#!/usr/bin/env python3

# pyre-strict

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
# @nolint

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.package import PackageExporter

try:
    from .test_modules import Nested, Simple
except ImportError:
    from test_modules import Nested, Simple  # pyre-ignore


def save(
    name: str, model: torch.nn.Module, eg: Optional[Tuple] = None  # pyre-ignore
) -> None:
    # pyre-fixme[10]: Name `p` is used but not defined.
    with PackageExporter(str(p / name)) as e:
        e.mock("iopath.**")
        e.intern("**")
        e.save_pickle("model", "model.pkl", model)
        if eg:
            e.save_pickle("model", "example.pkl", eg)


def post_process(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


parser = argparse.ArgumentParser(description="Generate Examples")
parser.add_argument("--install_dir", help="Root directory for all output files")


def main() -> None:
    global p
    args = parser.parse_args()
    if args.install_dir is None:
        p = Path(__file__).parent / "generated"
        p.mkdir(exist_ok=True)
    else:
        p = Path(args.install_dir)

    simple = Simple(10, 20)
    post_process(simple)

    nested = Nested(10, 20)
    post_process(nested)

    save("simple", simple, (torch.rand(10, 20),))
    save("nested", nested, (torch.rand(10, 20),))


if __name__ == "__main__":
    main()  # pragma: no cover
