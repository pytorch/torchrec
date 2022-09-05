#!/usr/bin/env python3

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
    from .test_modules import Simple
except ImportError:
    from test_modules import Simple  # pyre-ignore


def save(
    name: str, model: torch.nn.Module, eg: Optional[Tuple] = None  # pyre-ignore
) -> None:  # pyre-ignore
    with PackageExporter(str(p / name)) as e:
        e.mock("iopath.**")
        e.intern("**")
        e.save_pickle("model", "model.pkl", model)
        if eg:
            e.save_pickle("model", "example.pkl", eg)


parser = argparse.ArgumentParser(description="Generate Examples")
parser.add_argument("--install_dir", help="Root directory for all output files")

if __name__ == "__main__":
    args = parser.parse_args() # pyre-ignore
    if args.install_dir is None:
        p = Path(__file__).parent / "generated" # pyre-ignore
        p.mkdir(exist_ok=True)
    else:
        p = Path(args.install_dir)

    simple = Simple(10, 20)
    for param in simple.parameters():
        param.requires_grad = False

    save("simple", simple, (torch.rand(10, 20),))
