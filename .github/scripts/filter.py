#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os


def main():
    """
    Since FBGEMM doesn't publish CUDA 12 binaries, torchrec will not work with
    CUDA 12. As a result, we filter out CUDA 12 from the build matrix that
    determines with nightly builds are run.
    """

    full_matrix_string = os.environ["MAT"]
    full_matrix = json.loads(full_matrix_string)

    new_matrix_entries = []

    for entry in full_matrix["include"]:
        if entry["gpu_arch_version"] != "12.1":
            new_matrix_entries.append(entry)

    new_matrix = {"include": new_matrix_entries}
    print(json.dumps(new_matrix))


if __name__ == "__main__":
    main()
