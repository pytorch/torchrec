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
    For filtering out certain cuda versions in the build matrix that
    determines with nightly builds are run. This ensures TorchRec is
    always consistent in version compatibility with FBGEMM.
    """

    full_matrix_string = os.environ["MAT"]
    full_matrix = json.loads(full_matrix_string)

    new_matrix_entries = []

    for entry in full_matrix["include"]:
        new_matrix_entries.append(entry)

    new_matrix = {"include": new_matrix_entries}
    print(json.dumps(new_matrix))


if __name__ == "__main__":
    main()
