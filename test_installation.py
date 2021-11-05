#!/usr/bin/env python3

import os

import torchx.specs as specs
from torchx.components.base import torch_dist_role
from torchx.specs.api import Resource


def test_installation() -> specs.AppDef:
    cwd = os.getcwd()
    entrypoint = os.path.join(cwd, "test_installation_main.py")

    user = os.environ.get("USER")
    image = f"/data/home/{user}"

    return specs.AppDef(
        name="test_installation",
        roles=[
            torch_dist_role(
                name="trainer",
                image=image,
                # AWS p4d instance (https://aws.amazon.com/ec2/instance-types/p4/).
                resource=Resource(
                    cpu=96,
                    gpu=8,
                    memMB=-1,
                ),
                num_replicas=1,
                entrypoint=entrypoint,
                nproc_per_node="1",
                rdzv_backend="c10d",
                args=[],
            ),
        ],
    )
