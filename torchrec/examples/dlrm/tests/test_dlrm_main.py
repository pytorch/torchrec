#!/usr/bin/env python3

import os
import tempfile
import unittest
import uuid

from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torchrec.examples.dlrm.dlrm_main import main
from torchrec.tests import utils


class MainTest(unittest.TestCase):
    @classmethod
    def _run_trainer(cls) -> None:
        main(
            [
                "--limit_train_batches",
                "5",
                "--over_arch_layer_sizes",
                "8,1",
                "--dense_arch_layer_sizes",
                "8,8",
                "--embedding_dim",
                "8",
                "--num_embeddings",
                "64",
            ]
        )

    @utils.skip_if_asan
    def test_main_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lc = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=2,
                run_id=str(uuid.uuid4()),
                rdzv_backend="c10d",
                rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
                rdzv_configs={"store_type": "file"},
                start_method="spawn",
                monitor_interval=1,
                max_restarts=0,
            )

            elastic_launch(config=lc, entrypoint=self._run_trainer)()
