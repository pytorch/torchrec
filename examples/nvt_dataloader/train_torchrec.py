import argparse
from collections import defaultdict
import glob
import logging
import os
import sys

from typing import cast, List
from pyre_extensions import none_throws

import torch
import torch.distributed as dist
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torchrec import EmbeddingBagCollection
import torchrec.distributed as trec_dist
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.metrics.throughput import ThroughputMetric
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
import torchrec.optim as trec_optim

from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
)

from dlrm_train import DLRMTrain
from nvt_criteo_dataloader import NvtCriteoDataloader
import cupy


logger: logging.Logger = logging.getLogger(__name__)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="local batch size to use for training",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,128",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="1024,1024,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="/data/criteo_1tb/criteo_preproc",
        help="Location for parquet datafiles",
    )
    return parser.parse_args(argv)


def main(argv: List[str]):
    args = parse_args(argv)

    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ["LOCAL_RANK"]
    if not torch.distributed.is_initialized():
        dist.init_process_group(backend=backend)

    cupy.cuda.Device(rank).use()
    torch.cuda.set_device(device)


    rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_embeddings_per_feature = None
    if args.num_embeddings_per_feature is not None:
        num_embeddings_per_feature = list(
            map(int, args.num_embeddings_per_feature.split(","))
        )

    train_paths = sorted(glob.glob(os.path.join(args.train_path, "*", "*.parquet")))

    from time import perf_counter
    start_time = perf_counter()

    train_loader = NvtCriteoDataloader(
        train_paths,
        batch_size=args.batch_size,
        world_size=world_size,
        rank=rank,
    ).get_nvt_criteo_dataloader()

    # it = iter(train_loader)
    # for batch in it:
    #     pass
    # stop = perf_counter()
    # print(f"FINISHED, took {stop-start_time}")
    # return

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=none_throws(num_embeddings_per_feature)[feature_idx]
            if num_embeddings_per_feature is not None
            else args.num_embeddings,
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]

    train_model = DLRMTrain(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("meta")
        ),
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=list(map(int, args.dense_arch_layer_sizes.split(","))),
        over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
        dense_device=device,
    )

    # Enable optimizer fusion
    fused_params = {
        "learning_rate": args.learning_rate,
        "optimizer": OptimType.EXACT_ROWWISE_ADAGRAD,
    }

    sharders = cast(
        List[ModuleSharder[nn.Module]],
        [
            EmbeddingBagCollectionSharder(fused_params=fused_params),
        ],
    )

    pg = dist.GroupMember.WORLD
    constraints = defaultdict(lambda: trec_dist.planner.ParameterConstraints())
    for embedding_bag_config in eb_configs:
        constraints[embedding_bag_config.name].sharding_types = ["row_wise"]
        constraints[embedding_bag_config.name].kernel_types = ["batched_fused"]

    hbm_cap = torch.cuda.get_device_properties(device).total_memory
    local_world_size = trec_dist.comm.get_local_size(world_size)
    model = DistributedModelParallel(
        module=train_model,
        device=device,
        env=trec_dist.ShardingEnv.from_process_group(pg),
        plan=trec_dist.planner.EmbeddingShardingPlanner(
            topology=trec_dist.planner.Topology(
                world_size=world_size,
                compute_device=device.type,
                local_world_size=local_world_size,
                hbm_cap=hbm_cap,
                batch_size=args.batch_size,
            ),
            storage_reservation=trec_dist.planner.storage_reservations.HeuristicalStorageReservation(
                percentage=0.25,
            ),
            constraints=constraints,
        ).collective_plan(train_model, sharders, pg),
        sharders=sharders,
    )

    non_fused_optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        lambda params: torch.optim.Adagrad(params, lr=args.learning_rate),
    )

    opt = trec_optim.keyed.CombinedOptimizer(
        [non_fused_optimizer, model.fused_optimizer]
    )

    train_pipeline = TrainPipelineSparseDist(
        model,
        opt,
        device,
    )

    throughput = ThroughputMetric(
        batch_size=args.batch_size,
        world_size=world_size,
        window_seconds=30,
        warmup_steps=10,
    )

    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch}")
        it = iter(train_loader)
        step = 0
        losses = []
        while True:
            try:
                loss, _logits, _labels = train_pipeline.progress(it)
                # predictions = logits.sigmoid()
                # labels = labels.int()
                # ne_metric.update(predictions=predictions, labels=labels, weights=None)

                throughput.update()
                losses.append(loss)

                if step % 100 == 0 and step != 0:
                    throughput_val = throughput.compute()
                    if rank == 0:
                        print("step", step)
                        print("throughput", throughput_val)
                        print(
                            "binary cross entropy loss",
                            torch.mean(torch.stack(losses)) / (args.batch_size),
                        )
                    losses = []
                step += 1

            except StopIteration:
                print("Reached stop iteration")
                break


if __name__ == "__main__":
    main(sys.argv[1:])
