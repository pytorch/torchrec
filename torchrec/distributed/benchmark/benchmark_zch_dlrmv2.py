import argparse
import multiprocessing
import os
import sys
from typing import cast, Iterator, List, Optional

import torch
import torch.nn as nn
from pyre_extensions import none_throws
from torch import distributed as dist

from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    DAYS,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    InMemoryBinaryCriteoIterDataPipe,
)
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.mc_modules import ManagedCollisionCollectionSharder

from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology

from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)

from torchrec.distributed.types import ModuleSharder
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.hash_mc_evictions import (
    HashZchEvictionConfig,
    HashZchEvictionPolicyName,
)
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.mc_adapter import McEmbeddingBagCollectionAdapter

from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
)

from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
)
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.test_utils import get_free_port
from tqdm import tqdm

from .benchmark_zch_utils import BenchmarkMCProbe

from .data.dlrm_dataloader import get_dataloader


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--drop_last_training_batch",
        dest="drop_last_training_batch",
        action="store_true",
        help="Drop the last non-full training batch",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="batch size to use for validation and testing",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="number of validation batches",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="number of test batches",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["criteo_1t", "criteo_kaggle"],
        default="criteo_kaggle",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle",
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
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--interaction_branch1_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch1 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--interaction_branch2_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch2 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="Number of DCN layers in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="Low rank dimension for DCN in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--undersampling_rate",
        type=float,
        help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
        " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
        " All rows with label 1 will be kept. Value should be between 0 and 1."
        " When not supplied, no undersampling occurs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="--mmap_mode mmaps the dataset."
        " That is, the dataset is kept on disk but is accessed as if it were in memory."
        " --mmap_mode is intended mostly for faster debugging. Use --mmap_mode to bypass"
        " preloading the dataset when preloading takes too long or when there is "
        " insufficient memory available to load the full dataset.",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the Criteo dataset npy files.",
    )
    parser.add_argument(
        "--synthetic_multi_hot_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the MLPerf v2 synthetic multi-hot dataset npz files.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon for Adagrad optimizer.",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--shuffle_training_set",
        dest="shuffle_training_set",
        action="store_true",
        help="Shuffle the training set in memory. This will override mmap_mode",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Frequency at which validation will be run within an epoch.",
    )
    parser.set_defaults(
        pin_memory=None,
        mmap_mode=None,
        drop_last=None,
        shuffle_batches=None,
        shuffle_training_set=None,
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    parser.add_argument(
        "--collect_multi_hot_freqs_stats",
        dest="collect_multi_hot_freqs_stats",
        action="store_true",
        help="Flag to determine whether to collect stats on freq of embedding access.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default=None,
        help="Comma separated multihot size per sparse feature. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_distribution_type",
        type=str,
        choices=["uniform", "pareto"],
        default=None,
        help="Multi-hot distribution options.",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_decay_start", type=int, default=0)
    parser.add_argument("--lr_decay_steps", type=int, default=0)
    parser.add_argument(
        "--print_lr",
        action="store_true",
        help="Print learning rate every iteration.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TensorFloat-32 mode for matrix multiplications on A100 (or newer) GPUs.",
    )
    parser.add_argument(
        "--print_sharding_plan",
        action="store_true",
        help="Print the sharding plan used for each embedding table.",
    )
    return parser.parse_args(argv)


def main(rank, args):
    import fbvscode

    fbvscode.set_trace()
    # setup environment
    os.environ["RANK"] = str(rank)
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    # get dataset
    train_dataloader = get_dataloader(args, backend, "train")

    # create embedding configs
    ebc_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=(
                none_throws(args.num_embeddings_per_feature)[feature_idx]
                if args.num_embeddings is None
                else args.num_embeddings
            ),
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]

    # mc_modules = {}
    # for table_config in ebc_configs:
    #     table_name = table_config.name
    #     mc_modules[table_name] = HashZchManagedCollisionModule(  # MPZCH
    #         is_inference=False,
    #         zch_size=(table_config.num_embeddings),
    #         input_hash_size=1000,
    #         device=device,
    #         total_num_buckets=4,
    #         eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
    #         eviction_config=HashZchEvictionConfig(
    #             features=table_config.feature_names,
    #             single_ttl=2,
    #         ),
    #     )

    # ebc = EmbeddingBagCollection(tables=ebc_configs, device=torch.device("meta"))

    # managed_collision_ebc = ManagedCollisionEmbeddingBagCollection(  # ZCH or not
    #     embedding_bag_collection=ebc,
    #     managed_collision_collection=ManagedCollisionCollection(
    #         managed_collision_modules=mc_modules,
    #         embedding_configs=ebc.embedding_bag_configs(),
    #     ),
    #     allow_in_place_embed_weight_update=False,
    #     return_remapped_features=False,  # not return remapped features
    # )

    # get managed collision embedding bag collection
    # ebc = EmbeddingBagCollection(tables=ebc_configs, device=torch.device("meta"))
    managed_collision_ebc = (
        McEmbeddingBagCollectionAdapter(  # TODO: add switch for other ZCH or no ZCH
            tables=ebc_configs,
            input_hash_size=args.num_embeddings,
            device=torch.device("meta"),
            use_mpzch=True,
        )
    )

    # create model
    dlrm_model = DLRM(
        embedding_bag_collection=managed_collision_ebc,
        # embedding_bag_collection=ebc,
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=[int(x) for x in args.dense_arch_layer_sizes.split(",")],
        over_arch_layer_sizes=[int(x) for x in args.over_arch_layer_sizes.split(",")],
        dense_device=device,
    )

    print(dlrm_model)
    train_model = DLRMTrain(dlrm_model)

    # apply optimizer to the model
    embedding_optimizer = torch.optim.Adagrad if args.adagrad else torch.optim.SGD
    optimizer_kwargs = {"lr": args.learning_rate}
    if args.adagrad:
        optimizer_kwargs["eps"] = args.eps
    apply_optimizer_in_backward(
        embedding_optimizer,
        train_model.model.sparse_arch.embedding_bag_collection.parameters(),
        optimizer_kwargs,
    )

    # shard the model
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
        ),
        batch_size=args.batch_size,
        # If experience OOM, increase the percentage. see
        # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
    )

    sharders = get_default_sharders()
    sharders.append(cast(ModuleSharder[nn.Module], ManagedCollisionCollectionSharder()))

    plan = planner.collective_plan(train_model, sharders, dist.GroupMember.WORLD)

    model = DistributedModelParallel(
        module=train_model,
        device=device,
        plan=plan,
    )

    benchmark_probe = BenchmarkMCProbe(
        mcec=model._dmp_wrapped_module.module.model.sparse_arch.embedding_bag_collection.mc_embedding_bag_collection._managed_collision_collection._managed_collision_modules,
        mc_method="mpzch",
    )

    interval_num_batches_show_qps = 100

    total_time_in_training = 0
    total_num_queries_in_training = 0

    # train the model
    model.train()
    for epoch in range(args.epochs):
        starter_list = []
        ender_list = []
        num_queries_per_batch_list = []
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            benchmark_probe.record_mcec_state()
            # forward pass
            starter.record()
            loss, outputs = model(batch)
            ender.record()
            loss.backward()
            # do statistics
            num_queries_per_batch = len(batch.labels)
            starter_list.append(starter)
            ender_list.append(ender)
            num_queries_per_batch_list.append(num_queries_per_batch)
            if batch_idx % interval_num_batches_show_qps == 0 or batch_idx == len(
                train_dataloader
            ):
                if batch_idx == 0:
                    # skip the first batch since it is not a full batch
                    continue
                # synchronize all the threads to get the exact number of batches
                torch.cuda.synchronize()
                total_time_in_interval = 0
                for i in range(interval_num_batches_show_qps):
                    total_time_in_interval += starter_list[i].elapsed_time(
                        ender_list[i]
                    )
                total_time_in_interval /= 1000  # convert to seconds
                total_num_queries_in_interval = sum(num_queries_per_batch_list)
                qps_per_interval = (
                    total_num_queries_in_interval / total_time_in_interval
                )
                total_time_in_training += total_time_in_interval
                total_num_queries_in_training += total_num_queries_in_interval
                # get benchmark results
                benchmark_probe.update(
                    batch.sparse_features,
                    model._dmp_wrapped_module.module.model.sparse_arch.embedding_bag_collection.remapped_ids,
                )
                print(f"benchmark zch metrics: {benchmark_probe._mch_stats}")
                pbar.set_postfix(
                    {
                        "QPS": qps_per_interval,
                    }
                )
                pbar.update(interval_num_batches_show_qps)
                # reset the lists
                starter_list = []
                ender_list = []
                num_queries_per_batch_list = []

    print("Total time in training: ", total_time_in_training)
    print("Total number of queries in training: ", total_num_queries_in_training)
    print("Average QPS: ", total_num_queries_in_training / total_time_in_training)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    # set environment variables
    os.environ["MASTER_ADDR"] = str("localhost")
    os.environ["MASTER_PORT"] = str(get_free_port())

    ctx = multiprocessing.get_context("spawn")
    processes = []
    for rank in range(int(os.environ["WORLD_SIZE"])):
        p = ctx.Process(
            target=main,
            args=(
                rank,
                args,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
