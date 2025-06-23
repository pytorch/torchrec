import argparse
import csv
import json
import multiprocessing
import os
import sys
import time

from typing import cast, Dict, Iterator, List, Optional

import numpy as np

import torch
import torch.nn as nn
import torchmetrics  # @manual=fbsource//third-party/pypi/torchmetrics:torchmetrics
from pyre_extensions import none_throws
from torch import distributed as dist
from torch.utils.data import DataLoader

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

from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
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
        default=4096,
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
        "--num_embeddings",  # ratio of feature ids to embedding table size # 3 axis: x-bath_idx; y-collisions; zembedding table sizes
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
        default=0,
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
    parser.add_argument(
        "--input_hash_size",
        type=int,
        default=100_000,
        help="Input feature value range",
    )
    parser.add_argument(
        "--num_buckets",
        type=int,
        default=4,
        help="Number of buckets for identity table",
    )
    parser.add_argument(
        "--profiling_result_folder",
        type=str,
        default="profiling_result",
        help="Folder to save profiling results",
    )
    parser.add_argument(
        "--use_zch",
        action="store_true",
        help="If use zch or not",
    )
    return parser.parse_args(argv)


def hash_kjt(
    sparse_features: KeyedJaggedTensor, num_embeddings: int
) -> KeyedJaggedTensor:
    """
    convert the values in the input sparse_features to hashed ones in the range of [0, num_embeddings)
    """
    hashed_feature_values_dict = {}  # {feature_name: hashed_feature_values}
    for feature_name, feature_values_jt in sparse_features.to_dict().items():
        hashed_feature_values_dict[feature_name] = []
        for feature_value in feature_values_jt.values():
            feature_value = feature_value.unsqueeze(0)  # convert to [1, ]
            feature_value = feature_value.to(torch.uint64)  # convert to uint64
            hashed_feature_value = torch.ops.fbgemm.murmur_hash3(feature_value, 0, 0)
            # convert to int64
            hashed_feature_value = hashed_feature_value.to(
                torch.int64
            )  # convert to int64
            # convert to [0, num_embeddings)
            hashed_feature_value = (
                hashed_feature_value % num_embeddings
            )  # convert to [0, num_embeddings)
            # convert to [1, ]
            hashed_feature_value = hashed_feature_value.unsqueeze(0)  # convert to [1, ]
            hashed_feature_values_dict[feature_name].append(hashed_feature_value)
        hashed_feature_values_dict[feature_name] = JaggedTensor.from_dense(
            hashed_feature_values_dict[feature_name]
        )
    # convert to [batch_size, ]
    hashed_feature_kjt = KeyedJaggedTensor.from_jt_dict(hashed_feature_values_dict)
    return hashed_feature_kjt


# def hash_kjt(
#     sparse_features: KeyedJaggedTensor, num_embeddings: int
# ) -> KeyedJaggedTensor:
#     """
#     convert the values in the input sparse_features to hashed ones in the range of [0, num_embeddings)
#     """
#     hashed_feature_values_dict = {}  # {feature_name: hashed_feature_values}
#     for feature_name, feature_values_jt in sparse_features.to_dict().items():
#         hashed_feature_values_dict[feature_name] = []
#         feature_values = feature_values_jt.values()
#         feature_value = feature_values.to(torch.uint64)  # convert to uint64
#         hashed_feature_value = torch.ops.fbgemm.murmur_hash3(feature_value, 0, 0)
#         # convert to int64
#         hashed_feature_value = hashed_feature_value.to(torch.int64)  # convert to int64
#         # convert to [0, num_embeddings)
#         hashed_feature_value = (
#             hashed_feature_value % num_embeddings
#         )  # convert to [0, num_embeddings)
#         # convert to [1, ]
#         # hashed_feature_value = hashed_feature_value.unsqueeze(0)  # convert to [1, ]
#         # hashed_feature_values_dict[feature_name].append(hashed_feature_value)
#         hashed_feature_values_dict[feature_name] = JaggedTensor.from_dense(
#             hashed_feature_value
#         )
#     # convert to [batch_size, ]
#     hashed_feature_kjt = KeyedJaggedTensor.from_jt_dict(hashed_feature_values_dict)
#     return hashed_feature_kjt


def main(rank: int, args: argparse.Namespace, queue: multiprocessing.Queue) -> None:
    # seed everything for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # convert input hash size to num_embeddings if not using zch
    if not args.use_zch:
        args.input_hash_size = args.num_embeddings

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

    # TEST FOR DATASET HASH
    # train_dataloader = get_dataloader(args, backend, "train")
    # for batch in train_dataloader:
    #     batch = batch.to(device)
    #     print("before hash", batch.sparse_features.to_dict()["cat_0"].values()[:5])
    #     batch.sparse_features = hash_kjt(batch.sparse_features, args.num_embeddings)
    #     print("after hash", batch.sparse_features.to_dict()["cat_0"].values()[:5])
    #     break

    # exit(0)

    # END TEST FOR DATASET HASH

    # get dataset
    train_dataloader = get_dataloader(args, backend, "train")
    val_dataloader = get_dataloader(args, backend, "val")
    test_dataloader = get_dataloader(args, backend, "test")

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

    # get managed collision embedding bag collection
    if args.use_zch:
        ebc = (
            McEmbeddingBagCollectionAdapter(  # TODO: add switch for other ZCH or no ZCH
                tables=ebc_configs,
                input_hash_size=args.input_hash_size,
                device=torch.device("meta"),
                world_size=get_local_size(),
                use_mpzch=True,
                mpzch_num_buckets=args.num_buckets,
            )
        )
    else:
        ebc = EmbeddingBagCollection(tables=ebc_configs, device=torch.device("meta"))

    # create model
    dlrm_model = DLRM(
        embedding_bag_collection=ebc,
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

    collision_remapping_tensor_dict = (
        {}
    )  # feature_name: collision_tensor filled with all -1s with num_embedding size at the beginning, used only for non-zch case
    benchmark_probe = None
    if args.use_zch:
        benchmark_probe = BenchmarkMCProbe(
            mcec=model._dmp_wrapped_module.module.model.sparse_arch.embedding_bag_collection.mc_embedding_bag_collection._managed_collision_collection._managed_collision_modules,
            mc_method="mpzch",
            rank=rank,
        )

    interval_num_batches_show_qps = 50

    total_time_in_training = 0
    total_num_queries_in_training = 0

    # train the model
    for epoch_idx in range(args.epochs):
        model.train()
        starter_list = []
        ender_list = []
        num_queries_per_batch_list = []
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_idx}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            if args.use_zch:
                benchmark_probe.record_mcec_state(stage="before_fwd")
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
            if args.use_zch:
                benchmark_probe.record_mcec_state(stage="after_fwd")
                # update zch statistics
                benchmark_probe.update()
                # push the zch stats to the queue
                msg_content = {
                    "epoch_idx": epoch_idx,
                    "batch_idx": batch_idx,
                    "rank": rank,
                    "mch_stats": benchmark_probe.get_mch_stats(),
                }
                queue.put(
                    ("mch_stats", msg_content),
                )
            if batch_idx % interval_num_batches_show_qps == 0 or batch_idx == len(
                train_dataloader
            ):
                if batch_idx == 0:
                    # skip the first batch since it is not a full batch
                    continue
                # synchronize all the threads to get the exact number of batches
                torch.cuda.synchronize()
                # calculate the qps
                # NOTE: why do this qps calculation every interval_num_batches_show_qps batches?
                #   because performing this calculation needs to synchronize all the ranks by calling torch.cuda.synchronize()
                #   and this is a heavy operation (takes several milliseconds). So we only do this calculation every
                #   interval_num_batches_show_qps batches to reduce the overhead.
                ## get per batch time list by calculating the time difference between the start and end events of each batch
                per_batch_time_list = []
                for i in range(interval_num_batches_show_qps):
                    per_batch_time_list.append(
                        starter_list[i].elapsed_time(ender_list[i]) / 1000
                    )  # convert to seconds by dividing by 1000
                ## calculate the total time in the interval
                total_time_in_interval = sum(per_batch_time_list)
                ## calculate the total number of queries in the interval
                total_num_queries_in_interval = sum(num_queries_per_batch_list)
                ## fabricate the message and total_num_queries_in_interval to the queue
                interval_start_batch_idx = (
                    batch_idx - interval_num_batches_show_qps
                )  # the start batch index of the interval
                interval_end_batch_idx = (
                    batch_idx  # the end batch index of the interval
                )
                ## fabricate the message content
                msg_content = {
                    "epoch_idx": epoch_idx,
                    "rank": rank,
                    "interval_start_batch_idx": interval_start_batch_idx,
                    "interval_end_batch_idx": interval_end_batch_idx,
                    "per_batch_time_list": per_batch_time_list,
                    "per_batch_num_queries_list": num_queries_per_batch_list,
                }
                ## put the message into the queue
                queue.put(("duration_and_num_queries", msg_content))
                qps_per_interval = (
                    total_num_queries_in_interval / total_time_in_interval
                )
                total_time_in_training += total_time_in_interval
                total_num_queries_in_training += total_num_queries_in_interval
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
            # if batch_idx > 2:
            #     time.sleep(5)
            #     queue.put(("finished", {"rank": rank}))
            #     print("finished")
            #     exit(0)
        # after each epoch, do validation
        eval_result_dict = evaluation(model, val_dataloader, device)
        # print the evaluation result
        print(f"Epoch {epoch_idx} validation result: {eval_result_dict}")
        # send the evaluation result to the queue
        msg_content = {
            "epoch_idx": epoch_idx,
            "rank": rank,
            "eval_result_dict": eval_result_dict,
        }
        queue.put(("eval_result", msg_content))

    time.sleep(30)
    queue.put(("finished", {"rank": rank}))
    print("finished")
    return

    # print("Total time in training: ", total_time_in_training)
    # print("Total number of queries in training: ", total_num_queries_in_training)
    # print("Average QPS: ", total_num_queries_in_training / total_time_in_training)


def evaluation(model: DLRMTrain, data_loader: DataLoader, device: torch.device):
    """
    Evaluate the model on the given data loader.
    """
    model.eval()
    auroc = torchmetrics.AUROC(task="multiclass", num_classes=2).to(device)
    log_loss_list = []
    label_val_sums = 0
    num_labels = 0
    for batch in tqdm(data_loader):
        batch = batch.to(device)
        with torch.no_grad():
            loss, outputs = model(batch)
            loss_val, logits, labels = outputs
            preds = torch.sigmoid(logits)
            preds_reshaped = torch.stack((1 - preds, preds), dim=1)
            auroc.update(preds_reshaped, labels)
            # calculate log loss
            batch_log_loss_list = -(
                (1 + labels) / 2 * torch.log(preds)
                + (1 - labels) / 2 * torch.log(1 - preds)
            )
            log_loss_list.extend(batch_log_loss_list.tolist())
            label_val_sums += labels.sum().item()
            num_labels += labels.shape[0]
    auroc_result = auroc.compute().item()
    # calculate ne as mean(log_loss_list) / log_loss(avg_label)
    avg_label = label_val_sums / num_labels
    avg_label = torch.tensor(avg_label).to(device)
    avg_label_log_loss = -(
        avg_label * torch.log(avg_label) + (1 - avg_label) * torch.log(1 - avg_label)
    )
    ne = torch.mean(torch.tensor(log_loss_list)).item() / avg_label_log_loss.item()
    print(f"AUROC: {auroc_result}, NE: {ne}")
    eval_result_dict = {
        "auroc": auroc_result,
        "ne": ne,
    }
    return eval_result_dict


def statistic(args: argparse.Namespace, queue: multiprocessing.Queue):
    """
    The process to perform statistic calculations
    """
    mch_buffer = (
        {}
    )  # {epcoh_idx:{end_batch_idx: {rank: data_dict}}} where data dict is {metric_name: metric_value}
    num_processed_batches = 0  # counter of the number of processed batches
    world_size = int(os.environ["WORLD_SIZE"])  # world size
    finished_counter = 0  # counter of the number of finished processes

    # create a profiling result folder
    os.makedirs(args.profiling_result_folder, exist_ok=True)
    # create a csv file to save the zch_metrics
    if args.use_zch:
        zch_metrics_file_path = os.path.join(
            args.profiling_result_folder, "zch_metrics.csv"
        )
        with open(zch_metrics_file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch_idx",
                    "batch_idx",
                    "feature_name",
                    "hit_cnt",
                    "total_cnt",
                    "insert_cnt",
                    "collision_cnt",
                    "hit_rate",
                    "insert_rate",
                    "collision_rate",
                    "rank_total_cnt",
                ]
            )
    # create a csv file to save the qps_metrics
    qps_metrics_file_path = os.path.join(
        args.profiling_result_folder, "qps_metrics.csv"
    )
    with open(qps_metrics_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch_idx",
                "batch_idx",
                "rank",
                "num_queries",
                "duration",
                "qps",
            ]
        )
    # create a csv file to save the eval_metrics
    eval_metrics_file_path = os.path.join(
        args.profiling_result_folder, "eval_metrics.csv"
    )
    with open(eval_metrics_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch_idx",
                "rank",
                "auroc",
                "ne",
            ]
        )

    while finished_counter < world_size:
        try:
            # get the data from the queue
            msg_type, msg_content = queue.get(
                timeout=0.5
            )  # data are put into the queue im the form of (msg_type, epoch_idx, batch_idx, rank, rank_data_dict)
        except Exception as e:
            # if the queue is empty, check if all the processes have finished
            # if finished_counter >= world_size:
            #     print(f"All processes have finished. {finished_counter} / {world_size}")
            #     break
            # else:
            #     continue  # keep waiting for the queue to be filled
            # if queue is empty, check if all the processes have finished
            if finished_counter >= world_size:
                print(f"All processes have finished. {finished_counter} / {world_size}")
                break
            else:
                continue  # keep waiting for the queue to be filled
        # when getting the data, check if the data is from the last batch
        if (
            msg_type == "finished"
        ):  # if the message type is "finished", the process has finished
            rank = msg_content["rank"]
            finished_counter += 1
            print(f"Process {rank} has finished. {finished_counter} / {world_size}")
            continue
        elif msg_type == "mch_stats":
            if not args.use_zch:
                continue
            epoch_idx = msg_content["epoch_idx"]
            batch_idx = msg_content["batch_idx"]
            rank = msg_content["rank"]
            rank_batch_mch_stats = msg_content["mch_stats"]
            # other wise, aggregate the data into the buffer
            if epoch_idx not in mch_buffer:
                mch_buffer[epoch_idx] = {}
            if batch_idx not in mch_buffer[epoch_idx]:
                mch_buffer[epoch_idx][batch_idx] = {}
            mch_buffer[epoch_idx][batch_idx][rank] = rank_batch_mch_stats
            num_processed_batches += 1
            # check if we have all the data from all the ranks for a batch in an epoch
            # if we have all the data, combine the data from all the ranks
            if len(mch_buffer[epoch_idx][batch_idx]) == world_size:
                # create a dictionary to store the statistics for each batch
                batch_stats = (
                    {}
                )  # {feature_name: {hit_cnt: 0, total_cnt: 0, insert_cnt: 0, collision_cnt: 0}}
                # combine the data from all the ranks
                for mch_stats_rank_idx in mch_buffer[epoch_idx][batch_idx].keys():
                    rank_batch_mch_stats = mch_buffer[epoch_idx][batch_idx][
                        mch_stats_rank_idx
                    ]
                    # for each feature table in the mch stats information
                    for mch_stats_feature_name in rank_batch_mch_stats.keys():
                        # create the dictionary for the feature table if not created
                        if mch_stats_feature_name not in batch_stats:
                            batch_stats[mch_stats_feature_name] = {
                                "hit_cnt": 0,
                                "total_cnt": 0,
                                "insert_cnt": 0,
                                "collision_cnt": 0,
                                "rank_total_cnt": {},  # dictionary of {rank_idx: num_quries_mapped_to_the_rank}
                            }
                        # aggregate the data from all the ranks
                        ## aggregate the hit count
                        batch_stats[mch_stats_feature_name][
                            "hit_cnt"
                        ] += rank_batch_mch_stats[mch_stats_feature_name]["hit_cnt"]
                        ## aggregate the total count
                        batch_stats[mch_stats_feature_name][
                            "total_cnt"
                        ] += rank_batch_mch_stats[mch_stats_feature_name]["total_cnt"]
                        ## aggregate the insert count
                        batch_stats[mch_stats_feature_name][
                            "insert_cnt"
                        ] += rank_batch_mch_stats[mch_stats_feature_name]["insert_cnt"]
                        ## aggregate the collision count
                        batch_stats[mch_stats_feature_name][
                            "collision_cnt"
                        ] += rank_batch_mch_stats[mch_stats_feature_name][
                            "collision_cnt"
                        ]
                        ## for rank total count, get the data from the rank data dict
                        batch_stats[mch_stats_feature_name]["rank_total_cnt"][
                            mch_stats_rank_idx
                        ] = rank_batch_mch_stats[mch_stats_feature_name][
                            "rank_total_cnt"
                        ]
                # clear the buffer for the batch
                del mch_buffer[epoch_idx][batch_idx]
                # save the zch statistics to a file
                with open(zch_metrics_file_path, "a") as f:
                    writer = csv.writer(f)
                    for feature_name, stats in batch_stats.items():
                        hit_rate = stats["hit_cnt"] / stats["total_cnt"]
                        insert_rate = stats["insert_cnt"] / stats["total_cnt"]
                        collision_rate = stats["collision_cnt"] / stats["total_cnt"]
                        rank_total_cnt = json.dumps(stats["rank_total_cnt"])
                        writer.writerow(
                            [
                                epoch_idx,
                                batch_idx,
                                feature_name,
                                stats["hit_cnt"],
                                stats["total_cnt"],
                                stats["insert_cnt"],
                                stats["collision_cnt"],
                                hit_rate,
                                insert_rate,
                                collision_rate,
                                rank_total_cnt,
                            ]
                        )
        elif msg_type == "duration_and_num_queries":
            epoch_idx = msg_content["epoch_idx"]
            rank = msg_content["rank"]
            interval_start_batch_idx = msg_content["interval_start_batch_idx"]
            interval_end_batch_idx = msg_content["interval_end_batch_idx"]
            per_batch_time_list = msg_content["per_batch_time_list"]
            per_batch_num_queries_list = msg_content["per_batch_num_queries_list"]
            # save the qps statistics to a file
            with open(qps_metrics_file_path, "a") as f:
                writer = csv.writer(f)
                for i in range(len(per_batch_time_list)):
                    writer.writerow(
                        [
                            epoch_idx,
                            str(interval_end_batch_idx + i),
                            rank,
                            per_batch_num_queries_list[i],
                            per_batch_time_list[i],
                            (
                                per_batch_num_queries_list[i] / per_batch_time_list[i]
                                if per_batch_time_list[i] > 0
                                else 0
                            ),
                        ]
                    )
        elif msg_type == "eval_result":
            epoch_idx = msg_content["epoch_idx"]
            rank = msg_content["rank"]
            eval_result_dict = msg_content["eval_result_dict"]
            # save the evaluation result to a file
            with open(eval_metrics_file_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch_idx,
                        rank,
                        eval_result_dict["auroc"],
                        eval_result_dict["ne"],
                    ]
                )
        else:
            # raise a warning if the message type is not recognized
            print("Warning: Unknown message type")
            continue


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    # set environment variables
    os.environ["MASTER_ADDR"] = str("localhost")
    os.environ["MASTER_PORT"] = str(get_free_port())
    # set a multiprocessing context
    ctx = multiprocessing.get_context("spawn")
    # create a queue to communicate between processes
    queue = ctx.Queue()
    # create a process to perform statistic calculations
    stat_process = ctx.Process(
        target=statistic, args=(args, queue)
    )  # create a process to perform statistic calculations
    stat_process.start()
    # create a process to perform benchmarking
    train_processes = []
    for rank in range(int(os.environ["WORLD_SIZE"])):
        p = ctx.Process(
            target=main,
            args=(rank, args, queue),
        )
        p.start()
        train_processes.append(p)

    # wait for the training processes to finish
    for p in train_processes:
        p.join()
    # wait for the statistic process to finish
    stat_process.join()
