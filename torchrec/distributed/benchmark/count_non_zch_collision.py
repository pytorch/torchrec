import csv
import json
import multiprocessing
import os
import sys

import numpy as np

import torch
from benchmark_zch_dlrmv2 import parse_args
from data.dlrm_dataloader import get_dataloader
from torch import distributed as dist
from torchrec.test_utils import get_free_port
from tqdm import tqdm


def main(rank, args):
    # seed everything for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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

    train_dataloader = get_dataloader(args, backend, "train")

    # make folder to save the collision dict
    os.makedirs(args.profiling_result_folder, exist_ok=True)

    # collision dict
    collison_dict = {}  # feature_name: {remapped_id: original_id}
    collision_stat = {}  # feature_name: {hit: 0, collision: 0, total: 0}
    hash_value_lookup_table = (
        {}
    )  # feature_name: {original_id: remapped_id} # used to look up the remapped id for the original id to save the time of hashing the original id again
    remapping_tensor_dict = {}  # feature_name: remapping_tensor
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
                "rank_idx",
            ]
        )

    pbar = tqdm(train_dataloader, desc=f"Rank {rank}")
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        for feature_name, feature_values_jt in batch.sparse_features.to_dict().items():
            if feature_name not in collison_dict:
                collison_dict[feature_name] = {}
            if feature_name not in collision_stat:
                collision_stat[feature_name] = {
                    "hit_cnt": 0,
                    "collision_cnt": 0,
                    "total_cnt": 0,
                    "insert_cnt": 0,
                }
            if feature_name not in remapping_tensor_dict:
                remapping_tensor_dict[feature_name] = (
                    torch.zeros(args.num_embeddings, dtype=torch.int64) - 1
                ).to(
                    device
                )  # create a tensor of size [num_embeddings] and initialize it with -1
            num_empty_slots_before_remapping = (
                torch.sum(remapping_tensor_dict[feature_name] == -1).cpu().item()
            )  # count the number of empty slots in the remapping tensor
            if feature_name not in hash_value_lookup_table:
                hash_value_lookup_table[feature_name] = {}
            # create progress bar of feature values
            remapped_tensor_values = torch.zeros_like(feature_values_jt.values())
            input_feature_values = feature_values_jt.values()
            for feature_value_idx in range(len(input_feature_values)):
                feature_value = input_feature_values[feature_value_idx]
                if feature_value.cpu().item() in hash_value_lookup_table[feature_name]:
                    hashed_feature_value = hash_value_lookup_table[feature_name][
                        feature_value.cpu().item()
                    ]
                else:
                    feature_value = feature_value.unsqueeze(0)  # convert to [1, ]
                    feature_value = feature_value.to(torch.uint64)  # convert to uint64
                    hashed_feature_value = torch.ops.fbgemm.murmur_hash3(
                        feature_value, 0, 0
                    )
                    # convert to int64
                    hashed_feature_value = hashed_feature_value.to(
                        torch.int64
                    )  # convert to int64
                    # convert to [0, num_embeddings)
                    hashed_feature_value = (
                        (hashed_feature_value % args.num_embeddings).cpu().item()
                    )  # convert to [0, num_embeddings)
                    # save the hashed feature value to the lookup table
                    hash_value_lookup_table[feature_name][
                        feature_value.cpu().item()
                    ] = hashed_feature_value
                remapped_tensor_values[feature_value_idx] = hashed_feature_value
                # check if the remapping_tensor_dict at remapped_value's indexed slot value is -1
                if remapping_tensor_dict[feature_name][hashed_feature_value] == -1:
                    # if the remapping_tensor_dict at remapped_value's indexed slot value is -1, update the remapping_tensor_dict at remapped_value's indexed slot value to feature_value
                    remapping_tensor_dict[feature_name][
                        hashed_feature_value
                    ] = feature_value
            # check if the hashed feature value is in the collision dict
            num_empty_slots_after_remapping = (
                torch.sum(remapping_tensor_dict[feature_name] == -1).cpu().item()
            )  # count the number of empty slots in the remapping tensor
            insert_cnt = (
                num_empty_slots_before_remapping - num_empty_slots_after_remapping
            )
            hit_cnt = (
                (
                    torch.sum(
                        torch.eq(
                            input_feature_values,
                            remapping_tensor_dict[feature_name][remapped_tensor_values],
                        )
                    )
                    - insert_cnt
                )
                .cpu()
                .item()
            )
            total_cnt = len(input_feature_values)
            collision_cnt = total_cnt - hit_cnt - insert_cnt
            collision_stat[feature_name]["hit_cnt"] = hit_cnt
            collision_stat[feature_name]["collision_cnt"] = collision_cnt
            collision_stat[feature_name]["total_cnt"] = total_cnt
            collision_stat[feature_name]["insert_cnt"] = insert_cnt

        # save the collision stat
        with open(zch_metrics_file_path, "a") as f:
            writer = csv.writer(f)
            for feature_name, stats in collision_stat.items():
                hit_rate = stats["hit_cnt"] / stats["total_cnt"]
                insert_rate = stats["insert_cnt"] / stats["total_cnt"]
                collision_rate = stats["collision_cnt"] / stats["total_cnt"]
                writer.writerow(
                    [
                        0,
                        batch_idx,
                        feature_name,
                        stats["hit_cnt"],
                        stats["total_cnt"],
                        stats["insert_cnt"],
                        stats["collision_cnt"],
                        hit_rate,
                        insert_rate,
                        collision_rate,
                        rank,
                    ]
                )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    # set environment variables
    os.environ["MASTER_ADDR"] = str("localhost")
    os.environ["MASTER_PORT"] = str(get_free_port())
    # set a multiprocessing context
    ctx = multiprocessing.get_context("spawn")
    # create a process to perform benchmarking
    processes = []
    for rank in range(int(os.environ["WORLD_SIZE"])):
        p = ctx.Process(
            target=main,
            args=(rank, args),
        )
        p.start()
        processes.append(p)
