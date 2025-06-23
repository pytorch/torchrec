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

from arguments import parse_args

from benchmark_zch_utils import BenchmarkMCProbe, get_module_from_instance

from data.get_dataloader import get_dataloader
from models import make_model
from models.apply_optimizers import (
    apply_dense_optimizers,
    apply_sparse_optimizers,
    combine_optimizers,
)
from models.shard_model import shard_model
from torch import distributed as dist
from torch.utils.data import DataLoader

from torchrec.test_utils import get_free_port
from tqdm import tqdm


def main(rank: int, args: argparse.Namespace, queue: multiprocessing.Queue) -> None:
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

    # END TEST FOR DATASET HASH

    # get dataset
    train_dataloader = get_dataloader(args.dataset_name, args, "train")
    test_dataloader = get_dataloader(args.dataset_name, args, "test")

    # make the model
    model, model_configs = make_model(args.model_name, args, device)

    # apply optimizer to the sparse arch of the model
    apply_sparse_optimizers(model, args)

    # shard the model
    model = shard_model(model, device, args)

    # apply optimizer to the dense arch of the model
    dense_optimizer = apply_dense_optimizers(model, args)

    # combine the sparse and dense optimizers
    optimizer = combine_optimizers(model.fused_optimizer, dense_optimizer)

    benchmark_probe = None
    if len(args.zch_method) > 0:
        benchmark_probe = BenchmarkMCProbe(
            mcec=get_module_from_instance(
                model._dmp_wrapped_module,
                model_configs["managed_collision_module_attribute_path"],
            ),
            mc_method=args.zch_method,
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
            # batch = batch.to(device)
            batch = batch.to(device)
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            if len(args.zch_method) > 0:
                benchmark_probe.record_mcec_state(stage="before_fwd")
            # zero the gradients
            optimizer.zero_grad()
            # forward pass
            starter.record()
            loss, (loss_values, pred_logits, labels) = model(batch)
            ender.record()
            loss.backward()
            optimizer.step()
            # do statistics
            num_queries_per_batch = len(labels)
            starter_list.append(starter)
            ender_list.append(ender)
            num_queries_per_batch_list.append(num_queries_per_batch)
            if len(args.zch_method) > 0:
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
        # after each epoch, do validation
        eval_result_dict = evaluation(model, test_dataloader, device)
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


def evaluation(model: nn.Module, data_loader: DataLoader, device: torch.device):
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
            loss, (loss_values, pred_logits, labels) = model(batch)
            preds = torch.sigmoid(pred_logits)
            preds_reshaped = torch.stack((1 - preds, preds), dim=1)
            # update auroc
            auroc.update(preds_reshaped, labels)
            # calculate log loss
            batch_log_loss_list = -(
                (1 + labels) / (2 * torch.log(preds))
                + (1 - labels) / (2 * torch.log(1 - preds))
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


def statistic(args: argparse.Namespace, queue: multiprocessing.Queue) -> None:
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
    if len(args.zch_method) > 0:
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
            if len(args.zch_method) == 0:
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
            per_batch_time_list = msg_content["per_batch_time_list"]
            per_batch_num_queries_list = msg_content["per_batch_num_queries_list"]
            # save the qps statistics to a file
            with open(qps_metrics_file_path, "a") as f:
                writer = csv.writer(f)
                for i in range(len(per_batch_time_list)):
                    writer.writerow(
                        [
                            epoch_idx,
                            str(interval_start_batch_idx + i),
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
    # # create a process to perform statistic calculations
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
