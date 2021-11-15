#!/usr/bin/env python3

# pyre-unsafe

import click
import numpy as np
import tabulate
import torch

try:
    torch.ops.load_library("fbgemm_gpu_py.so")
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings_cpu")


@click.command()
@click.option("--num-ads", default=1024, type=int)
@click.option("--embedding-dimension", default=300, type=int)
@click.option("--ads-tables", default=400, type=int)
@click.option("--iters", default=10, type=int)
@click.option("--p2p_bw", is_flag=True, default=False)
@click.option("--dst-device", default=0, type=int)
def main(num_ads, embedding_dimension, ads_tables, iters, p2p_bw, dst_device) -> None:
    torch.cuda.set_device(dst_device)
    num_gpus = torch.cuda.device_count()
    ad_ds = [embedding_dimension * ads_tables for _ in range(num_gpus)]
    batch_indices = torch.zeros(num_ads).long().cuda()
    pooled_ad_embeddings = [
        torch.randn(
            num_ads, ad_d, dtype=torch.float16, device=torch.device(f"cuda:{i}")
        )
        for i, ad_d in enumerate(ad_ds)
    ]

    def benchmark_torch_function(iters: int, f, *args) -> float:
        f(*args)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iters):
            f(*args)
        end_event.record()
        torch.cuda.synchronize()
        return (start_event.elapsed_time(end_event) * 1.0e-3) / iters
    if p2p_bw:
        print("Pairwise GPU Copy Bandwidth (GB/s)")
        p2p_copy_bw = np.zeros((num_gpus, num_gpus))
        for i in range(num_gpus):
            for j in range(num_gpus):
                with torch.cuda.device(i):
                    t = benchmark_torch_function(
                        iters,
                        lambda: pooled_ad_embeddings[i].copy_(pooled_ad_embeddings[j])
                        if i != j
                        else pooled_ad_embeddings[i].clone(),
                    )
                    p2p_copy_bw[i, j] = pooled_ad_embeddings[i].numel() * 2 / t / 1.0e9
        table = tabulate.tabulate(
            p2p_copy_bw,
            headers=[f"GPU {i}" for i in range(num_gpus)],
            tablefmt="fancy_grid",
            floatfmt=".0f",
        )
        print(table)

    streams = [torch.cuda.Stream(device=i) for i in range(num_gpus)]
    import contextlib

    with contextlib.ExitStack() as stack:
        for stream in streams:
            stack.enter_context(torch.cuda.stream(stream))

        t = benchmark_torch_function(
            iters,
            lambda: torch.ops.fbgemm.merge_pooled_embeddings(
                pooled_ad_embeddings, batch_indices.size(0), batch_indices.device
            ),
        )
        merged = torch.ops.fbgemm.merge_pooled_embeddings(
            pooled_ad_embeddings, batch_indices.size(0), batch_indices.device
        )
    print(
        f"Merge, B: {num_ads}, D: {embedding_dimension}, T: {ads_tables}, Num GPUs: {num_gpus}, Destination GPU: {dst_device} Output Size: {merged.numel() * 2 / 1.0e6:.2f}MB, BW: {merged.numel() * 2 / t / 1.0e9:.2f}GB/s, t: {t * 1.0e3:.2f}ms"
    )


if __name__ == "__main__":
    main()
