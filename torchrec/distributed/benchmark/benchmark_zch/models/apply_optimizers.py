import argparse
import os

import torch
import torch.nn as nn
import yaml
from benchmark_zch_utils import get_module_from_instance
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter


def apply_sparse_optimizers(model: nn.Module, args: argparse.Namespace) -> None:
    if args.sparse_optim == "adagrad":
        embedding_optimizer = torch.optim.Adagrad
    elif args.sparse_optim == "sgd":
        embedding_optimizer = torch.optim.SGD
    optimizer_kwargs = {"lr": args.learning_rate}
    if args.sparse_optim == "adagrad":
        optimizer_kwargs["eps"] = args.eps
    # get model's embedding module's attribute path
    model_config_file_path = os.path.join(
        os.path.dirname(__file__), "configs", f"{args.model_name}.yaml"
    )
    with open(model_config_file_path, "r") as f:
        model_config = yaml.safe_load(f)
    embedding_module_attribute_path = model_config["embedding_module_attribute_path"]
    apply_optimizer_in_backward(
        embedding_optimizer,
        get_module_from_instance(model, embedding_module_attribute_path).parameters(),
        optimizer_kwargs,
    )


def apply_dense_optimizers(model: nn.Module, args: argparse.Namespace) -> None:
    def optimizer_with_params():
        if args.dense_optim == "adagrad":
            return lambda params: torch.optim.Adagrad(
                params, lr=args.learning_rate, eps=args.eps
            )
        elif args.dense_optim == "sgd":
            return lambda params: torch.optim.SGD(params, lr=args.learning_rate)
        else:
            raise NotImplementedError("Optimizer not supported")

    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        optimizer_with_params(),
    )
    return dense_optimizer


def combine_optimizers(sparse_optimizers, dense_optimizers) -> torch.optim.Optimizer:
    return CombinedOptimizer([sparse_optimizers, dense_optimizers])
