from typing import Dict

import torch.nn as nn
from torchrec.distributed.types import ShardingPlan


__all__ = ["_get_sharded_modules_recursive"]


def _get_sharded_modules_recursive(
    module: nn.Module,
    path: str,
    plan: ShardingPlan,
) -> Dict[str, nn.Module]:
    """
    Get all sharded modules of module from `plan`.
    """
    params_plan = plan.get_plan_for_module(path)
    if params_plan:
        return {path: (module, params_plan)}

    res = {}
    for name, child in module.named_children():
        new_path = f"{path}.{name}" if path else name
        res.update(_get_sharded_modules_recursive(child, new_path, plan))
    return res
