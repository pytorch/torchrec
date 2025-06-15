from typing import Dict

import torch
import torch.nn as nn
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
)

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class BenchmarkMCProbe(nn.Module):
    def __init__(
        self,
        mcec: Dict[str, ManagedCollisionEmbeddingCollection],
        mc_method: str,  # method for managing collisions, one of ["zch", "mpzch"]
    ) -> None:
        super().__init__()
        # self._mcec is a pointer to the mcec object passed in
        self._mcec = mcec
        # record the mc_method
        self._mc_method = mc_method
        # get the output_offsets of the mcec
        self.per_table_output_offsets = (
            {}
        )  # dict of {table_name [str]: output_offsets [torch.Tensor]} TODO: find out relationship between table_name and feature_name
        if self._mc_method == "mpzch":
            for table_name, mcec_module in self._mcec.items():
                self.per_table_output_offsets[table_name] = (
                    mcec_module._output_global_offset_tensor
                )
        # create a dictionary to store the state of mch modules before forward
        self._prev_mch_state = (
            {}
        )  # dictionary of {table_name [str]: {module_name [str]: state [int]}}
        # create a dictionary to store the statistics of mch modules
        self._mch_stats = (
            {}
        )  # dictionary of {table_name [str]: {metric_name [str]: metric_value [int]}}

    def record_mcec_state(self) -> None:
        for table_name, mc_module in self._mcec.items():
            if self._mc_method == "zch":
                self._prev_mch_state[table_name] = {
                    "remapping_table": mc_module._mch_sorted_raw_ids
                }
                num_empty_slots = torch.sum(
                    torch.ge(
                        mc_module._mch_sorted_raw_ids, torch.iinfo(torch.int64).max
                    )
                )
                self._prev_mch_state[table_name]["num_empty_slots"] = num_empty_slots
            elif self._mc_method == "mpzch":
                self._prev_mch_state[table_name] = {
                    "remapping_table": mc_module._hash_zch_identities.to_dense().squeeze()
                }
                num_empty_slots = torch.sum(
                    torch.lt(mc_module._hash_zch_identities, 0)
                )  # num of empty slots equals to the number of -1s in the hash_zch_identities
                self._prev_mch_state[table_name]["num_empty_slots"] = num_empty_slots

    def update(
        self, input_kjt: KeyedJaggedTensor, remapped_ids: KeyedJaggedTensor
    ) -> None:
        remapped_ids = remapped_ids.to_dict()
        for table_name, mch_module in self._mcec.items():
            # initialize the dictionary for this table if it doesn't exist
            if table_name not in self._mch_stats:
                self._mch_stats[table_name] = {}  # dict of {metric_name: metric_value}

            # calculate the number of hits
            if "remapping_table" in self._prev_mch_state[table_name]:
                if "num_hits" not in self._mch_stats[table_name]:
                    self._mch_stats[table_name]["num_hits"] = 0
                prev_remapping_table = self._prev_mch_state[table_name][
                    "remapping_table"
                ]
                for feature_name, remapped_ids_jt in remapped_ids.items():
                    curr_remapped_ids = (
                        remapped_ids_jt.values()
                        - self.per_table_output_offsets[table_name]
                    )
                    prev_remapped_ids = torch.index_select(
                        prev_remapping_table, 0, curr_remapped_ids
                    )
                    self._mch_stats[table_name]["num_hits"] += torch.sum(
                        torch.eq(curr_remapped_ids, prev_remapped_ids)
                    )
            # calculate the number of insertions
            if "num_empty_slots" in self._prev_mch_state[table_name]:
                prev_num_empty_slots = self._prev_mch_state[table_name][
                    "num_empty_slots"
                ]
                num_current_empty_slots = (
                    torch.sum(
                        torch.ge(
                            mch_module._mch_sorted_raw_ids, torch.iinfo(torch.int64).max
                        )
                    )
                    if self._mc_method == "zch"
                    else torch.sum(torch.lt(mch_module._hash_zch_identities, 0))
                )
                self._mch_stats[table_name]["num_insertions"] = (
                    num_current_empty_slots - prev_num_empty_slots
                )
            # calculate the number of total queries
            if "num_queries" not in self._mch_stats[table_name]:
                self._mch_stats[table_name]["num_queries"] = 0
            num_input_values = 0
            for feature_name, input_jt in input_kjt.to_dict().items():
                num_input_values += input_jt.values().numel()
            self._mch_stats[table_name]["num_queries"] += num_input_values
            # calculate number of collisions: number of queries - number of hits - number of insertions
            self._mch_stats[table_name]["num_collisions"] = (
                self._mch_stats[table_name]["num_queries"]
                - self._mch_stats[table_name]["num_hits"]
                - self._mch_stats[table_name]["num_insertions"]
            )
