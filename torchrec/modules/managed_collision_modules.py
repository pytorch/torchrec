#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import abc
from typing import Any, Callable, cast, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist

from torch import nn
from torchrec.sparse.jagged_tensor import JaggedTensor


class ManagedCollisionModule(nn.Module):
    """
    Abstract base class for ManagedCollisionModule.
    Maps input ids to range [0, max_output_id).

    Args:
        max_output_id (int): Max output value of remapped ids.
        max_input_id (int): Max value of input range i.e. [0, max_input_id)
        remapping_range_start_index (int): Relative start index of remapping range
        device (torch.device): default compute device.

    Example::
        jt = JaggedTensor(...)
        mcm = ManagedCollisionModule(...)
        mcm_jt = mcm(fp)
    """

    def __init__(
        self,
        max_output_id: int,
        device: torch.device,
        remapping_range_start_index: int = 0,
        max_input_id: int = 2**40,
    ) -> None:
        # slots is the number of rows to map from global id to
        # for example, if we want to manage 1000 ids to 10 slots
        super().__init__()
        self._max_input_id: int = max_input_id
        self._remapping_range_start_index = remapping_range_start_index
        self._max_output_id = max_output_id
        self._device = device

    @abc.abstractmethod
    def preprocess(self, features: JaggedTensor) -> JaggedTensor:
        pass

    @abc.abstractmethod
    def profile(
        self,
        features: JaggedTensor,
    ) -> None:
        pass

    @abc.abstractmethod
    def evict(self) -> Optional[torch.Tensor]:
        """
        Returns None if no eviction should be done this iteration. Otherwise, return ids of slots to reset.
        On eviction, this module should reset its state for those slots, with the assumptionn that the downstream module
        will handle this properly.
        """
        pass

    @abc.abstractmethod
    def remap(self, features: JaggedTensor) -> JaggedTensor:
        pass

    def local_map_global_offset(self) -> int:
        """
        Returns the offset in the global id space of the current map.
        """
        return self._remapping_range_start_index

    @abc.abstractmethod
    def forward(
        self,
        features: JaggedTensor,
        mc_kwargs: Optional[Dict[str, Any]] = None,
    ) -> JaggedTensor:
        """
        Args:
        features (JaggedTensor]): feature representation
        mc_kwargs (Optional[Dict[str, Any]]): optional args dict to pass to MC module
        Returns:
            JaggedTensor: modified JT
        """
        pass

    @abc.abstractmethod
    def rebuild_with_max_output_id(
        self,
        max_output_id: int,
        remapping_range_start_index: int,
        device: torch.device,
    ) -> "ManagedCollisionModule":
        """
        Used for creating local MC modules for RW sharding, hack for now
        """
        pass


class TrivialManagedCollisionModule(ManagedCollisionModule):
    def __init__(
        self,
        max_output_id: int,
        device: torch.device,
        remapping_range_start_index: int = 0,
        max_input_id: int = 2**64,
    ) -> None:
        super().__init__(
            max_output_id, device, remapping_range_start_index, max_input_id
        )
        self.register_buffer(
            "count",
            torch.zeros(
                (max_output_id,),
                device=device,
            ),
        )

    @torch.no_grad()
    def preprocess(self, features: JaggedTensor) -> JaggedTensor:
        values = features.values() % self._max_output_id
        return JaggedTensor(
            values=values,
            lengths=features.lengths(),
            offsets=features.offsets(),
            weights=features.weights_or_none(),
        )

    @torch.no_grad()
    def profile(
        self,
        features: JaggedTensor,
    ) -> None:
        values = features.values()
        self.count[values] += 1

    @torch.no_grad()
    def remap(self, features: JaggedTensor) -> JaggedTensor:
        # no-op as self.preprocess maps input to correct range
        values = features.values()
        return JaggedTensor(
            values=values,
            lengths=features.lengths(),
            offsets=features.offsets(),
            weights=features.weights_or_none(),
        )

    @torch.no_grad()
    def forward(
        self,
        features: JaggedTensor,
        mc_kwargs: Optional[Dict[str, Any]] = None,
    ) -> JaggedTensor:
        self.profile(features)
        return self.remap(features)

    @torch.no_grad()
    def evict(self) -> Optional[torch.Tensor]:
        return None

    def rebuild_with_max_output_id(
        self,
        max_output_id: int,
        remapping_range_start_index: int,
        device: Optional[torch.device] = None,
    ) -> "TrivialManagedCollisionModule":
        return type(self)(
            max_output_id=max_output_id,
            remapping_range_start_index=remapping_range_start_index,
            device=device or self._device,
            max_input_id=self._max_input_id,
        )


class MCHEvictionPolicyMetadataInfo(NamedTuple):
    metadata_name: str
    is_mch_metadata: bool
    is_history_metadata: bool


class MCHEvictionPolicy(abc.ABC):
    @property
    @abc.abstractmethod
    def metadata_info(self) -> List[MCHEvictionPolicyMetadataInfo]:
        pass

    @abc.abstractmethod
    def record_history_metadata(
        self,
        current_iter: int,
        incoming_ids: torch.Tensor,
        history_metadata: Dict[str, torch.Tensor],
    ) -> None:
        """
        Args:
        current_iter (int): current iteration
        incoming_ids (torch.Tensor): incoming ids
        history_metadata (Dict[str, torch.Tensor]): history metadata dict

        Compute and record metadata based on incoming ids
            for the implemented eviction policy.
        """
        pass

    @abc.abstractmethod
    def coalesce_history_metadata(
        self,
        current_iter: int,
        history_metadata: Dict[str, torch.Tensor],
        unique_ids_counts: torch.Tensor,
        unique_inverse_mapping: torch.Tensor,
        additional_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
        history_metadata (Dict[str, torch.Tensor]): history metadata dict
        additional_ids (torch.Tensor): additional ids to be used as part of history
        unique_inverse_mapping (torch.Tensor): torch.unique inverse mapping generated from
            torch.cat[history_accumulator, additional_ids]. used to map history metadata tensor
            indices to their coalesced tensor indices.

        Coalesce metadata history buffers and return dict of processed metadata tensors.
        """
        pass

    @abc.abstractmethod
    def update_metadata_and_generate_eviction_scores(
        self,
        current_iter: int,
        mch_size: int,
        coalesced_history_argsort_mapping: torch.Tensor,
        coalesced_history_sorted_unique_ids_counts: torch.Tensor,
        coalesced_history_mch_matching_elements_mask: torch.Tensor,
        coalesced_history_mch_matching_indices: torch.Tensor,
        mch_metadata: Dict[str, torch.Tensor],
        coalesced_history_metadata: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:


        Returns Tuple of (evicted_indices, selected_new_indices) where:
            evicted_indices are indices in the mch map to be evicted, and
            selected_new_indices are the indices of the ids in the coalesced
            history that are to be added to the mch.
        """
        pass

    def _compute_selected_eviction_and_replacement_indices(
        self,
        pivot: int,
        eviction_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE these are like indices
        argsorted_eviction_scores = torch.argsort(
            eviction_scores, descending=True, stable=True
        )

        # indices with values >= zch_size in the top zch_size scores correspond
        #   to new incoming ids to be added to zch
        selected_new_ids_mask = argsorted_eviction_scores[:pivot] >= pivot
        # indices with values < zch_size outside the top zch_size scores correspond
        #   to existing zch ids to be evicted
        evicted_ids_mask = argsorted_eviction_scores[pivot:] < pivot
        evicted_indices = argsorted_eviction_scores[pivot:][evicted_ids_mask]
        selected_new_indices = (
            argsorted_eviction_scores[:pivot][selected_new_ids_mask] - pivot
        )

        return evicted_indices, selected_new_indices


class LFU_EvictionPolicy(MCHEvictionPolicy):
    def __init__(self) -> None:
        self._metadata_info = [
            MCHEvictionPolicyMetadataInfo(
                metadata_name="counts",
                is_mch_metadata=True,
                is_history_metadata=False,
            ),
        ]

    @property
    def metadata_info(self) -> List[MCHEvictionPolicyMetadataInfo]:
        return self._metadata_info

    def record_history_metadata(
        self,
        current_iter: int,
        incoming_ids: torch.Tensor,
        history_metadata: Dict[str, torch.Tensor],
    ) -> None:
        # no-op; no history buffers
        pass

    def coalesce_history_metadata(
        self,
        current_iter: int,
        history_metadata: Dict[str, torch.Tensor],
        unique_ids_counts: torch.Tensor,
        unique_inverse_mapping: torch.Tensor,
        additional_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # no-op; no history buffers
        return {}

    def update_metadata_and_generate_eviction_scores(
        self,
        current_iter: int,
        mch_size: int,
        coalesced_history_argsort_mapping: torch.Tensor,
        coalesced_history_sorted_unique_ids_counts: torch.Tensor,
        coalesced_history_mch_matching_elements_mask: torch.Tensor,
        coalesced_history_mch_matching_indices: torch.Tensor,
        mch_metadata: Dict[str, torch.Tensor],
        coalesced_history_metadata: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mch_counts = mch_metadata["counts"]
        # update metadata for matching ids
        mch_counts[
            coalesced_history_mch_matching_indices
        ] += coalesced_history_sorted_unique_ids_counts[
            coalesced_history_mch_matching_elements_mask
        ]

        # incoming non-matching ids
        new_sorted_uniq_ids_counts = coalesced_history_sorted_unique_ids_counts[
            ~coalesced_history_mch_matching_elements_mask
        ]

        merged_counts = torch.cat(
            [
                mch_counts,
                new_sorted_uniq_ids_counts,
            ]
        )
        # calculate evicted and replacement indices
        (
            evicted_indices,
            selected_new_indices,
        ) = self._compute_selected_eviction_and_replacement_indices(
            mch_size,
            merged_counts,
        )

        # update metadata for evicted ids
        mch_counts[evicted_indices] = new_sorted_uniq_ids_counts[selected_new_indices]

        return evicted_indices, selected_new_indices


class DistanceLFU_EvictionPolicy(MCHEvictionPolicy):
    def __init__(self, decay_exponent: int = 2) -> None:
        self._metadata_info = [
            MCHEvictionPolicyMetadataInfo(
                metadata_name="counts",
                is_mch_metadata=True,
                is_history_metadata=False,
            ),
            MCHEvictionPolicyMetadataInfo(
                metadata_name="last_access_iter",
                is_mch_metadata=True,
                is_history_metadata=True,
            ),
        ]
        self._decay_exponent = decay_exponent

    @property
    def metadata_info(self) -> List[MCHEvictionPolicyMetadataInfo]:
        return self._metadata_info

    def record_history_metadata(
        self,
        current_iter: int,
        incoming_ids: torch.Tensor,
        history_metadata: Dict[str, torch.Tensor],
    ) -> None:
        history_last_access_iter = history_metadata["last_access_iter"]
        history_last_access_iter[:] = current_iter

    def coalesce_history_metadata(
        self,
        current_iter: int,
        history_metadata: Dict[str, torch.Tensor],
        unique_ids_counts: torch.Tensor,
        unique_inverse_mapping: torch.Tensor,
        additional_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        coalesced_history_metadata: Dict[str, torch.Tensor] = {}
        history_last_access_iter = history_metadata["last_access_iter"]
        if additional_ids is not None:
            history_last_access_iter = torch.cat(
                [
                    history_last_access_iter,
                    torch.full_like(additional_ids, current_iter),
                ]
            )
        coalesced_history_metadata["last_access_iter"] = torch.zeros_like(
            unique_ids_counts
        ).scatter_reduce_(
            0,
            unique_inverse_mapping,
            history_last_access_iter,
            reduce="amax",
            include_self=False,
        )
        return coalesced_history_metadata

    def update_metadata_and_generate_eviction_scores(
        self,
        current_iter: int,
        mch_size: int,
        coalesced_history_argsort_mapping: torch.Tensor,
        coalesced_history_sorted_unique_ids_counts: torch.Tensor,
        coalesced_history_mch_matching_elements_mask: torch.Tensor,
        coalesced_history_mch_matching_indices: torch.Tensor,
        mch_metadata: Dict[str, torch.Tensor],
        coalesced_history_metadata: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mch_counts = mch_metadata["counts"]
        mch_last_access_iter = mch_metadata["last_access_iter"]

        # sort coalesced history metadata
        coalesced_history_metadata["last_access_iter"].copy_(
            coalesced_history_metadata["last_access_iter"][
                coalesced_history_argsort_mapping
            ]
        )
        coalesced_history_sorted_uniq_ids_last_access_iter = coalesced_history_metadata[
            "last_access_iter"
        ]

        # update metadata for matching ids
        mch_counts[
            coalesced_history_mch_matching_indices
        ] += coalesced_history_sorted_unique_ids_counts[
            coalesced_history_mch_matching_elements_mask
        ]
        mch_last_access_iter[
            coalesced_history_mch_matching_indices
        ] = coalesced_history_sorted_uniq_ids_last_access_iter[
            coalesced_history_mch_matching_elements_mask
        ]

        # incoming non-matching ids
        new_sorted_uniq_ids_counts = coalesced_history_sorted_unique_ids_counts[
            ~coalesced_history_mch_matching_elements_mask
        ]
        new_sorted_uniq_ids_last_access = (
            coalesced_history_sorted_uniq_ids_last_access_iter[
                ~coalesced_history_mch_matching_elements_mask
            ]
        )
        merged_counts = torch.cat(
            [
                mch_counts,
                new_sorted_uniq_ids_counts,
            ]
        )
        merged_access_iter = torch.cat(
            [
                mch_last_access_iter,
                new_sorted_uniq_ids_last_access,
            ]
        )
        merged_weighted_distance = torch.pow(
            current_iter - merged_access_iter + 1,
            self._decay_exponent,
        )
        # merged eviction scores are the eviction scores calculated for the
        #   tensor torch.cat[_mch_sorted_raw_ids, frequency_sorted_uniq_ids[~matching_eles]]
        # lower scores are evicted first.
        merged_eviction_scores = torch.div(merged_counts, merged_weighted_distance)

        # calculate evicted and replacement indices
        (
            evicted_indices,
            selected_new_indices,
        ) = self._compute_selected_eviction_and_replacement_indices(
            mch_size,
            merged_eviction_scores,
        )

        # update metadata for evicted ids
        mch_counts[evicted_indices] = new_sorted_uniq_ids_counts[selected_new_indices]
        mch_last_access_iter[evicted_indices] = new_sorted_uniq_ids_last_access[
            selected_new_indices
        ]

        return evicted_indices, selected_new_indices


class MCHManagedCollisionModule(ManagedCollisionModule):
    def __init__(
        self,
        # total output range size incl. zch (i.e. total_size >= zch_size)
        max_output_id: int,
        device: torch.device,
        # hash_func(input_ids, hash_size) -> hashed_input_ids
        hash_func: Callable[[torch.Tensor, int], torch.Tensor],
        is_train: bool,
        # max rolling tracking window size in number of _total_ IDs
        max_history_size: int,
        zch_size: int,
        eviction_policy: MCHEvictionPolicy,
        force_update_on_step: int = -1,
        #####
        remapping_range_start_index: int = 0,
        max_input_id: int = 2**62,
    ) -> None:
        super().__init__(
            max_output_id, device, remapping_range_start_index, max_input_id
        )

        self._is_train = is_train
        self._max_history_size = max_history_size
        assert (
            self._max_output_id >= zch_size
        ), "zch_size must be less then or equal to max_output_id"
        self._zch_size = zch_size
        assert self._zch_size > 0, "zch_size must be > 0"
        self._hash_size: int = self._max_output_id - self._zch_size
        assert self._hash_size > 0, (
            "hash_size (= max_output_id - zch_size) must be "
            ">= 1 for valid output mapping for non-ZCH IDs"
        )
        self._hash_func = hash_func
        self._remapping_range_start_index = remapping_range_start_index
        self._force_update_on_step: int = (
            force_update_on_step
            if force_update_on_step > 0
            else torch.iinfo(torch.int32).max
        )
        self._eviction_policy = eviction_policy

        self._current_iter: int = 0
        self._most_recent_update_iter: int = 0

        ## ------ mch info ------
        self.register_buffer(
            "_mch_sorted_raw_ids",
            torch.full(
                (self._zch_size + 1,),
                torch.iinfo(torch.int64).max,
                dtype=torch.int64,
                device=self._device,
            ),
        )
        self.register_buffer(
            "_mch_remapped_ids_mapping",
            torch.arange(self._zch_size, dtype=torch.int64, device=self._device),
        )

        ## ------ history info ------
        if self._is_train:
            self.register_buffer(
                "_history_accumulator",
                torch.empty(
                    self._max_history_size,
                    dtype=torch.int64,
                    device=self._device,
                ),
                # not checkpointed
                persistent=False,
            )
            self._mch_metadata: Dict[str, torch.Tensor] = {}
            self._history_metadata: Dict[str, torch.Tensor] = {}
            self._init_metadata_buffers()
            self._current_history_buffer_offset: int = 0
            self._evicted_emb_indices: torch.Tensor = torch.empty(
                (1,), device=self._device
            )
            self._evicted: bool = False

        # HACK for checkpointing
        # currently changing world_size between
        # saving/loading is not supported

        self._world_size: int = 1
        if dist.is_initialized():
            self._world_size = dist.get_world_size()

        def _post_state_dict_hook(
            module: MCHManagedCollisionModule,
            destination: Dict[str, torch.Tensor],
            prefix: str,
            _local_metadata: Dict[str, Any],
        ) -> None:
            # trim sorted_raw_ids anchor
            destination[prefix + "_mch_sorted_raw_ids"] = destination[
                prefix + "_mch_sorted_raw_ids"
            ][:-1]
            # update _mch_remapped_ids_mapping from local to global mapping
            destination[prefix + "_mch_remapped_ids_mapping"].add_(
                self.local_map_global_offset()
            )
            # self._device doesn't update if module.to(..) is called
            device = destination[prefix + "_mch_sorted_raw_ids"].device
            destination[prefix + "_current_iter_tensor"] = torch.full(
                (1,), self._current_iter, dtype=torch.int64, device=device
            )
            destination[prefix + "_most_recent_update_iter_tensor"] = torch.full(
                (1,),
                self._most_recent_update_iter,
                dtype=torch.int64,
                device=device,
            )
            destination[prefix + "_world_size_tensor"] = torch.full(
                (1,), self._world_size, dtype=torch.int64, device=device
            )

        def _load_state_dict_pre_hook(
            module: "MCHManagedCollisionModule",
            state_dict: Dict[str, torch.Tensor],
            prefix: str,
            *args: Any,
        ) -> None:
            # add sorted_raw_ids anchor
            state_dict[prefix + "_mch_sorted_raw_ids"] = torch.cat(
                [
                    state_dict[prefix + "_mch_sorted_raw_ids"],
                    torch.tensor(
                        [torch.iinfo(torch.int64).max],
                        dtype=torch.int64,
                        device=state_dict[prefix + "_mch_sorted_raw_ids"].device,
                    ),
                ]
            )
            # update _mch_remapped_ids_mapping from global to local mapping
            state_dict[prefix + "_mch_remapped_ids_mapping"].sub_(
                self.local_map_global_offset()
            )
            module._current_iter = cast(
                int, state_dict.pop(prefix + "_current_iter_tensor").item()
            )
            module._most_recent_update_iter = cast(
                int, state_dict.pop(prefix + "_most_recent_update_iter_tensor").item()
            )
            # HACK for checkpointing continued
            module._world_size = cast(
                int, state_dict.pop(prefix + "_world_size_tensor").item()
            )
            if dist.is_initialized():
                assert dist.get_world_size() == module._world_size
            else:
                assert module._world_size == 1

        self._register_state_dict_hook(_post_state_dict_hook)
        self._register_load_state_dict_pre_hook(
            _load_state_dict_pre_hook, with_module=True
        )

    def _init_metadata_buffers(self) -> None:
        eviction_metadata_info = self._eviction_policy.metadata_info
        for metadata in eviction_metadata_info:
            metadata_name, is_mch_metadata, is_history_metadata = metadata
            # mch_metadata
            if is_mch_metadata:
                buffer_name = "_mch_" + metadata_name
                self.register_buffer(
                    buffer_name,
                    torch.zeros(
                        (self._zch_size,),
                        dtype=torch.int64,
                        device=self._device,
                    ),
                )
                self._mch_metadata[metadata_name] = getattr(self, buffer_name)
            # history_metadata
            if is_history_metadata:
                buffer_name = "_history_" + metadata_name
                self.register_buffer(
                    buffer_name,
                    torch.zeros(
                        self._max_history_size,
                        dtype=torch.int64,
                        device=self._device,
                    ),
                    # not checkpointed
                    persistent=False,
                )
                self._history_metadata[metadata_name] = getattr(self, buffer_name)

    @torch.no_grad()
    def preprocess(self, features: JaggedTensor) -> JaggedTensor:
        values = self._hash_func(features.values(), self._max_input_id)
        return JaggedTensor(
            values=values,
            lengths=features.lengths(),
            offsets=features.offsets(),
            weights=features.weights_or_none(),
        )

    @torch.no_grad()
    def _match_indices(
        self, sorted_sequence: torch.Tensor, search_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        searched_indices = torch.searchsorted(sorted_sequence, search_values)
        retrieved_ids = sorted_sequence[searched_indices]
        matching_eles = retrieved_ids == search_values
        matched_indices = searched_indices[matching_eles]
        return (matching_eles, matched_indices)

    @torch.no_grad()
    def _sort_mch_buffers(self) -> None:
        mch_sorted_raw_ids = self._mch_sorted_raw_ids[:-1]
        argsorted_sorted_raw_ids = torch.argsort(mch_sorted_raw_ids, stable=True)
        mch_sorted_raw_ids.copy_(mch_sorted_raw_ids[argsorted_sorted_raw_ids])
        self._mch_remapped_ids_mapping.copy_(
            self._mch_remapped_ids_mapping[argsorted_sorted_raw_ids]
        )
        for mch_metadata_buffer in self._mch_metadata.values():
            mch_metadata_buffer.copy_(mch_metadata_buffer[argsorted_sorted_raw_ids])

    @torch.no_grad()
    def _update_and_evict(
        self,
        uniq_ids: torch.Tensor,
        uniq_ids_counts: torch.Tensor,
        uniq_ids_metadata: Dict[str, torch.Tensor],
    ) -> None:
        argsorted_uniq_ids_counts = torch.argsort(
            uniq_ids_counts, descending=True, stable=True
        )
        frequency_sorted_uniq_ids = uniq_ids[argsorted_uniq_ids_counts]
        frequency_sorted_uniq_ids_counts = uniq_ids_counts[argsorted_uniq_ids_counts]

        matching_eles, matched_indices = self._match_indices(
            self._mch_sorted_raw_ids, frequency_sorted_uniq_ids
        )

        new_frequency_sorted_uniq_ids = frequency_sorted_uniq_ids[~matching_eles]

        # evicted_indices are indices in the mch map to be evicted, and
        #   selected_new_indices are the indices of the ids in the coalesced
        #   history that are to be added to the mch.
        (
            evicted_indices,
            selected_new_indices,
        ) = self._eviction_policy.update_metadata_and_generate_eviction_scores(
            self._current_iter,
            self._zch_size,
            argsorted_uniq_ids_counts,
            frequency_sorted_uniq_ids_counts,
            matching_eles,
            matched_indices,
            self._mch_metadata,
            uniq_ids_metadata,
        )

        self._mch_sorted_raw_ids[evicted_indices] = new_frequency_sorted_uniq_ids[
            selected_new_indices
        ]

        # NOTE evicted ids for emb reset
        # if evicted flag is already set, then existing evicted ids havent been
        # consumed by evict(). append new evicted ids to the list
        if self._evicted:
            self._evicted_emb_indices = torch.unique(
                torch.cat(
                    [
                        self._evicted_emb_indices,
                        self._mch_remapped_ids_mapping[evicted_indices],
                    ]
                )
            )
        else:
            self._evicted_emb_indices = self._mch_remapped_ids_mapping[evicted_indices]
        self._evicted = True

        # re-sort for next search
        self._sort_mch_buffers()

    @torch.no_grad()
    def _coalesce_history(self, additional_ids: Optional[torch.Tensor] = None) -> None:
        current_history_accumulator = self._history_accumulator[
            : self._current_history_buffer_offset
        ]
        if additional_ids is not None:
            current_history_accumulator = torch.cat(
                [current_history_accumulator, additional_ids]
            )
        uniq_ids, uniq_inverse_mapping, uniq_ids_counts = torch.unique(
            current_history_accumulator,
            return_inverse=True,
            return_counts=True,
        )
        coalesced_eviction_history_metadata = (
            self._eviction_policy.coalesce_history_metadata(
                self._current_iter,
                {
                    metadata_name: metadata_buffer[
                        : self._current_history_buffer_offset
                    ]
                    for metadata_name, metadata_buffer in self._history_metadata.items()
                },
                uniq_ids_counts,
                uniq_inverse_mapping,
                additional_ids=additional_ids,
            )
        )
        self._update_and_evict(
            uniq_ids, uniq_ids_counts, coalesced_eviction_history_metadata
        )
        # reset buffer offset
        self._current_history_buffer_offset = 0
        # update most recent update step
        self._most_recent_update_iter = self._current_iter

    @torch.no_grad()
    def profile(
        self,
        features: JaggedTensor,
        force_update: bool = False,
        count_multiplier: Optional[int] = None,
    ) -> None:
        if self._is_train and self.training:
            multiplier = (
                count_multiplier
                if count_multiplier is not None and count_multiplier > 1
                else 1
            )
            values = features.values()
            if multiplier > 1:
                values = values.repeat(multiplier)
            num_incoming_values = values.size(0)
            free_elements = self._max_history_size - self._current_history_buffer_offset
            # check if need to coalesce by one of the following conditions:
            # 1. buffer will be full with incoming ids
            # 2. current iteration has reached max update step
            # 3. force update is set
            if (
                num_incoming_values >= free_elements
                or self._current_iter - self._most_recent_update_iter
                >= self._force_update_on_step
                or force_update
            ):
                self._coalesce_history(values)
            else:
                self._history_accumulator[
                    self._current_history_buffer_offset : self._current_history_buffer_offset
                    + num_incoming_values
                ] = values
                self._eviction_policy.record_history_metadata(
                    self._current_iter,
                    values,
                    {
                        metadata_name: metadata_buffer[
                            self._current_history_buffer_offset : self._current_history_buffer_offset
                            + num_incoming_values
                        ]
                        for metadata_name, metadata_buffer in self._history_metadata.items()
                    },
                )
                self._current_history_buffer_offset += num_incoming_values

            self._current_iter += 1

    @torch.no_grad()
    def remap(self, features: JaggedTensor) -> JaggedTensor:
        values = features.values()
        remapped_ids = torch.empty_like(values)

        # compute overlap between incoming IDs and remapping table
        searched_indices = torch.searchsorted(self._mch_sorted_raw_ids, values)
        retrieved_indices = self._mch_sorted_raw_ids[searched_indices]
        # identify matching inputs IDs
        matching_indices = retrieved_indices == values
        # update output with remapped matching IDs
        remapped_ids[matching_indices] = self._mch_remapped_ids_mapping[
            searched_indices[matching_indices]
        ]
        # select non-matching values
        non_matching_values = values[~matching_indices]
        # hash non-matching values
        hashed_non_matching = self._hash_func(non_matching_values, self._hash_size)
        # offset hash ids to their starting range
        remapped_ids[~matching_indices] = hashed_non_matching + self._zch_size

        return JaggedTensor(
            values=remapped_ids,
            lengths=features.lengths(),
            offsets=features.offsets(),
            weights=features.weights_or_none(),
        )

    @torch.no_grad()
    def forward(
        self,
        features: JaggedTensor,
        mc_kwargs: Optional[Dict[str, Any]] = None,
    ) -> JaggedTensor:
        """
        Args:
        features (JaggedTensor]): feature representation
        mc_kwargs (Optional[Dict[str, Any]]: optional args dict to pass to MC module
            MCHManagedCollisionModule supports:
                1. force_update (bool): force update this step
                2. count_multiplier (int): if provided, multiply
                    count of incoming ids by this value
        Returns:
            JaggedTensor: modified JT
        """
        force_update = (
            mc_kwargs.get("force_update", False) if mc_kwargs is not None else False
        )
        count_multiplier = (
            mc_kwargs.get("count_multiplier", None) if mc_kwargs is not None else None
        )
        self.profile(
            features,
            force_update=force_update,
            count_multiplier=count_multiplier,
        )
        return self.remap(features)

    @torch.no_grad()
    def evict(self) -> Optional[torch.Tensor]:
        if self._evicted:
            self._evicted = False
            return self._evicted_emb_indices
        else:
            return None

    def rebuild_with_max_output_id(
        self,
        max_output_id: int,
        remapping_range_start_index: int,
        device: Optional[torch.device] = None,
    ) -> "MCHManagedCollisionModule":
        return type(self)(
            max_output_id=max_output_id,
            remapping_range_start_index=remapping_range_start_index,
            zch_size=self._zch_size,
            device=device or self._device,
            max_input_id=self._max_input_id,
            is_train=self._is_train,
            max_history_size=self._max_history_size,
            hash_func=self._hash_func,
            force_update_on_step=self._force_update_on_step,
            eviction_policy=self._eviction_policy,
        )
