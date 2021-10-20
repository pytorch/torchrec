#!/usr/bin/env python3

from typing import Optional, List, Dict, Tuple

import torch
import torch.fx

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


def _cumsum(o: List[int]) -> List[int]:
    ret = [0]
    total = 0
    for i in o:
        total += i
        ret.append(total)
    return ret


def _to_offsets(lengths: torch.Tensor) -> torch.Tensor:
    return torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)


def _to_lengths(offsets: torch.Tensor) -> torch.Tensor:
    return offsets[1:] - offsets[:-1]


def _maybe_compute_lengths(
    lengths: Optional[torch.Tensor], offsets: Optional[torch.Tensor]
) -> torch.Tensor:
    if lengths is None:
        assert offsets is not None
        lengths = _to_lengths(offsets)
    return lengths


torch.fx.wrap("_maybe_compute_lengths")


def _maybe_compute_offsets(
    lengths: Optional[torch.Tensor], offsets: Optional[torch.Tensor]
) -> torch.Tensor:
    if offsets is None:
        assert lengths is not None
        offsets = _to_offsets(lengths)
    return offsets


torch.fx.wrap("_maybe_compute_offsets")


def _get_weights_or_throw(weights: Optional[torch.Tensor]) -> torch.Tensor:
    assert weights is not None, "This (Keyed)JaggedTensor doesn't have weights."
    return weights


torch.fx.wrap("_get_weights_or_throw")


def _assert_offsets_or_lengths_is_provided(
    offsets: Optional[torch.Tensor], lengths: Optional[torch.Tensor]
) -> None:
    assert offsets is not None or lengths is not None, "Must provide lengths or offsets"


torch.fx.wrap("_assert_offsets_or_lengths_is_provided")


def _regroup_keyed_tensors(
    keyed_tensors: List["KeyedTensor"], groups: List[List[str]]
) -> List[torch.Tensor]:
    embedding_dicts = [keyed_tensor.to_dict() for keyed_tensor in keyed_tensors]
    lengths = [keyed_tensor.length_per_key() for keyed_tensor in keyed_tensors]
    indices = [keyed_tensor._key_indices() for keyed_tensor in keyed_tensors]
    key_dim = keyed_tensors[0].key_dim()

    key_to_idx: dict[str, int] = {}
    for (i, keyed_tensor) in enumerate(keyed_tensors):
        for key in keyed_tensor.keys():
            key_to_idx[key] = i

    # Rearrange values based on groups with a single torch.cat operation.
    cat_input: List[torch.Tensor] = []
    for group in groups:
        for name in group:
            cat_input.append(embedding_dicts[key_to_idx[name]][name])
    rearranged_values = torch.cat(cat_input, key_dim)

    # Provide views over the rearranged values with a single torch.split operation.
    split_lengths: List[int] = []
    for group in groups:
        group_length = 0
        for name in group:
            group_length += lengths[key_to_idx[name]][indices[key_to_idx[name]][name]]
        split_lengths.append(group_length)

    return list(rearranged_values.split(split_lengths, dim=key_dim))


torch.fx.wrap("_regroup_keyed_tensors")


def _values_string(values: torch.Tensor, start: int, end: int) -> str:
    return "[" + ", ".join([str(value.item()) for value in values[start:end]]) + "]"


def _jagged_values_string(
    values: torch.Tensor,
    offsets: torch.Tensor,
    offset_start: int,
    offset_end: int,
) -> str:
    return (
        "["
        + ", ".join(
            [
                _values_string(values, offsets[index], offsets[index + 1])
                for index in range(offset_start, offset_end)
            ]
        )
        + "]"
    )


class JaggedTensor(metaclass=torch.fx.ProxyableClassMeta):
    """Represents an (optionally weighted) jagged tensor

    A `JaggedTensor` is a tensor with a *jagged dimension* which is dimension whose
    slices may be of different lengths. See KeyedJaggedTensor for full example.

    Implementation is torch.jit.script-able
    """

    def __init__(
        self,
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> None:
        self._values: torch.Tensor = values
        self._weights: Optional[torch.Tensor] = weights
        _assert_offsets_or_lengths_is_provided(offsets, lengths)
        if offsets is not None:
            _assert_tensor_has_no_elements_or_has_integers(offsets, "offsets")
        if lengths is not None:
            _assert_tensor_has_no_elements_or_has_integers(lengths, "lengths")
        self._lengths: Optional[torch.Tensor] = lengths
        self._offsets: Optional[torch.Tensor] = offsets

    @staticmethod
    def empty(is_weighted: bool = False) -> "JaggedTensor":
        weights = torch.tensor([]) if is_weighted else None
        return JaggedTensor(
            values=torch.tensor([]),
            offsets=torch.tensor([]),
            lengths=torch.tensor([]),
            weights=weights,
        )

    @staticmethod
    def from_dense_lengths(
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> "JaggedTensor":
        """
        Constructs `JaggedTensor` from dense values/weights of shape (B, N,).

        Note that `lengths` is still of shape (B,).
        """
        mask2d = torch.arange(values.size(1)).expand(
            values.size(0), -1
        ) < lengths.unsqueeze(-1)
        return JaggedTensor(
            values=values[mask2d],
            weights=weights[mask2d] if weights is not None else None,
            lengths=lengths,
        )

    def lengths(self) -> torch.Tensor:
        _lengths = _maybe_compute_lengths(self._lengths, self._offsets)
        self._lengths = _lengths
        return _lengths

    def offsets(self) -> torch.Tensor:
        _offsets = _maybe_compute_offsets(self._lengths, self._offsets)
        self._offsets = _offsets
        return _offsets

    def values(self) -> torch.Tensor:
        return self._values

    def weights(self) -> torch.Tensor:
        return _get_weights_or_throw(self._weights)

    def weights_or_none(self) -> Optional[torch.Tensor]:
        return self._weights

    def to(self, device: torch.device, non_blocking: bool = False) -> "JaggedTensor":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        return JaggedTensor(
            values=self._values.to(device, non_blocking=non_blocking),
            weights=weights.to(device, non_blocking=non_blocking)
            if weights is not None
            else None,
            lengths=lengths.to(device, non_blocking=non_blocking)
            if lengths is not None
            else None,
            offsets=offsets.to(device, non_blocking=non_blocking)
            if offsets is not None
            else None,
        )

    # pyre-ignore [56]
    @torch.jit.unused
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self._values.record_stream(stream)
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        if weights is not None:
            weights.record_stream(stream)
        if lengths is not None:
            lengths.record_stream(stream)
        if offsets is not None:
            offsets.record_stream(stream)

    def __str__(self) -> str:
        offsets = self.offsets()

        if self._weights is None:
            return (
                "JaggedTensor({\n    "
                + _jagged_values_string(self._values, offsets, 0, len(offsets) - 1)
                + "\n})\n"
            )

        return (
            "JaggedTensor({\n"
            + '    "values": '
            + _jagged_values_string(self._values, offsets, 0, len(offsets) - 1)
            + ',\n    "weights": '
            + _jagged_values_string(
                _get_weights_or_throw(self._weights), offsets, 0, len(offsets) - 1
            )
            + "\n})\n"
        )


def _assert_tensor_has_no_elements_or_has_integers(
    tensor: torch.Tensor, tensor_name: str
) -> None:
    assert tensor.numel() == 0 or tensor.dtype in [
        torch.long,
        torch.int,
        torch.short,
        torch.int8,
        torch.uint8,
    ], "{} must be of integer type, but got {}".format(tensor_name, tensor.dtype)


torch.fx.wrap("_assert_tensor_has_no_elements_or_has_integers")


def _maybe_compute_index_per_key(
    keys: List[str],
    index_per_key: Optional[Dict[str, int]],
) -> Dict[str, int]:
    if index_per_key is None:
        index_per_key = {key: i for i, key in enumerate(keys)}
    return index_per_key


torch.fx.wrap("_maybe_compute_index_per_key")


def _maybe_compute_lengths_kjt(
    device: torch.device,
    lengths: Optional[torch.Tensor],
    offsets: Optional[torch.Tensor],
    length_per_key: List[int],
) -> torch.Tensor:
    if lengths is None:
        if offsets is not None:
            lengths = _to_lengths(offsets)
        else:
            lengths = torch.tensor(
                length_per_key,
                device=device,
                dtype=torch.int,
            )
    return lengths


torch.fx.wrap("_maybe_compute_lengths_kjt")


def _maybe_compute_offsets_kjt(
    device: torch.device,
    lengths: Optional[torch.Tensor],
    offsets: Optional[torch.Tensor],
    length_per_key: List[int],
) -> torch.Tensor:
    if offsets is None:
        if lengths is not None:
            offsets = _to_offsets(lengths)
        else:
            offsets = torch.tensor(
                _cumsum(length_per_key),
                device=device,
                dtype=torch.int,
            )
    return offsets


torch.fx.wrap("_maybe_compute_offsets_kjt")


def _maybe_compute_stride_kjt(
    keys: List[str],
    stride: Optional[int],
    lengths: Optional[torch.Tensor],
    offsets: Optional[torch.Tensor],
) -> int:
    if stride is None:
        if len(keys) == 0:
            stride = 0
        elif offsets is not None:
            stride = (offsets.numel() - 1) // len(keys)
        elif lengths is not None:
            stride = lengths.numel() // len(keys)
        else:
            stride = 1
    return stride


torch.fx.wrap("_maybe_compute_stride_kjt")


def _maybe_compute_length_per_key(
    keys: List[str],
    stride: int,
    length_per_key: Optional[List[int]],
    lengths: Optional[torch.Tensor],
    offsets: Optional[torch.Tensor],
) -> List[int]:
    if length_per_key is None:
        if len(keys) and offsets is not None:
            _length: List[int] = (
                torch.sum((offsets[1:] - offsets[:-1]).view(-1, stride), dim=1)
                .cpu()
                .tolist()
            )
        elif len(keys) and lengths is not None:
            _length: List[int] = (
                torch.sum(lengths.view(-1, stride), dim=1).cpu().tolist()
            )
        else:
            _length: List[int] = []
        length_per_key = _length
    return length_per_key


torch.fx.wrap("_maybe_compute_length_per_key")


def _maybe_compute_offset_per_key(
    keys: List[str],
    stride: int,
    length_per_key: Optional[List[int]],
    offset_per_key: Optional[List[int]],
    lengths: Optional[torch.Tensor],
    offsets: Optional[torch.Tensor],
) -> Tuple[List[int], List[int]]:
    if length_per_key is None:
        _length: List[int] = _maybe_compute_length_per_key(
            keys, stride, length_per_key, lengths, offsets
        )
        length_per_key = _length
        offset_per_key = _cumsum(length_per_key)
    elif offset_per_key is None:
        offset_per_key = _cumsum(length_per_key)
    return length_per_key, offset_per_key


torch.fx.wrap("_maybe_compute_offset_per_key")


def _jagged_tensor_string(
    key: str,
    values: torch.Tensor,
    weights: Optional[torch.Tensor],
    offsets: torch.Tensor,
    offset_start: int,
    offset_end: int,
) -> str:
    if weights is None:
        return '"{}": '.format(key) + _jagged_values_string(
            values, offsets, offset_start, offset_end
        )

    return (
        '"{}"'.format(key)
        + ': {\n        "values": '
        + _jagged_values_string(values, offsets, offset_start, offset_end)
        + ',\n        "weights": '
        + _jagged_values_string(
            _get_weights_or_throw(weights), offsets, offset_start, offset_end
        )
        + "\n    }"
    )


def _trim_keys(keys: List[str]) -> List[str]:
    trimed_keys: List[str] = []
    trimed_key_dict: Dict[str, bool] = {}
    for key in keys:
        trimed_key = key.split("@")[0]
        assert (
            trimed_key not in trimed_key_dict
        ), f"{trimed_key} vs {trimed_key_dict} and all keys {keys}"
        trimed_keys.append(trimed_key)
        trimed_key_dict[trimed_key] = True
    return trimed_keys


torch.fx.wrap("_trim_keys")


def _kjt_to_jt_dict(
    stride: int,
    keys: List[str],
    length_per_key: List[int],
    values: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    weights: Optional[torch.Tensor],
) -> Dict[str, JaggedTensor]:
    jt_dict: Dict[str, JaggedTensor] = {}
    values_list = torch.split(values, length_per_key)
    lengths_tuple = torch.unbind(lengths.view(-1, stride), dim=0)
    if weights is not None:
        weights_list = torch.split(weights, length_per_key)
        for idx, key in enumerate(keys):
            length = lengths_tuple[idx]
            offset = _to_offsets(length)
            jt_dict[key] = JaggedTensor(
                lengths=length,
                offsets=offset,
                values=values_list[idx],
                weights=weights_list[idx],
            )
    else:
        for idx, key in enumerate(keys):
            length = lengths_tuple[idx]
            offset = _to_offsets(length)
            jt_dict[key] = JaggedTensor(
                lengths=length,
                offsets=offset,
                values=values_list[idx],
            )
    return jt_dict


torch.fx.wrap("_kjt_to_jt_dict")


class KeyedJaggedTensor(metaclass=torch.fx.ProxyableClassMeta):
    """Represents an (optionally weighted) keyed jagged tensor

    A `JaggedTensor` is a tensor with a *jagged dimension* which is dimension whose
    slices may be of different lengths.  Keyed on first dimesion, and jagged on last dimension

    For example:
                 0       1        2  <-- dim_1
    "Feature0"   [V0,V1] None    [V2]
    "Feature1"   [V3]    [V4]    [V5,V6,V7]
      ^
     dim_0

    dim_0: keyed dimension (ie. `Feature0`, `Feature1`)
    dim_1: optional second dimension (ie. batch size)
    dim_2: The jagged dimension which has slice lengths between 0-3 in the above example

    We represent this data with following inputs:

    values: torch.Tensor = [V0, V1, V2, V3, V4, V5, V6, V7], V == any tensor datatype
    weights: torch.Tensor = [W0, W1, W2, W3, W4, W5, W6, W7], W == any tensor datatype
    lengths: torch.Tensor = [2, 0, 1, 1, 1, 3], representing the jagged slice
    offsets: torch.Tensor = [0, 2, 2, 3, 4, 5, 8], offsets from 0 for each jagged slice
    keys: List[int] = ["Feature0", "Feature1"], which corresponds to each value of dim_0
    index_per_key: Dict[str, int] = {"Feature0": 0, "Feature1: 1}, index for key in keys
    offset_per_key: List[int] = [0, 3, 8], start offset for each key and final offset



    Implementation is torch.jit.script-able
    """

    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        # Below exposed to ensure torch.script-able
        length_per_key: Optional[List[int]] = None,
        offset_per_key: Optional[List[int]] = None,
        index_per_key: Optional[Dict[str, int]] = None,
    ) -> None:
        self._keys: List[str] = keys
        self._values: torch.Tensor = values
        self._weights: Optional[torch.Tensor] = weights
        if offsets is not None:
            _assert_tensor_has_no_elements_or_has_integers(offsets, "offsets")
        if lengths is not None:
            _assert_tensor_has_no_elements_or_has_integers(lengths, "lengths")
        self._lengths: Optional[torch.Tensor] = lengths
        self._offsets: Optional[torch.Tensor] = offsets
        stride = _maybe_compute_stride_kjt(keys, stride, lengths, offsets)
        self._stride: int = stride

        # lazy fields
        self._length_per_key: Optional[List[int]] = length_per_key
        self._offset_per_key: Optional[List[int]] = offset_per_key
        self._index_per_key: Optional[Dict[str, int]] = index_per_key

    @staticmethod
    def from_offsets_sync(
        keys: List[str],
        values: torch.Tensor,
        offsets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
    ) -> "KeyedJaggedTensor":
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            weights=weights,
            offsets=offsets,
            stride=stride,
        )
        return kjt.sync()

    @staticmethod
    def from_lengths_sync(
        keys: List[str],
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
    ) -> "KeyedJaggedTensor":
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            weights=weights,
            lengths=lengths,
            stride=stride,
        )
        return kjt.sync()

    @staticmethod
    def empty(
        is_weighted: bool = False, device: Optional[torch.device] = None
    ) -> "KeyedJaggedTensor":
        weights = None
        if is_weighted is True:
            weights = torch.tensor([], device=device) if device else torch.tensor([])

        return KeyedJaggedTensor(
            keys=[],
            values=torch.tensor([], device=device) if device else torch.tensor([]),
            weights=weights,
            stride=0,
        )

    @staticmethod
    def empty_like(kjt: "KeyedJaggedTensor") -> "KeyedJaggedTensor":
        return KeyedJaggedTensor(
            keys=[],
            values=torch.tensor([], device=kjt.device(), dtype=kjt.values().dtype),
            weights=None
            if kjt.weights_or_none() is None
            else torch.tensor([], device=kjt.device(), dtype=kjt.weights().dtype),
            lengths=None,
            offsets=None,
            stride=kjt.stride(),
        )

    def sync(self) -> "KeyedJaggedTensor":
        self.length_per_key()
        self.offset_per_key()
        return self

    def device(self) -> torch.device:
        return self._values.device

    def lengths(self) -> torch.Tensor:
        _lengths = _maybe_compute_lengths(self._lengths, self._offsets)
        self._lengths = _lengths
        return _lengths

    def offsets(self) -> torch.Tensor:
        _offsets = _maybe_compute_offsets(self._lengths, self._offsets)
        self._offsets = _offsets
        return _offsets

    def keys(self) -> List[str]:
        return self._keys

    def values(self) -> torch.Tensor:
        return self._values

    def weights(self) -> torch.Tensor:
        return _get_weights_or_throw(self._weights)

    def weights_or_none(self) -> Optional[torch.Tensor]:
        return self._weights

    def stride(self) -> int:
        return self._stride

    def _key_indices(self) -> Dict[str, int]:
        _index_per_key = _maybe_compute_index_per_key(
            self._keys,
            self._index_per_key,
        )
        self._index_per_key: Dict[str, int] = _index_per_key
        return _index_per_key

    def length_per_key(self) -> List[int]:
        _length_per_key = _maybe_compute_length_per_key(
            self._keys,
            self.stride(),
            self._length_per_key,
            self._lengths,
            self._offsets,
        )
        self._length_per_key = _length_per_key
        return _length_per_key

    def offset_per_key(self) -> List[int]:
        _length_per_key, _offset_per_key = _maybe_compute_offset_per_key(
            self._keys,
            self.stride(),
            self._length_per_key,
            self._offset_per_key,
            self._lengths,
            self._offsets,
        )
        self._length_per_key = _length_per_key
        self._offset_per_key = _offset_per_key
        return _offset_per_key

    def split(
        self, segments: List[int], trim: bool = True
    ) -> List["KeyedJaggedTensor"]:
        split_list: List[KeyedJaggedTensor] = []
        start = 0
        start_offset = 0
        _length_per_key = self.length_per_key()
        _offset_per_key = self.offset_per_key()
        for segment in segments:
            end = start + segment
            end_offset = _offset_per_key[end]
            keys: List[str] = self._keys[start:end]
            if segment == len(self._keys):
                # no torch slicing required
                split_list.append(
                    KeyedJaggedTensor(
                        keys=_trim_keys(self._keys) if trim else self._keys,
                        values=self._values,
                        weights=self.weights_or_none(),
                        lengths=self._lengths,
                        offsets=self._offsets,
                        stride=self._stride,
                        length_per_key=self._length_per_key,
                        offset_per_key=self._offset_per_key,
                        index_per_key=self._index_per_key,
                    )
                )
            elif segment == 0:
                split_list.append(
                    KeyedJaggedTensor(
                        keys=keys,
                        values=torch.tensor(
                            [], device=self.device(), dtype=self._values.dtype
                        ),
                        weights=None
                        if self.weights_or_none() is None
                        else torch.tensor(
                            [],
                            device=self.device(),
                            dtype=self.weights().dtype,
                        ),
                        lengths=torch.tensor([], device=self.device(), dtype=torch.int),
                        offsets=torch.tensor([], device=self.device(), dtype=torch.int),
                        stride=self._stride,
                        length_per_key=None,
                        offset_per_key=None,
                        index_per_key=None,
                    )
                )
            else:
                split_length_per_key = _length_per_key[start:end]
                split_list.append(
                    KeyedJaggedTensor(
                        keys=_trim_keys(keys) if trim else keys,
                        values=self._values[start_offset:end_offset],
                        weights=None
                        if self.weights_or_none() is None
                        else self.weights()[start_offset:end_offset],
                        lengths=self.lengths()[
                            start * self._stride : end * self._stride
                        ],
                        offsets=None,
                        stride=self._stride,
                        length_per_key=split_length_per_key,
                        offset_per_key=None,
                        index_per_key=None,
                    )
                )
            start = end
            start_offset = end_offset
        return split_list

    def permute(
        self, indices: List[int], indices_tensor: Optional[torch.Tensor] = None
    ) -> "KeyedJaggedTensor":

        if indices_tensor is None:
            indices_tensor = torch.tensor(
                indices, dtype=torch.int, device=self.device()
            )

        length_per_key = self.length_per_key()
        permuted_keys: List[str] = []
        permuted_length_per_key: List[int] = []
        permuted_lengths_sum = 0
        seen: Dict[str, int] = {}
        for index in indices:
            key = self._keys[index]
            count = seen.get(key, 0)
            permuted_keys.append(f"{key}@copy_{count}" if count else key)
            permuted_lengths_sum += length_per_key[index]
            permuted_length_per_key.append(length_per_key[index])
            seen[key] = count + 1

        (
            permuted_lengths,
            permuted_values,
            permuted_weights,
        ) = torch.ops.fbgemm.permute_sparse_data(
            indices_tensor,
            self.lengths().view(len(self._keys), -1),
            self.values(),
            self.weights_or_none(),
            permuted_lengths_sum,
        )

        kjt = KeyedJaggedTensor(
            keys=permuted_keys,
            values=permuted_values,
            weights=permuted_weights,
            lengths=permuted_lengths.view(-1),
            offsets=None,
            stride=self._stride,
            length_per_key=permuted_length_per_key if len(permuted_keys) > 0 else None,
            offset_per_key=None,
            index_per_key=None,
        )
        return kjt

    def bucketize(
        self,
        num_buckets: int,
        block_sizes: torch.Tensor,
        output_permute: bool = False,
        bucketize_pos: bool = False,
    ) -> Tuple["KeyedJaggedTensor", Optional[torch.Tensor]]:
        """
        Bucketize the `values` in KeyedJaggedTensor into `num_buckets` buckets, `lengths` are readjusted based on the bucketization results.

        Args:
            num_buckets (int): The number of buckets to bucketize the values into.
            block_sizes: (torch.Tensor): The bucket sizes for the keyed dimension.
            output_permute (bool): Output the memory location mapping from the unbucketized values to bucketized values or not.
            bucketize_pos (bool):  Output the changed position of the bucketized values or not.
        Returns:
            The bucketized `KeyedJaggedTensor` and the optional permute mapping from the unbucketized values to bucketized values.
        """
        num_features = len(self.keys())
        assert (
            block_sizes.numel() == num_features
        ), f"Expecting block sizes for {num_features} features, but {block_sizes.numel()} received."

        # kernel expects them to be same type, cast to avoid type mismatch
        block_sizes_new_type = block_sizes.type(self.values().type())
        (
            bucketized_lengths,
            bucketized_indices,
            bucketized_weights,
            pos,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            self.lengths().view(-1),
            self.values(),
            bucketize_pos=bucketize_pos,
            sequence=output_permute,
            block_sizes=block_sizes_new_type,
            my_size=num_buckets,
            weights=self.weights_or_none(),
        )
        bucketized_keys: List[str] = []
        for bucket in range(num_buckets):
            for key in self.keys():
                bucketized_keys.append(f"{key}@bucket_{bucket}")

        return (
            KeyedJaggedTensor(
                keys=bucketized_keys,
                values=bucketized_indices,
                weights=pos if bucketize_pos else bucketized_weights,
                lengths=bucketized_lengths.view(-1),
                offsets=None,
                stride=self.stride(),
                length_per_key=None,
                offset_per_key=None,
                index_per_key=None,
            ),
            unbucketize_permute,
        )

    def __getitem__(self, key: str) -> JaggedTensor:
        offset_per_key = self.offset_per_key()
        index = self._key_indices()[key]
        start_offset = offset_per_key[index]
        end_offset = offset_per_key[index + 1]
        return JaggedTensor(
            values=self._values[start_offset:end_offset],
            weights=None
            if self.weights_or_none() is None
            else self.weights()[start_offset:end_offset],
            lengths=self.lengths()[index * self._stride : (index + 1) * self._stride],
            offsets=None,
        )

    def to_dict(self) -> Dict[str, JaggedTensor]:
        return _kjt_to_jt_dict(
            self.stride(),
            self.keys(),
            self.length_per_key(),
            self.values(),
            self.lengths(),
            self.offsets(),
            self.weights_or_none(),
        )

    # pyre-ignore [56]
    @torch.jit.unused
    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self._values.record_stream(stream)
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        if weights is not None:
            weights.record_stream(stream)
        if lengths is not None:
            lengths.record_stream(stream)
        if offsets is not None:
            offsets.record_stream(stream)

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "KeyedJaggedTensor":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets
        length_per_key = self._length_per_key
        offset_per_key = self._offset_per_key
        index_per_key = self._index_per_key

        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values.to(device, non_blocking=non_blocking),
            weights=weights.to(device, non_blocking=non_blocking)
            if weights is not None
            else None,
            lengths=lengths.to(device, non_blocking=non_blocking)
            if lengths is not None
            else None,
            offsets=offsets.to(device, non_blocking=non_blocking)
            if offsets is not None
            else None,
            stride=self._stride,
            length_per_key=length_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key,
        )

    def __str__(self) -> str:
        if self._offsets is None and self._lengths is None:
            return "KeyedJaggedTensor()\n"
        offsets = self.offsets()

        step = (len(offsets) - 1) // len(self._keys)
        return (
            "KeyedJaggedTensor({\n"
            + ",\n".join(
                [
                    "    "
                    + _jagged_tensor_string(
                        self._keys[index],
                        self._values,
                        self._weights,
                        offsets,
                        index * step,
                        (index + 1) * step,
                    )
                    for index in range(len(self._keys))
                ]
            )
            + "\n})\n"
        )

    def pin_memory(self) -> "KeyedJaggedTensor":
        weights = self._weights
        lengths = self._lengths
        offsets = self._offsets

        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values.pin_memory(),
            weights=weights.pin_memory() if weights is not None else None,
            lengths=lengths.pin_memory() if lengths is not None else None,
            offsets=offsets.pin_memory() if offsets is not None else None,
            stride=self._stride,
            length_per_key=self._length_per_key,
            offset_per_key=self._offset_per_key,
            index_per_key=self._index_per_key,
        )


def _maybe_compute_offset_per_key_kt(
    length_per_key: List[int],
    offset_per_key: Optional[List[int]],
) -> List[int]:
    if offset_per_key is None:
        offset_per_key = _cumsum(length_per_key)
    return offset_per_key


torch.fx.wrap("_maybe_compute_offset_per_key_kt")


def _keyed_values_string(values: torch.Tensor) -> str:
    return (
        "["
        + ", ".join([_values_string(value, 0, len(value)) for value in values])
        + "]"
    )


class KeyedTensor(metaclass=torch.fx.ProxyableClassMeta):
    """
    KeyedTensor holds a concatenated list of dense tensors
    each of which can be accessed by a key.
    Keyed dimension can be variable length (length_per_key).
    Common use cases uses include storage of pooled embeddings of different dimensions.

    Constructor Args:
        keys (List[str]): list of keys
        length_per_key (List[int]): length of each key along key dimension
        values (torch.Tensor): dense tensor, concatenated typically along key dimension
        key_dim (int): key dimension, zero indexed - defaults to 1 (typically B is 0-dimension)

    Implementation is torch.jit.script-able


    Example:
        kt is KeyedTensor holding

                    0           1           2
            "Embedding A"    [1,1]       [1,1]        [1,1]
            "Embedding B"    [2,1,2]     [2,1,2]      [2,1,2]
            "Embedding C"    [3,1,2,3]   [3,1,2,3]    [3,1,2,3]
        >>> tensor_list = [
                torch.tensor([[1,1]] * 3),
                torch.tensor([[2,1,2]] * 3),
                torch.tensor([[3,1,2,3]] * 3),
            ]
        >>> keys = ["Embedding A", "Embedding B", "Embedding C"]
        >>> kt = KeyedTensor.from_tensor_list(keys, tensor_list)
        >>> kt.values()
            tensor([[1, 1, 2, 1, 2, 3, 1, 2, 3],
            [1, 1, 2, 1, 2, 3, 1, 2, 3],
            [1, 1, 2, 1, 2, 3, 1, 2, 3]])
        >>> kt["Embedding B"]
            tensor([[2, 1, 2],
            [2, 1, 2],
            [2, 1, 2]])
    """

    def __init__(
        self,
        keys: List[str],
        length_per_key: List[int],
        values: torch.Tensor,
        key_dim: int = 1,
        # Below exposed to ensure torch.script-able
        offset_per_key: Optional[List[int]] = None,
        index_per_key: Optional[Dict[str, int]] = None,
    ) -> None:
        self._keys = keys
        self._length_per_key = length_per_key
        self._values = values
        self._key_dim = key_dim

        self._offset_per_key: Optional[List[int]] = offset_per_key
        self._index_per_key: Optional[Dict[str, int]] = index_per_key

    @staticmethod
    def from_tensor_list(
        keys: List[str], tensors: List[torch.Tensor], key_dim: int = 1, cat_dim: int = 1
    ) -> "KeyedTensor":
        length_per_key = [tensor.shape[key_dim] for tensor in tensors]
        return KeyedTensor(
            keys=keys,
            length_per_key=length_per_key,
            values=torch.cat(tensors, dim=cat_dim),
            key_dim=key_dim,
        )

    def keys(self) -> List[str]:
        return self._keys

    def values(self) -> torch.Tensor:
        return self._values

    def key_dim(self) -> int:
        return self._key_dim

    def offset_per_key(self) -> List[int]:
        _offset_per_key = _maybe_compute_offset_per_key_kt(
            self._length_per_key,
            self._offset_per_key,
        )
        self._offset_per_key = _offset_per_key
        return _offset_per_key

    def length_per_key(self) -> List[int]:
        return self._length_per_key

    def _key_indices(self) -> Dict[str, int]:
        _index_per_key = _maybe_compute_index_per_key(
            self._keys,
            self._index_per_key,
        )
        self._index_per_key = _index_per_key
        return _index_per_key

    def __getitem__(self, key: str) -> torch.Tensor:
        index = self._key_indices()[key]
        start = self.offset_per_key()[index]
        length = self._length_per_key[index]
        # pyre-ignore [16]: Undefined attribute `torch.Tensor` has no attribute `narrow`
        return self._values.narrow(dim=self._key_dim, start=start, length=length)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        indices = self._key_indices()
        lengths = self._length_per_key
        split_values = self._values.split(lengths, dim=self._key_dim)
        return {key: split_values[index] for (key, index) in indices.items()}

    @staticmethod
    def regroup(
        keyed_tensors: List["KeyedTensor"], groups: List[List[str]]
    ) -> List[torch.Tensor]:
        return _regroup_keyed_tensors(keyed_tensors, groups)

    def __str__(self) -> str:
        if len(self._keys) == 0:
            return "KeyedTensor()\n"

        return (
            "KeyedTensor({\n"
            + ",\n".join(
                [
                    '    "{}": '.format(key) + _keyed_values_string(self[key])
                    for key in self._keys
                ]
            )
            + "\n})\n"
        )
