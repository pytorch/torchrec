#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import (
    Iterator,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    Tuple,
)

import numpy as np
import torch
import torch.utils.data.datapipes as dp
from iopath.common.file_io import PathManagerFactory, PathManager
from pyre_extensions import none_throws
from torch.utils.data import IterDataPipe
from torchrec.datasets.utils import (
    LoadFiles,
    ReadLinesFromCSV,
    safe_cast,
    PATH_MANAGER_KEY,
    Batch,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


INT_FEATURE_COUNT = 13
CAT_FEATURE_COUNT = 26
DEFAULT_LABEL_NAME = "label"
DEFAULT_INT_NAMES: List[str] = [f"int_{idx}" for idx in range(INT_FEATURE_COUNT)]
DEFAULT_CAT_NAMES: List[str] = [f"cat_{idx}" for idx in range(CAT_FEATURE_COUNT)]
DEFAULT_COLUMN_NAMES: List[str] = [
    DEFAULT_LABEL_NAME,
    *DEFAULT_INT_NAMES,
    *DEFAULT_CAT_NAMES,
]

COLUMN_TYPE_CASTERS: List[Callable[[Union[int, str]], Union[int, str]]] = [
    lambda val: safe_cast(val, int, 0),
    *(lambda val: safe_cast(val, int, 0) for _ in range(INT_FEATURE_COUNT)),
    *(lambda val: safe_cast(val, str, "") for _ in range(CAT_FEATURE_COUNT)),
]


def _default_row_mapper(example: List[str]) -> Dict[str, Union[int, str]]:
    column_names = reversed(DEFAULT_COLUMN_NAMES)
    column_type_casters = reversed(COLUMN_TYPE_CASTERS)
    return {
        next(column_names): next(column_type_casters)(val) for val in reversed(example)
    }


class CriteoIterDataPipe(IterDataPipe):
    """
    IterDataPipe that can be used to stream either the Criteo 1TB Click Logs Dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) or the
    Kaggle/Criteo Display Advertising Dataset
    (https://www.kaggle.com/c/criteo-display-ad-challenge/) from the source TSV
    files.

    Args:
        paths (Iterable[str]): local paths to TSV files that constitute the Criteo
            dataset.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each
            split TSV line.
        open_kw: options to pass to underlying invocation of
            iopath.common.file_io.PathManager.open.

    Example:
        >>> datapipe = CriteoIterDataPipe(
        >>>     ("/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv")
        >>> )
        >>> datapipe = dp.iter.Batcher(datapipe, 100)
        >>> datapipe = dp.iter.Collator(datapipe)
        >>> batch = next(iter(datapipe))
    """

    def __init__(
        self,
        paths: Iterable[str],
        *,
        # pyre-ignore[2]
        row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
        # pyre-ignore[2]
        **open_kw,
    ) -> None:
        self.paths = paths
        self.row_mapper = row_mapper
        self.open_kw: Any = open_kw  # pyre-ignore[4]

    # pyre-ignore[3]
    def __iter__(self) -> Iterator[Any]:
        worker_info = torch.utils.data.get_worker_info()
        paths = self.paths
        if worker_info is not None:
            paths = (
                path
                for (idx, path) in enumerate(paths)
                if idx % worker_info.num_workers == worker_info.id
            )
        datapipe = LoadFiles(paths, mode="r", **self.open_kw)
        datapipe = ReadLinesFromCSV(datapipe, delimiter="\t")
        if self.row_mapper:
            datapipe = dp.iter.Mapper(datapipe, self.row_mapper)
        yield from datapipe


def criteo_terabyte(
    paths: Iterable[str],
    *,
    # pyre-ignore[2]
    row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    # pyre-ignore[2]
    **open_kw,
) -> IterDataPipe:
    """`Criteo 1TB Click Logs <https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/>`_ Dataset
    Args:
        paths (Iterable[str]): local paths to TSV files that constitute the Criteo 1TB
            dataset.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each
            split TSV line.
        open_kw: options to pass to underlying invocation of
            iopath.common.file_io.PathManager.open.

    Example:
        >>> datapipe = criteo_terabyte(
        >>>     ("/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv")
        >>> )
        >>> datapipe = dp.iter.Batcher(datapipe, 100)
        >>> datapipe = dp.iter.Collator(datapipe)
        >>> batch = next(iter(datapipe))
    """
    return CriteoIterDataPipe(paths, row_mapper=row_mapper, **open_kw)


def criteo_kaggle(
    path: str,
    *,
    # pyre-ignore[2]
    row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    # pyre-ignore[2]
    **open_kw,
) -> IterDataPipe:
    """`Kaggle/Criteo Display Advertising <https://www.kaggle.com/c/criteo-display-ad-challenge/>`_ Dataset
    Args:
        root (str): local path to train or test dataset file.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each split TSV line.
        open_kw: options to pass to underlying invocation of iopath.common.file_io.PathManager.open.

    Example:
        >>> train_datapipe = criteo_kaggle(
        >>>     "/home/datasets/criteo_kaggle/train.txt",
        >>> )
        >>> example = next(iter(train_datapipe))
        >>> test_datapipe = criteo_kaggle(
        >>>     "/home/datasets/criteo_kaggle/test.txt",
        >>> )
        >>> example = next(iter(test_datapipe))
    """
    return CriteoIterDataPipe((path,), row_mapper=row_mapper, **open_kw)


class BinaryCriteoUtils:
    """
    Utility functions used to preprocess, save, load, partition, etc. the Criteo
    dataset in a binary (numpy) format.
    """

    @staticmethod
    def tsv_to_npys(
        in_file: str,
        out_dense_file: str,
        out_sparse_file: str,
        out_labels_file: str,
        path_manager_key: str = PATH_MANAGER_KEY,
    ) -> None:
        """
        Convert one Criteo tsv file to three npy files: one for dense (np.float32), one
        for sparse (np.int32), and one for labels (np.int32).

        Args:
            in_file (str): Input tsv file path.
            out_dense_file (str): Output dense npy file path.
            out_sparse_file (str): Output sparse npy file path.
            out_labels_file (str): Output labels npy file path.
            path_manager_key (str): Path manager key used to load from different
                filesystems.

        Returns:
            None.
        """

        def row_mapper(row: List[str]) -> Tuple[List[int], List[int], int]:
            label = safe_cast(row[0], int, 0)
            dense = [safe_cast(row[i], int, 0) for i in range(1, 1 + INT_FEATURE_COUNT)]
            sparse = [
                int(safe_cast(row[i], str, "0") or "0", 16)
                for i in range(
                    1 + INT_FEATURE_COUNT, 1 + INT_FEATURE_COUNT + CAT_FEATURE_COUNT
                )
            ]
            return dense, sparse, label  # pyre-ignore[7]

        dense, sparse, labels = [], [], []
        for (row_dense, row_sparse, row_label) in CriteoIterDataPipe(
            [in_file], row_mapper=row_mapper
        ):
            dense.append(row_dense)
            sparse.append(row_sparse)
            labels.append(row_label)

        # PyTorch tensors can't handle uint32, but we can save space by not
        # using int64. Numpy will automatically handle dense values >= 2 ** 31.
        dense_np = np.array(dense, dtype=np.int32)
        del dense
        sparse_np = np.array(sparse, dtype=np.int32)
        del sparse
        labels_np = np.array(labels, dtype=np.int32)
        del labels

        # Log is expensive to compute at runtime.
        dense_np += 3
        dense_np = np.log(dense_np, dtype=np.float32)

        # To be consistent with dense and sparse.
        labels_np = labels_np.reshape((-1, 1))

        path_manager = PathManagerFactory().get(path_manager_key)
        for (fname, arr) in [
            (out_dense_file, dense_np),
            (out_sparse_file, sparse_np),
            (out_labels_file, labels_np),
        ]:
            with path_manager.open(fname, "wb") as fout:
                np.save(fout, arr)

    @staticmethod
    def get_shape_from_npy(
        path: str, path_manager_key: str = PATH_MANAGER_KEY
    ) -> Tuple[int, ...]:
        """
        Returns the shape of an npy file using only its header.

        Args:
            path (str): Input npy file path.
            path_manager_key (str): Path manager key used to load from different
                filesystems.

        Returns:
            shape (Tuple[int, ...]): Shape tuple.
        """
        path_manager = PathManagerFactory().get(path_manager_key)
        with path_manager.open(path, "rb") as fin:
            np.lib.format.read_magic(fin)
            shape, _order, _dtype = np.lib.format.read_array_header_1_0(fin)
            return shape

    @staticmethod
    def get_file_idx_to_row_range(
        lengths: List[int],
        rank: int,
        world_size: int,
    ) -> Dict[int, Tuple[int, int]]:
        """
        Given a rank, world_size, and the lengths (number of rows) for a list of files,
        return which files and which portions of those files (represented as row ranges
        - all range indices are inclusive) should be handled by the rank. Each rank
        will be assigned the same number of rows.

        The ranges are determined in such a way that each rank deals with large
        continuous ranges of files. This enables each rank to reduce the amount of data
        it needs to read while avoiding seeks.

        Args:
            lengths (List[int]): A list of row counts for each file.
            rank (int): rank.
            world_size (int): world size.

        Returns:
            output (Dict[int, Tuple[int, int]]): Mapping of which files to the range in
                those files to be handled by the rank. The keys of this dict are indices
                of lengths.
        """

        # All ..._g variables are globals indices (meaning they range from 0 to
        # total_length - 1). All ..._l variables are local indices (meaning they range
        # from 0 to lengths[i] - 1 for the ith file).

        total_length = sum(lengths)
        rows_per_rank = total_length // world_size

        # Global indices that rank is responsible for. All ranges (left, right) are
        # inclusive.
        rank_left_g = rank * rows_per_rank
        rank_right_g = (rank + 1) * rows_per_rank - 1

        output = {}

        # Find where range (rank_left_g, rank_right_g) intersects each file's range.
        file_left_g, file_right_g = -1, -1
        for idx, length in enumerate(lengths):
            file_left_g = file_right_g + 1
            file_right_g = file_left_g + length - 1

            # If the ranges overlap.
            if rank_left_g <= file_right_g and rank_right_g >= file_left_g:
                overlap_left_g, overlap_right_g = max(rank_left_g, file_left_g), min(
                    rank_right_g, file_right_g
                )

                # Convert overlap in global numbers to (local) numbers specific to the
                # file.
                overlap_left_l = overlap_left_g - file_left_g
                overlap_right_l = overlap_right_g - file_left_g
                output[idx] = (overlap_left_l, overlap_right_l)

        return output

    @staticmethod
    def load_npy_range(
        fname: str,
        start_row: int,
        num_rows: int,
        path_manager_key: str = PATH_MANAGER_KEY,
    ) -> np.ndarray:
        """
        Load part of an npy file.

        NOTE: Assumes npy represents a numpy array of ndim 2.

        Args:
            fname (str): path string to npy file.
            start_row (int): starting row from the npy file.
            num_rows (int): number of rows to get from the npy file.
            path_manager_key (str): Path manager key used to load from different
                filesystems.

        Returns:
            output (np.ndarray): numpy array with the desired range of data from the
                supplied npy file.
        """
        path_manager = PathManagerFactory().get(path_manager_key)
        with path_manager.open(fname, "rb") as fin:
            np.lib.format.read_magic(fin)
            shape, _order, dtype = np.lib.format.read_array_header_1_0(fin)
            if len(shape) == 2:
                total_rows, row_size = shape
            else:
                raise ValueError("Cannot load range for npy with ndim == 2.")

            if not (0 <= start_row < total_rows):
                raise ValueError(
                    f"start_row ({start_row}) is out of bounds. It must be between 0 "
                    f"and {total_rows - 1}, inclusive."
                )
            if not (start_row + num_rows <= total_rows):
                raise ValueError(
                    f"num_rows ({num_rows}) exceeds number of available rows "
                    f"({total_rows}) for the given start_row ({start_row})."
                )

            offset = start_row * row_size * dtype.itemsize
            fin.seek(offset, os.SEEK_CUR)
            num_entries = num_rows * row_size
            data = np.fromfile(fin, dtype=dtype, count=num_entries)
            return data.reshape((num_rows, row_size))


class InMemoryBinaryCriteoIterDataPipe(IterDataPipe):
    """
    Datapipe designed to operate over binary (npy) versions of Criteo datasets. Loads
    the entire dataset into memory to prevent disk speed from affecting throughout. Each
    rank reads only the data for the portion of the dataset it is responsible for.

    The torchrec/datasets/scripts/preprocess_criteo.py script can be used to convert
    the Criteo tsv files to the npy files expected by this dataset.

    Args:
        dense_paths (List[str]): List of path strings to dense npy files.
        sparse_paths (List[str]): List of path strings to sparse npy files.
        labels_paths (List[str]): List of path strings to labels npy files.
        batch_size (int): batch size.
        rank (int): rank.
        world_size (int): world size.
        hashes (Optional[int]): List of max categorical feature value for each feature.
            Length of this list should be CAT_FEATURE_COUNT.
        path_manager_key (str): Path manager key used to load from different
            filesystems.

    Example:
    >>> template = "/home/datasets/criteo/1tb_binary/day_{}_{}.npy"
    >>> datapipe = InMemoryBinaryCriteoIterDataPipe(
    >>>     dense_paths=[template.format(0, "dense"), template.format(1, "dense")],
    >>>     sparse_paths=[template.format(0, "sparse"), template.format(1, "sparse")],
    >>>     labels_paths=[template.format(0, "labels"), template.format(1, "labels")],
    >>>     batch_size=1024,
    >>>     rank=torch.distributed.get_rank(),
    >>>     world_size=torch.distributed.get_world_size(),
    >>> )
    >>> batch = next(iter(datapipe))
    """

    def __init__(
        self,
        dense_paths: List[str],
        sparse_paths: List[str],
        labels_paths: List[str],
        batch_size: int,
        rank: int,
        world_size: int,
        hashes: Optional[List[int]] = None,
        path_manager_key: str = PATH_MANAGER_KEY,
    ) -> None:
        self.dense_paths = dense_paths
        self.sparse_paths = sparse_paths
        self.labels_paths = labels_paths
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.hashes = hashes
        self.path_manager_key = path_manager_key
        self.path_manager: PathManager = PathManagerFactory().get(path_manager_key)

        self._load_data_for_rank()
        self.num_rows_per_file: List[int] = [a.shape[0] for a in self.dense_arrs]
        self.num_batches: int = sum(self.num_rows_per_file) // batch_size

        # These values are the same for the KeyedJaggedTensors in all batches, so they
        # are computed once here. This avoids extra work from the KeyedJaggedTensor sync
        # functions.
        self._num_ids_in_batch: int = CAT_FEATURE_COUNT * batch_size
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.lengths: torch.Tensor = torch.ones(
            (self._num_ids_in_batch,), dtype=torch.int32
        )
        self.offsets: torch.Tensor = torch.arange(
            0, self._num_ids_in_batch + 1, dtype=torch.int32
        )
        self.stride = batch_size
        self.length_per_key: List[int] = CAT_FEATURE_COUNT * [batch_size]
        self.offset_per_key: List[int] = [
            batch_size * i for i in range(CAT_FEATURE_COUNT + 1)
        ]
        self.index_per_key: Dict[str, int] = {
            key: i for (i, key) in enumerate(self.keys)
        }

    def _load_data_for_rank(self) -> None:
        file_idx_to_row_range = BinaryCriteoUtils.get_file_idx_to_row_range(
            lengths=[
                BinaryCriteoUtils.get_shape_from_npy(
                    path, path_manager_key=self.path_manager_key
                )[0]
                for path in self.dense_paths
            ],
            rank=self.rank,
            world_size=self.world_size,
        )

        self.dense_arrs, self.sparse_arrs, self.labels_arrs = [], [], []
        for arrs, paths in zip(
            [self.dense_arrs, self.sparse_arrs, self.labels_arrs],
            [self.dense_paths, self.sparse_paths, self.labels_paths],
        ):
            for idx, (range_left, range_right) in file_idx_to_row_range.items():
                arrs.append(
                    BinaryCriteoUtils.load_npy_range(
                        paths[idx],
                        range_left,
                        range_right - range_left + 1,
                        path_manager_key=self.path_manager_key,
                    )
                )

        if self.hashes is not None:
            hashes_np = np.array(self.hashes).reshape((1, CAT_FEATURE_COUNT))
            for sparse_arr in self.sparse_arrs:
                sparse_arr %= hashes_np

    def _np_arrays_to_batch(
        self, dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray
    ) -> Batch:
        return Batch(
            dense_features=torch.from_numpy(dense),
            sparse_features=KeyedJaggedTensor(
                keys=self.keys,
                # transpose + reshape(-1) incurs an additional copy.
                values=torch.from_numpy(sparse.transpose(1, 0).reshape(-1)),
                lengths=self.lengths,
                offsets=self.offsets,
                stride=self.stride,
                length_per_key=self.length_per_key,
                offset_per_key=self.offset_per_key,
                index_per_key=self.index_per_key,
            ),
            labels=torch.from_numpy(labels.reshape(-1)),
        )

    def __iter__(self) -> Iterator[Batch]:
        # Invariant: buffer never contains more than batch_size rows.
        buffer: Optional[List[np.ndarray]] = None

        def append_to_buffer(
            dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray
        ) -> None:
            nonlocal buffer
            if buffer is None:
                buffer = [dense, sparse, labels]
            else:
                for idx, arr in enumerate([dense, sparse, labels]):
                    buffer[idx] = np.concatenate((buffer[idx], arr))

        # Maintain a buffer that can contain up to batch_size rows. Fill buffer as
        # much as possible on each iteration. Only return a new batch when batch_size
        # rows are filled.
        file_idx = 0
        row_idx = 0
        batch_idx = 0
        while batch_idx < self.num_batches:
            buffer_row_count = 0 if buffer is None else none_throws(buffer)[0].shape[0]
            if buffer_row_count == self.batch_size:
                yield self._np_arrays_to_batch(*none_throws(buffer))
                batch_idx += 1
                buffer = None
            else:
                rows_to_get = min(
                    self.batch_size - buffer_row_count,
                    self.num_rows_per_file[file_idx] - row_idx,
                )
                slice_ = slice(row_idx, row_idx + rows_to_get)
                append_to_buffer(
                    self.dense_arrs[file_idx][slice_, :],
                    self.sparse_arrs[file_idx][slice_, :],
                    self.labels_arrs[file_idx][slice_, :],
                )
                row_idx += rows_to_get

                if row_idx >= self.num_rows_per_file[file_idx]:
                    file_idx += 1
                    row_idx = 0

    def __len__(self) -> int:
        return self.num_batches
