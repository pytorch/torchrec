#!/usr/bin/env python3

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

import torch.utils.data.datapipes as dp
from torch.utils.data import IterDataPipe
from torchrec.datasets.utils import LoadFiles, ReadLinesFromCSV, safe_cast


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


def _criteo(
    paths: Iterable[str],
    *,
    # pyre-ignore[2]
    row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    # pyre-ignore[2]
    **open_kw,
) -> IterDataPipe:
    datapipe = LoadFiles(paths, mode="r", **open_kw)
    datapipe = ReadLinesFromCSV(datapipe, delimiter="\t")
    if row_mapper:
        datapipe = dp.iter.Mapper(datapipe, row_mapper)
    return datapipe


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
        paths (str): local paths to TSV files that constitute the Criteo 1TB dataset.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each split TSV line.
        open_kw: options to pass to underlying invocation of iopath.common.file_io.PathManager.open.

    Example:
        >>> datapipe = criteo_terabyte(
        >>>     ("/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv")
        >>> )
        >>> datapipe = dp.iter.Batcher(datapipe, 100)
        >>> datapipe = dp.iter.Collator(datapipe)
        >>> batch = next(iter(datapipe))
    """
    return _criteo(paths, row_mapper=row_mapper, **open_kw)


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
    return _criteo((path,), row_mapper=row_mapper, **open_kw)
