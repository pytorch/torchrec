#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import abc
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torchmetrics import Metric
from torchrec.metrics.metrics_config import RecComputeMode, RecTaskInfo
from torchrec.metrics.metrics_namespace import (
    compose_metric_key,
    MetricNameBase,
    MetricNamespaceBase,
    MetricPrefix,
)


RecModelOutput = Union[torch.Tensor, Dict[str, torch.Tensor]]


@dataclass(frozen=True)
class MetricComputationReport:
    name: MetricNameBase
    metric_prefix: MetricPrefix
    value: torch.Tensor


DefaultValueT = TypeVar("DefaultValueT")
ComputeIterType = Iterator[
    Tuple[RecTaskInfo, MetricNameBase, torch.Tensor, MetricPrefix]
]

MAX_BUFFER_COUNT = 1000


class RecMetricException(Exception):
    pass


class WindowBuffer:
    def __init__(self, max_size: int, max_buffer_count: int) -> None:
        self._max_size: int = max_size
        self._max_buffer_count: int = max_buffer_count

        self._buffers: Deque[torch.Tensor] = deque(maxlen=max_buffer_count)
        self._used_sizes: Deque[int] = deque(maxlen=max_buffer_count)
        self._window_used_size = 0

    def aggregate_state(
        self, window_state: torch.Tensor, curr_state: torch.Tensor, size: int
    ) -> None:
        def remove(window_state: torch.Tensor) -> None:
            window_state -= self._buffers.popleft()
            self._window_used_size -= self._used_sizes.popleft()

        if len(self._buffers) == self._buffers.maxlen:
            remove(window_state)

        self._buffers.append(curr_state)
        self._used_sizes.append(size)
        window_state += curr_state
        self._window_used_size += size

        while self._window_used_size > self._max_size:
            remove(window_state)

    @property
    def buffers(self) -> Deque[torch.Tensor]:
        return self._buffers


class RecMetricComputation(Metric, abc.ABC):
    r"""The internal computation class template.
    A metric implementation should overwrite update() and compute(). These two
    APIs focuses the actual mathematical meaning of the metric, without the
    detail knowledge of model output and task information.

    Args:
        my_rank (int): the rank of this trainer.
        batch_size (int): batch size used by this trainer.
        n_tasks (int): the number tasks this communication obj
           will have to compute.
        window_size (int): the window size for the window metric.
        compute_on_all_ranks (bool): whether to compute metrics on all ranks. This
           is necessary if non-leader rank want to consum metrics result.
        process_group (Optional[ProcessGroup]): the process group used for the
            communication. Will use the default process group if not specified.
    """
    _batch_window_buffers: Optional[Dict[str, WindowBuffer]]

    def __init__(
        self,
        my_rank: int,
        batch_size: int,
        n_tasks: int,
        window_size: int,
        compute_on_all_ranks: bool = False,
        # pyre-fixme[11]: Annotation `ProcessGroup` is not defined as a type.
        process_group: Optional[dist.ProcessGroup] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(process_group=process_group, *args, **kwargs)

        self._my_rank = my_rank
        self._n_tasks = n_tasks
        self._batch_size = batch_size
        self._window_size = window_size
        self._compute_on_all_ranks = compute_on_all_ranks
        if self._window_size > 0:
            self._batch_window_buffers = {}
        else:
            self._batch_window_buffers = None
        self._add_state(
            "has_valid_update",
            torch.zeros(self._n_tasks, dtype=torch.uint8),
            add_window_state=False,
            dist_reduce_fx=lambda x: torch.any(x, dim=0).byte(),
            persistent=True,
        )

    @staticmethod
    def get_window_state_name(state_name: str) -> str:
        return f"window_{state_name}"

    def get_window_state(self, state_name: str) -> torch.Tensor:
        return getattr(self, self.get_window_state_name(state_name))

    def _add_state(
        self, name: str, default: DefaultValueT, add_window_state: bool, **kwargs: Any
    ) -> None:
        # pyre-fixme[6]: Expected `Union[List[typing.Any], torch.Tensor]` for 2nd
        #  param but got `DefaultValueT`.
        super().add_state(name, default, **kwargs)
        if add_window_state:
            if self._batch_window_buffers is None:
                raise RuntimeError(
                    "Users is adding a window state while window metric is disabled."
                )
            kwargs["persistent"] = False
            window_state_name = self.get_window_state_name(name)
            # Avoid pyre error
            assert isinstance(default, torch.Tensor)
            super().add_state(window_state_name, default.detach().clone(), **kwargs)
            self._batch_window_buffers[window_state_name] = WindowBuffer(
                max_size=self._window_size,
                max_buffer_count=MAX_BUFFER_COUNT,
            )

    def _aggregate_window_state(
        self, state_name: str, state: torch.Tensor, num_samples: int
    ) -> None:
        if self._batch_window_buffers is None:
            raise RuntimeError(
                "Users is adding a window state while window metric is disabled."
            )
        window_state_name = self.get_window_state_name(state_name)
        assert self._batch_window_buffers is not None
        self._batch_window_buffers[window_state_name].aggregate_state(
            getattr(self, window_state_name), curr_state=state, size=num_samples
        )

    @abc.abstractmethod
    # pyre-fixme[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> None:  # pragma: no cover
        pass

    @abc.abstractmethod
    def _compute(self) -> List[MetricComputationReport]:  # pragma: no cover
        pass

    def pre_compute(self) -> None:
        r"""If a metric need to do some work before `compute()`, the metric
        has to override this `pre_compute()`. One possible usage is to do
        some pre-processing of the local state before `compute()` as TorchMetric
        wraps `RecMetricComputation.compute()` and will do the global aggregation
        before `RecMetricComputation.compute()` is called.
        """
        return

    def compute(self) -> List[MetricComputationReport]:
        if self._my_rank == 0 or self._compute_on_all_ranks:
            return self._compute()
        else:
            return []

    def local_compute(self) -> List[MetricComputationReport]:
        return self._compute()


class RecMetric(nn.Module, abc.ABC):
    r"""The main class template to implement a recommendation metric.
    This class contains the recommendation tasks information (RecTaskInfo) and
    the actual computation object (RecMetricComputation). RecMetric processes
    all the information related to RecTaskInfo and models and pass the required
    signals to the computation object, allowing the implementation of
    RecMetricComputation to focus on the mathemetical meaning.

    A new metric that inherit RecMetric must override the following attributes
    in its own __init__(): `_namespace` and `_metrics_computations`. No other
    methods should be overridden.

    Args:
        world_size (int): the number of trainers.
        my_rank (int): the rank of this trainer.
        batch_size (int): batch size used by this trainer.
        tasks (List[RecTaskInfo]): the information of the model tasks.
        compute_mode (RecComputeMode): the computation mode. See RecComputeMode.
        window_size (int): the window size for the window metric.
        fused_update_limit (int): the maximum number of updates to be fused.
        compute_on_all_ranks (bool): whether to compute metrics on all ranks. This
           is necessary if non-leader rank want to consume global metrics result.
        process_group (Optional[ProcessGroup]): the process group used for the
            communication. Will use the default process group if not specified.

    Call Args:
        Not supported.

    Returns:
        Not supported.

    Example::

        ne = NEMetric(
                 world_size=4,
                 my_rank=0,
                 batch_size=128,
                 tasks=DefaultTaskInfo,
             )
    """
    _computation_class: Type[RecMetricComputation]
    _namespace: MetricNamespaceBase
    _metrics_computations: nn.ModuleList

    _tasks: List[RecTaskInfo]
    _window_size: int
    _tasks_iter: Callable[[str], ComputeIterType]
    _update_buffers: Dict[str, List[RecModelOutput]]
    _default_weights: Dict[Tuple[int, ...], torch.Tensor]

    PREDICTIONS: str = "predictions"
    LABELS: str = "labels"
    WEIGHTS: str = "weights"

    def __init__(
        self,
        world_size: int,
        my_rank: int,
        batch_size: int,
        tasks: List[RecTaskInfo],
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        window_size: int = 100,
        fused_update_limit: int = 0,
        compute_on_all_ranks: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        **kwargs: Any,
    ) -> None:
        # TODO(stellaya): consider to inherit from TorchMetrics.Metric or
        # TorchMetrics.MetricCollection.
        if (
            compute_mode == RecComputeMode.FUSED_TASKS_COMPUTATION
            and fused_update_limit > 0
        ):
            raise ValueError(
                "The fused tasks computation and the fused update cannot be set at the same time"
            )
        super().__init__()
        self._world_size = world_size
        self._my_rank = my_rank
        self._window_size = math.ceil(window_size / world_size)
        self._batch_size = batch_size
        self._tasks = tasks
        self._compute_mode = compute_mode
        self._fused_update_limit = fused_update_limit
        self._default_weights = {}
        self._update_buffers = {
            self.PREDICTIONS: [],
            self.LABELS: [],
            self.WEIGHTS: [],
        }
        if compute_mode == RecComputeMode.FUSED_TASKS_COMPUTATION:
            n_metrics = 1
            task_per_metric = len(self._tasks)
            self._tasks_iter = self._fused_tasks_iter
        else:
            n_metrics = len(self._tasks)
            task_per_metric = 1
            self._tasks_iter = self._unfused_tasks_iter

        self._metrics_computations: nn.ModuleList = nn.ModuleList(
            [
                # This Pyre error seems to be Pyre's bug as it can be inferred by mypy
                # according to https://github.com/python/mypy/issues/3048.
                # pyre-fixme[45]: Cannot instantiate abstract class `RecMetricCoputation`.
                self._computation_class(
                    my_rank,
                    batch_size,
                    task_per_metric,
                    self._window_size,
                    compute_on_all_ranks,
                    process_group,
                    **kwargs,
                )
                for _ in range(n_metrics)
            ]
        )

    # TODO(stellaya): Refactor the _[fused, unfused]_tasks_iter methods and replace the
    # compute_scope str input with an enum
    def _fused_tasks_iter(self, compute_scope: str) -> ComputeIterType:
        assert len(self._metrics_computations) == 1
        self._metrics_computations[0].pre_compute()
        for metric_report in getattr(
            self._metrics_computations[0], compute_scope + "compute"
        )():
            for task, metric_value, has_valid_update in zip(
                self._tasks,
                metric_report.value,
                self._metrics_computations[0].has_valid_update,
            ):
                valid_metric_value = (
                    metric_value
                    if has_valid_update > 0
                    else torch.zeros_like(metric_value)
                )
                yield task, metric_report.name, valid_metric_value, compute_scope + metric_report.metric_prefix.value

    def _unfused_tasks_iter(self, compute_scope: str) -> ComputeIterType:
        for task, metric_computation in zip(self._tasks, self._metrics_computations):
            metric_computation.pre_compute()
            for metric_report in getattr(
                metric_computation, compute_scope + "compute"
            )():
                valid_metric_value = (
                    metric_report.value
                    if metric_computation.has_valid_update[0] > 0
                    else torch.zeros_like(metric_report.value)
                )
                yield task, metric_report.name, valid_metric_value, compute_scope + metric_report.metric_prefix.value

    def _fuse_update_buffers(self) -> Dict[str, RecModelOutput]:
        def fuse(outputs: List[RecModelOutput]) -> RecModelOutput:
            assert len(outputs) > 0
            if isinstance(outputs[0], torch.Tensor):
                return torch.cat(cast(List[torch.Tensor], outputs))
            else:
                task_outputs: Dict[str, List[torch.Tensor]] = defaultdict(list)
                for output in outputs:
                    assert isinstance(output, dict)
                    for task_name, tensor in output.items():
                        task_outputs[task_name].append(tensor)
                return {
                    name: torch.cat(tensors) for name, tensors in task_outputs.items()
                }

        ret: Dict[str, RecModelOutput] = {}
        for key, output_list in self._update_buffers.items():
            if len(output_list) > 0:
                ret[key] = fuse(output_list)
            else:
                assert key == self.WEIGHTS
            output_list.clear()
        return ret

    def _check_fused_update(self, force: bool) -> None:
        if self._fused_update_limit <= 0:
            return
        if len(self._update_buffers[self.PREDICTIONS]) == 0:
            return
        if (
            not force
            and len(self._update_buffers[self.PREDICTIONS]) < self._fused_update_limit
        ):
            return
        fused_arguments = self._fuse_update_buffers()
        self._update(
            predictions=fused_arguments[self.PREDICTIONS],
            labels=fused_arguments[self.LABELS],
            weights=fused_arguments.get(self.WEIGHTS, None),
        )

    def _create_default_weights(self, predictions: torch.Tensor) -> torch.Tensor:
        weights = self._default_weights.get(predictions.size(), None)
        if weights is None:
            weights = torch.ones_like(predictions)
            self._default_weights[predictions.size()] = weights
        return weights

    def _check_nonempty_weights(self, weights: torch.Tensor) -> torch.Tensor:
        return torch.gt(torch.count_nonzero(weights, dim=-1), 0)

    def _update(
        self,
        *,
        predictions: RecModelOutput,
        labels: RecModelOutput,
        weights: Optional[RecModelOutput],
    ) -> None:
        with torch.no_grad():
            if self._compute_mode == RecComputeMode.FUSED_TASKS_COMPUTATION:
                assert isinstance(predictions, torch.Tensor)
                predictions = predictions.view(-1, self._batch_size)
                assert isinstance(labels, torch.Tensor)
                labels = labels.view(-1, self._batch_size)
                if weights is None:
                    weights = self._create_default_weights(predictions)
                else:
                    assert isinstance(weights, torch.Tensor)
                    weights = weights.view(-1, self._batch_size)
                has_valid_weights = self._check_nonempty_weights(weights)
                if torch.any(has_valid_weights):
                    self._metrics_computations[0].update(
                        predictions=predictions, labels=labels, weights=weights
                    )
                    self._metrics_computations[0].has_valid_update.logical_or_(
                        has_valid_weights
                    ).byte()
            else:
                for task, metric_ in zip(self._tasks, self._metrics_computations):
                    if task.name not in predictions:
                        continue
                    if torch.numel(predictions[task.name]) == 0:
                        assert torch.numel(labels[task.name]) == 0
                        assert weights is None or torch.numel(weights[task.name]) == 0
                        continue
                    task_predictions = predictions[task.name].view(1, -1)
                    task_labels = labels[task.name].view(1, -1)
                    if weights is None:
                        task_weights = self._create_default_weights(task_predictions)
                    else:
                        task_weights = weights[task.name].view(1, -1)
                    has_valid_weights = self._check_nonempty_weights(task_weights)
                    if torch.any(has_valid_weights):
                        metric_.update(
                            predictions=task_predictions,
                            labels=task_labels,
                            weights=task_weights,
                        )
                        metric_.has_valid_update.logical_or_(has_valid_weights).byte()

    def update(
        self,
        *,
        predictions: RecModelOutput,
        labels: RecModelOutput,
        weights: Optional[RecModelOutput],
    ) -> None:
        if self._fused_update_limit > 0:
            self._update_buffers[self.PREDICTIONS].append(predictions)
            self._update_buffers[self.LABELS].append(labels)
            if weights is not None:
                self._update_buffers[self.WEIGHTS].append(weights)
            self._check_fused_update(force=False)
        else:
            self._update(predictions=predictions, labels=labels, weights=weights)

    # The implementation of compute is very similar to local_compute, but compute overwrites
    # the abstract method compute in torchmetrics.Metric, which is wrapped by _wrap_compute
    def compute(self) -> Dict[str, torch.Tensor]:
        self._check_fused_update(force=True)
        ret = {}
        for task, metric_name, metric_value, prefix in self._tasks_iter(""):
            metric_key = compose_metric_key(
                self._namespace, task.name, metric_name, prefix
            )
            ret[metric_key] = metric_value
        return ret

    def local_compute(self) -> Dict[str, torch.Tensor]:
        self._check_fused_update(force=True)
        ret = {}
        for task, metric_name, metric_value, prefix in self._tasks_iter("local_"):
            metric_key = compose_metric_key(
                self._namespace, task.name, metric_name, prefix
            )
            ret[metric_key] = metric_value
        return ret

    def sync(self) -> None:
        for computation in self._metrics_computations:
            computation.sync()

    def unsync(self) -> None:
        for computation in self._metrics_computations:
            if computation._is_synced:
                computation.unsync()

    def reset(self) -> None:
        for computation in self._metrics_computations:
            computation.reset()

    def get_memory_usage(self) -> Dict[torch.Tensor, int]:
        r"""Estimates the memory of the rec metric instance's
        underlying tensors; returns the map of tensor to size
        """
        tensor_map = {}
        attributes_q = deque(self.__dict__.values())
        while attributes_q:
            attribute = attributes_q.popleft()
            if isinstance(attribute, torch.Tensor):
                tensor_map[attribute] = (
                    math.prod(attribute.shape) * attribute.element_size()
                )
            elif isinstance(attribute, WindowBuffer):
                attributes_q.extend(attribute.buffers)
            elif isinstance(attribute, Mapping):
                attributes_q.extend(attribute.values())
            elif isinstance(attribute, Sequence) and not isinstance(attribute, str):
                attributes_q.extend(attribute)
            elif hasattr(attribute, "__dict__") and not isinstance(attribute, Enum):
                attributes_q.extend(attribute.__dict__.values())
        return tensor_map

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # We need to flush the cached output to ensure checkpointing correctness.
        self._check_fused_update(force=True)
        destination = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return self._metrics_computations.state_dict(
            destination=destination,
            prefix=f"{prefix}_metrics_computations.",
            keep_vars=keep_vars,
        )


class RecMetricList(nn.Module):
    """
    A list module to encapulate multiple RecMetric instances and provide the
    same interfaces as RecMetric.

    Args:
        rec_metrics (List[RecMetric]: the list of the input RecMetrics.

    Call Args:
        Not supported.

    Returns:
        Not supported.

    Example::

        ne = NEMetric(
                 world_size=4,
                 my_rank=0,
                 batch_size=128,
                 tasks=DefaultTaskInfo
             )
        metrics = RecMetricList([ne])
    """

    rec_metrics: nn.ModuleList

    def __init__(self, rec_metrics: List[RecMetric]) -> None:
        # TODO(stellaya): consider to inherit from TorchMetrics.MetricCollection.
        # The prequsite to use MetricCollection is that RecMetric inherits from
        # TorchMetrics.Metric or TorchMetrics.MetricCollection

        super().__init__()
        self.rec_metrics = nn.ModuleList(rec_metrics)

    def __len__(self) -> int:
        return len(self.rec_metrics)

    def __getitem__(self, idx: int) -> nn.Module:
        return self.rec_metrics[idx]

    def update(
        self,
        *,
        predictions: RecModelOutput,
        labels: RecModelOutput,
        weights: RecModelOutput,
    ) -> None:
        for metric in self.rec_metrics:
            metric.update(predictions=predictions, labels=labels, weights=weights)

    def compute(self) -> Dict[str, torch.Tensor]:
        ret = {}
        for metric in self.rec_metrics:
            ret.update(metric.compute())
        return ret

    def local_compute(self) -> Dict[str, torch.Tensor]:
        ret = {}
        for metric in self.rec_metrics:
            ret.update(metric.local_compute())
        return ret

    def sync(self) -> None:
        for metric in self.rec_metrics:
            metric.sync()

    def unsync(self) -> None:
        for metric in self.rec_metrics:
            metric.unsync()

    def reset(self) -> None:
        for metric in self.rec_metrics:
            metric.reset()
