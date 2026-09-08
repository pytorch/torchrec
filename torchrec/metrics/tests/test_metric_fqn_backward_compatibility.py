#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
These tests compares current metric FQNs against a golden baseline and fails if:
1. Any state_dict key is REMOVED (breaks loading old checkpoints into new code)
2. Any state_dict key is ADDED (breaks loading old checkpoints in DCP clients
   unless allow_partial_load=True, which most production trainers don't use)

## How to Fix Breaking Changes

If you need to add a new buffer/state to a metric:
1. Consider if it can be non-persistent (won't appear in state_dict)
2. If it must be persistent, coordinate with the trainers team to enable
   allow_partial_load for metrics, OR add a migration path
3. Update the golden snapshot with --update-golden after confirming the change
   won't break production training jobs

To update the golden snapshot after intentional changes:
    python -m torchrec.metrics.tests.test_metric_fqn_backward_compatibility --update-golden
"""

import contextlib
import copy
import datetime
import inspect
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)

import torch
import torch.distributed as dist
from torchrec.checkpoint.schema import registered_schema_stable_classes
from torchrec.metrics.accuracy import AccuracyMetric
from torchrec.metrics.auc import AUCMetric
from torchrec.metrics.auprc import AUPRCMetric
from torchrec.metrics.average import AverageMetric
from torchrec.metrics.cali_free_ne import CaliFreeNEMetric
from torchrec.metrics.calibration import CalibrationMetric
from torchrec.metrics.calibration_with_recalibration import (
    RecalibratedCalibrationMetric,
)
from torchrec.metrics.cpu_comms_metric_module import CPUCommsRecMetricModule
from torchrec.metrics.cpu_offloaded_metric_module import CPUOffloadedRecMetricModule
from torchrec.metrics.ctr import CTRMetric
from torchrec.metrics.gauc import GAUCMetric
from torchrec.metrics.hindsight_target_pr import HindsightTargetPRMetric
from torchrec.metrics.mae import MAEMetric
from torchrec.metrics.metric_module import RecMetricModule
from torchrec.metrics.metrics_config import BatchSizeStage, RecComputeMode, RecTaskInfo
from torchrec.metrics.mse import MSEMetric
from torchrec.metrics.multi_label_precision import MultiLabelPrecisionMetric
from torchrec.metrics.multiclass_recall import MulticlassRecallMetric
from torchrec.metrics.ndcg import NDCGMetric
from torchrec.metrics.ne import NEMetric
from torchrec.metrics.ne_positive import NEPositiveMetric
from torchrec.metrics.ne_with_recalibration import RecalibratedNEMetric
from torchrec.metrics.nmse import NMSEMetric
from torchrec.metrics.noop_metric_module import NoOpMetricModule
from torchrec.metrics.num_missing_labels import NumMissingLabelsMetric
from torchrec.metrics.num_positive_samples import NumPositiveSamplesMetric
from torchrec.metrics.output import OutputMetric
from torchrec.metrics.precision import PrecisionMetric
from torchrec.metrics.precision_session import PrecisionSessionMetric
from torchrec.metrics.rauc import RAUCMetric
from torchrec.metrics.rec_metric import RecMetric, RecMetricException, RecMetricList
from torchrec.metrics.recall import RecallMetric
from torchrec.metrics.recall_session import RecallSessionMetric
from torchrec.metrics.scalar import ScalarMetric
from torchrec.metrics.segmented_ne import SegmentedNEMetric
from torchrec.metrics.serving_calibration import ServingCalibrationMetric
from torchrec.metrics.serving_ne import ServingNEMetric
from torchrec.metrics.sum_weights import SumWeightsMetric
from torchrec.metrics.tensor_weighted_avg import TensorWeightedAvgMetric
from torchrec.metrics.throughput import ThroughputMetric
from torchrec.metrics.tower_qps import TowerQPSMetric
from torchrec.metrics.unweighted_ne import UnweightedNEMetric
from torchrec.metrics.weighted_avg import WeightedAvgMetric
from torchrec.metrics.weighted_sum_predictions import WeightedSumPredictionsMetric
from torchrec.metrics.xauc import XAUCMetric
from torchrec.test_utils import init_process_group_single_rank


# Path to the golden snapshot file
GOLDEN_SNAPSHOT_PATH = Path(__file__).parent / "metric_fqn_golden_snapshot.json"


def create_test_task(
    task_name: str = "test_task",
    with_tensor_name: bool = False,
    with_session_metric_def: bool = False,
) -> RecTaskInfo:
    from torchrec.metrics.metrics_config import SessionMetricDef

    session_metric_def = None
    if with_session_metric_def:
        session_metric_def = SessionMetricDef(
            session_var_name=f"{task_name}-session",
            top_threshold=1,
            run_ranking_of_labels=False,
        )

    return RecTaskInfo(
        name=task_name,
        label_name=f"{task_name}-label",
        prediction_name=f"{task_name}-prediction",
        weight_name=f"{task_name}-weight",
        tensor_name=f"{task_name}-tensor" if with_tensor_name else None,
        session_metric_def=session_metric_def,
    )


def build_metric(
    metric_class: Type[RecMetric],
    compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
    task_names: Optional[List[str]] = None,
    use_tensor_task: bool = False,
    use_session_task: bool = False,
    **kwargs: Any,
) -> RecMetric:
    """One metric under the fixed configuration every golden entry is taken at.

    Callers used to build the metric once to read its state_dict keys and again
    to read its buffers, which doubled the construction work and let the two
    reads drift apart.
    """
    if task_names is None:
        task_names = ["test_task"]

    tasks = [
        create_test_task(
            name,
            with_tensor_name=use_tensor_task,
            with_session_metric_def=use_session_task,
        )
        for name in task_names
    ]

    return metric_class(
        world_size=1,
        my_rank=0,
        batch_size=32,
        tasks=tasks,
        compute_mode=compute_mode,
        window_size=100,
        fused_update_limit=0,
        **kwargs,
    )


def extract_state_dict_keys(
    metric_class: Type[RecMetric],
    compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
    **kwargs: Any,
) -> List[str]:
    return sorted(
        build_metric(metric_class, compute_mode, **kwargs).state_dict().keys()
    )


def buffer_fqns(module: torch.nn.Module) -> Tuple[List[str], List[str]]:
    """Buffer names split by whether they reach the state_dict.

    A buffer is in the state_dict exactly when it is persistent and not None,
    so membership answers both questions. The earlier version matched a buffer
    name against any state key containing it, which called a buffer persistent
    on a coincidental substring.
    """
    saved = set(module.state_dict().keys())
    persistent: List[str] = []
    non_persistent: List[str] = []
    for name, value in module.named_buffers():
        if value is None:
            continue
        (persistent if name in saved else non_persistent).append(name)
    return sorted(persistent), sorted(non_persistent)


def get_metric_snapshot_key(
    metric_class: Type[RecMetric],
    compute_mode: RecComputeMode,
    variant: str = "",
) -> str:
    key = f"{metric_class.__name__}_{compute_mode.name}"
    if variant:
        key = f"{key}_{variant}"
    return key


# List of metrics to test with their configurations
# Format: (metric_class, compute_modes_to_test, extra_kwargs, variants)
# ThroughputMetric is excluded as it's not a RecMetric subclass (it's nn.Module)
METRICS_TO_TEST: List[
    Tuple[Type[RecMetric], List[RecComputeMode], Dict[str, Any], List[str]]
] = [
    # Core metrics with persistent state
    # include_logloss only gates which metrics _compute reports, so it adds no
    # state and produced an entry identical to the default one above.
    (NEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (CalibrationMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (CTRMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (MSEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (
        MSEMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"include_r_squared": True},
        ["with_r_squared"],
    ),
    (MAEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (WeightedAvgMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (AccuracyMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (PrecisionMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (RecallMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (TowerQPSMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (NMSEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (AverageMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (HindsightTargetPRMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (NDCGMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (XAUCMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (ScalarMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (
        MultiLabelPrecisionMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"num_labels": 1},
        [""],
    ),
    # Metrics with non-persistent state (AUC family)
    (AUCMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (AUPRCMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (RAUCMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (GAUCMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    # TensorWeightedAvgMetric requires tensor_name in tasks
    (
        TensorWeightedAvgMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"use_tensor_task": True},
        [""],
    ),
    (CaliFreeNEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (NEPositiveMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (ServingNEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (UnweightedNEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (RecalibratedNEMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (ServingCalibrationMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (
        RecalibratedCalibrationMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {},
        [""],
    ),
    (OutputMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    # MulticlassRecallMetric requires number_of_classes
    (
        MulticlassRecallMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"number_of_classes": 3},
        [""],
    ),
    # SegmentedNEMetric requires num_groups and grouping_keys
    (
        SegmentedNEMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"num_groups": 2, "grouping_keys": "test_task-grouping"},
        [""],
    ),
    # Session-level metrics require session_metric_def in tasks
    (
        PrecisionSessionMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"use_session_task": True},
        [""],
    ),
    (
        RecallSessionMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {"use_session_task": True},
        [""],
    ),
    # FUSED mode tests
    (NEMetric, [RecComputeMode.FUSED_TASKS_COMPUTATION], {}, [""]),
    (CalibrationMetric, [RecComputeMode.FUSED_TASKS_COMPUTATION], {}, [""]),
    (WeightedAvgMetric, [RecComputeMode.FUSED_TASKS_COMPUTATION], {}, [""]),
    # New utility metrics
    (NumMissingLabelsMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (NumPositiveSamplesMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (SumWeightsMetric, [RecComputeMode.UNFUSED_TASKS_COMPUTATION], {}, [""]),
    (
        WeightedSumPredictionsMetric,
        [RecComputeMode.UNFUSED_TASKS_COMPUTATION],
        {},
        [""],
    ),
]


def generate_golden_snapshot() -> Dict[str, Dict[str, Any]]:
    """
    Generate a complete golden snapshot of all metrics.

    Every entry must build. Skipping a metric that raises would write a file
    missing that metric, and the write still succeeds, so a partial snapshot
    replaces a good one.

    Returns:
        Dictionary mapping metric keys to their FQN information.
    """
    snapshot: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []

    for metric_class, compute_modes, kwargs, variants in METRICS_TO_TEST:
        for compute_mode in compute_modes:
            for variant in variants:
                key = get_metric_snapshot_key(metric_class, compute_mode, variant)
                if key in snapshot:
                    raise ValueError(
                        f"Two METRICS_TO_TEST rows produce the key {key!r}. The "
                        "second would overwrite the first, and both rows would "
                        "then check against one baseline. Give the rows "
                        "different variant names."
                    )
                try:
                    metric = build_metric(metric_class, compute_mode, **kwargs)
                    state_dict_keys = sorted(metric.state_dict().keys())
                    persistent_fqns, non_persistent_fqns = buffer_fqns(metric)
                except Exception as e:
                    failures.append(f"  {key}: {type(e).__name__}: {e}")
                    continue

                snapshot[key] = {
                    "metric_class": metric_class.__name__,
                    "compute_mode": compute_mode.name,
                    "variant": variant,
                    "state_dict_keys": state_dict_keys,
                    "persistent_buffer_fqns": persistent_fqns,
                    "non_persistent_buffer_fqns": non_persistent_fqns,
                }

    if failures:
        detail = "\n".join(failures)
        raise RuntimeError(f"Could not build every metric in the table:\n{detail}")

    return snapshot


def load_golden_snapshot() -> Dict[str, Dict[str, Any]]:
    if not GOLDEN_SNAPSHOT_PATH.exists():
        return {}
    with open(GOLDEN_SNAPSHOT_PATH, "r") as f:
        return json.load(f)


def load_required_golden_snapshot() -> Dict[str, Dict[str, Any]]:
    """The snapshot, for callers that say nothing useful without one.

    A test comparing against an empty file passes by having nothing to
    compare, which reads as coverage.

    Regeneration against an empty baseline is worse: every generated entry
    looks brand new, so the addition and removal gates find nothing to object
    to and the result becomes the baseline. Emptying this file is the cheapest
    way to defeat them.
    """
    snapshot = load_golden_snapshot()
    if not snapshot:
        raise RuntimeError(
            f"{GOLDEN_SNAPSHOT_PATH} is missing or empty. It is checked in, so "
            "restore it from source control rather than writing a new one. A "
            "regenerated baseline authorizes whatever the code produces today."
        )
    return snapshot


def save_golden_snapshot(snapshot: Dict[str, Dict[str, Any]]) -> None:
    with open(GOLDEN_SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
        f.write("\n")


class MetricFQNBackwardCompatibilityTest(unittest.TestCase):
    """
    Test suite for metric FQN backward compatibility.

    These tests ensure that changes to metrics don't break checkpoint loading
    by verifying that:
    1. No persistent buffer FQNs are removed
    2. No state_dict keys are removed

    New additions are allowed as long as they have proper load_state_dict hooks
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden_snapshot = load_required_golden_snapshot()
        cls.current_snapshot = None

    def _check_metric_compatibility(
        self,
        metric_class: Type[RecMetric],
        compute_mode: RecComputeMode,
        variant: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Check a single metric for backward compatibility with DCP CheckpointClient.

        This test simulates what happens when a new model version (with potentially
        new state_dict keys) tries to load an old checkpoint:

        1. REMOVED keys: Old checkpoint has keys that new model doesn't expect.
           - This typically works fine (load_state_dict ignores extra keys)
           - But indicates the metric structure changed unexpectedly

        2. ADDED keys: New model has keys that old checkpoint doesn't have.
           - This BREAKS DCP CheckpointClient! The client validates that ALL model
             FQNs exist in the checkpoint metadata before loading.
           - Raises InvalidParamQualNameException at the DCP level.

        Raises assertion error if incompatible changes are detected.
        """
        key = get_metric_snapshot_key(metric_class, compute_mode, variant)

        if key not in self.golden_snapshot:
            self.fail(
                f"No golden entry for {key}. Run with --update-golden to create it. "
                "Skipping here would silently drop this metric's regression "
                "coverage the moment its entry goes missing."
            )

        baseline = self.golden_snapshot[key]

        metric = build_metric(metric_class, compute_mode, **kwargs)
        current_state_dict_keys = set(metric.state_dict().keys())
        current_persistent_fqns, _ = buffer_fqns(metric)
        current_persistent_fqns_set = set(current_persistent_fqns)

        baseline_state_dict_keys = set(baseline["state_dict_keys"])
        baseline_persistent_fqns = set(baseline["persistent_buffer_fqns"])

        removed_keys = baseline_state_dict_keys - current_state_dict_keys
        if removed_keys:
            self.fail(
                f"BREAKING CHANGE in {key}: state_dict keys removed: {sorted(removed_keys)}. "
                "This will cause old checkpoints to fail loading. "
                "If this is intentional, update the golden snapshot."
            )

        removed_fqns = baseline_persistent_fqns - current_persistent_fqns_set
        if removed_fqns:
            self.fail(
                f"BREAKING CHANGE in {key}: persistent buffer FQNs removed: {sorted(removed_fqns)}. "
                "This will cause old checkpoints to fail loading. "
                "If this is intentional, update the golden snapshot."
            )

        added_keys = current_state_dict_keys - baseline_state_dict_keys
        added_fqns = current_persistent_fqns_set - baseline_persistent_fqns

        if added_keys or added_fqns:
            has_backward_compat = self._verify_backward_compatibility(
                metric_class, compute_mode, added_keys, **kwargs
            )

            if not has_backward_compat:
                self.fail(
                    f"BREAKING CHANGE in {key}: state_dict keys added: {sorted(added_keys)}.\n"
                    "This will cause DCP CheckpointClient to fail with InvalidParamQualNameException "
                    "when loading old checkpoints (the client validates that ALL model FQNs exist "
                    "in checkpoint metadata before loading).\n\n"
                    "To fix:\n"
                    "1. Make the new state non-persistent (use persistent=False in add_state)\n"
                    "2. OR coordinate with trainers to enable allow_partial_load for metrics\n"
                    "3. OR run with --update-golden if this change is intentional and coordinated"
                )

    def _verify_backward_compatibility(
        self,
        metric_class: Type[RecMetric],
        compute_mode: RecComputeMode,
        added_keys: Set[str],
        **kwargs: Any,
    ) -> bool:
        """
        Check if added state_dict keys are backward compatible.

        Note: While torchmetrics.Metric handles missing keys gracefully at the
        PyTorch load_state_dict level (keeping default values), this does NOT
        help at the DCP CheckpointClient level. The DCP client validates FQNs
        BEFORE calling load_state_dict, so it fails before PyTorch hooks run.

        For added keys, we always return False to require explicit acknowledgment:
        1. Confirm the change is intentional
        2. Verify coordination with trainer teams
        3. Update the golden snapshot with --update-golden

        Returns:
            True if no added keys (backward compatible)
            False if there are added keys (requires developer acknowledgment)
        """
        # If there are added keys, require user to update golden snapshot
        return len(added_keys) == 0

    def test_every_metric_in_the_table(self) -> None:
        """One check per METRICS_TO_TEST row, driven by the table itself.

        Hand-written methods used to do this, one per row. Deleting one left
        its golden entry in place and the suite green, which is the coverage
        loss test_no_orphaned_golden_entries is meant to catch. Generation,
        checking, and that orphan check now read the same table.
        """
        for metric_class, compute_modes, kwargs, variants in METRICS_TO_TEST:
            for compute_mode in compute_modes:
                for variant in variants:
                    with self.subTest(
                        metric=metric_class.__name__,
                        compute_mode=compute_mode.name,
                        variant=variant,
                    ):
                        self._check_metric_compatibility(
                            metric_class, compute_mode, variant=variant, **kwargs
                        )


class MetricStateSnapshotTest(unittest.TestCase):
    """
    Test suite for verifying metric state snapshot capabilities.

    These tests verify that metrics properly register their states
    and can be serialized/deserialized correctly.
    """

    def _test_metric_state_roundtrip(
        self,
        metric_class: Type[RecMetric],
        compute_mode: RecComputeMode = RecComputeMode.UNFUSED_TASKS_COMPUTATION,
        **kwargs: Any,
    ) -> None:
        tasks = [create_test_task("task1")]

        original = metric_class(
            world_size=1,
            my_rank=0,
            batch_size=32,
            tasks=tasks,
            compute_mode=compute_mode,
            window_size=100,
            **kwargs,
        )

        initial_state = original.state_dict()

        restored = metric_class(
            world_size=1,
            my_rank=0,
            batch_size=32,
            tasks=tasks,
            compute_mode=compute_mode,
            window_size=100,
            **kwargs,
        )

        restored.load_state_dict(initial_state, strict=True)

        restored_state = restored.state_dict()
        self.assertEqual(
            set(initial_state.keys()),
            set(restored_state.keys()),
            f"State dict keys mismatch for {metric_class.__name__}",
        )

        for key in initial_state:
            torch.testing.assert_close(
                initial_state[key],
                restored_state[key],
                msg=f"State mismatch for key {key} in {metric_class.__name__}",
            )

    def test_ne_metric_state_roundtrip(self) -> None:
        self._test_metric_state_roundtrip(NEMetric)

    def test_calibration_metric_state_roundtrip(self) -> None:
        self._test_metric_state_roundtrip(CalibrationMetric)

    def test_mse_metric_state_roundtrip(self) -> None:
        self._test_metric_state_roundtrip(MSEMetric)

    def test_weighted_avg_metric_state_roundtrip(self) -> None:
        self._test_metric_state_roundtrip(WeightedAvgMetric)


# Fixture values for the golden cases.
_THROUGHPUT_BATCH_SIZE_STAGES: List[BatchSizeStage] = [
    BatchSizeStage(batch_size=32, max_iters=100),
    BatchSizeStage(batch_size=64, max_iters=None),
]


def _make_throughput_metric(
    batch_size_stages: Optional[List[BatchSizeStage]] = None,
) -> ThroughputMetric:
    return ThroughputMetric(
        batch_size=32,
        world_size=1,
        window_seconds=100,
        warmup_steps=10,
        batch_size_stages=batch_size_stages,
    )


def _make_rec_metric_module(
    throughput_metric: Optional[ThroughputMetric] = None,
) -> RecMetricModule:
    tasks = [create_test_task("task1")]
    return RecMetricModule(
        batch_size=32,
        world_size=1,
        rec_tasks=tasks,
        rec_metrics=RecMetricList(
            [
                NEMetric(
                    world_size=1,
                    my_rank=0,
                    batch_size=32,
                    tasks=tasks,
                    compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
                    window_size=100,
                )
            ]
        ),
        throughput_metric=throughput_metric,
    )


@dataclass(frozen=True)
class _GoldenCase:
    """One golden snapshot: which entry it is, and how to build it.

    The builder is self-contained, so a case cannot be paired with another
    case's configuration. `stable_id` is written by hand rather than derived
    from the class, so moving a file does not silently change the key and
    abandon the entry it used to match.
    """

    stable_id: str
    expected_class: Type[torch.nn.Module]
    variant: str
    build: Callable[[], torch.nn.Module]
    # Called on every module this case builds. A case whose module owns threads
    # or other process state declares how to release it.
    cleanup: Optional[Callable[[torch.nn.Module], None]] = None

    @property
    def key(self) -> str:
        return f"{self.stable_id}_{self.variant}" if self.variant else self.stable_id


_REC_METRIC_MODULE_DEFAULT = _GoldenCase(
    "RecMetricModule", RecMetricModule, "", _make_rec_metric_module
)

_CORE_SCHEMA_CASES: Tuple[_GoldenCase, ...] = (
    _GoldenCase("ThroughputMetric", ThroughputMetric, "", _make_throughput_metric),
    _GoldenCase(
        "ThroughputMetric",
        ThroughputMetric,
        "with_batch_size_stages",
        lambda: _make_throughput_metric(_THROUGHPUT_BATCH_SIZE_STAGES),
    ),
    _REC_METRIC_MODULE_DEFAULT,
    _GoldenCase(
        "RecMetricModule",
        RecMetricModule,
        "with_throughput",
        lambda: _make_rec_metric_module(_make_throughput_metric()),
    ),
)


def _module_fixture_kwargs() -> Dict[str, Any]:
    """Construction args carrying one real metric.

    An empty RecMetricList would produce an empty state_dict, and a golden entry
    with no keys records nothing about the module's shape.
    """
    tasks = [create_test_task("task1")]
    return {
        "batch_size": 32,
        "world_size": 1,
        "rec_tasks": tasks,
        "rec_metrics": RecMetricList(
            [
                NEMetric(
                    world_size=1,
                    my_rank=0,
                    batch_size=32,
                    tasks=tasks,
                    compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
                    window_size=100,
                )
            ]
        ),
    }


def _shutdown(module: torch.nn.Module) -> None:
    """Release a CPUOffloadedRecMetricModule's worker threads.

    Two threads plus an atexit hook per instance, and one case builds three.
    shutdown() is idempotent.
    """
    # pyre-ignore[16]
    module.shutdown()


@contextlib.contextmanager
def _single_rank_process_group() -> Iterator[None]:
    """A process group for CPUOffloadedRecMetricModule's compute worker.

    shutdown() wakes that worker, which calls dist.new_group and then re-raises
    whatever it caught. So building a module without a group succeeds and its
    teardown throws, which is why generation needs this as much as the tests do.

    Only destroys a group it created, so a caller that brought its own keeps it.
    """
    created = not dist.is_initialized()
    if created:
        init_process_group_single_rank("gloo")
    try:
        yield
    finally:
        if created:
            dist.destroy_process_group()


# Enrolled with @checkpoint_schema_stable under these same ids. Not RecMetric
# subclasses, so _discover_all_recmetric_subclasses cannot see them.
_MODULE_SCHEMA_CASES: Tuple[_GoldenCase, ...] = (
    _GoldenCase("noop_metric_module", NoOpMetricModule, "", NoOpMetricModule),
    _GoldenCase(
        "cpu_comms_rec_metric_module",
        CPUCommsRecMetricModule,
        "",
        lambda: CPUCommsRecMetricModule(**_module_fixture_kwargs()),
    ),
    # state_dict() reads the comms tree and load_state_dict() writes the offloaded
    # one, but both carry the same keys, so this pins the shape and not which tree
    # a load reached. test_cpu_offloaded_metric_module covers the load direction.
    _GoldenCase(
        "cpu_offloaded_rec_metric_module",
        CPUOffloadedRecMetricModule,
        "",
        lambda: CPUOffloadedRecMetricModule(
            model_out_device=torch.device("cpu"), **_module_fixture_kwargs()
        ),
        cleanup=_shutdown,
    ),
)

_SCHEMA_CASES: Tuple[_GoldenCase, ...] = _CORE_SCHEMA_CASES + _MODULE_SCHEMA_CASES


class GoldenCaseTest(unittest.TestCase):
    """Compares each golden case against the entry it wrote.

    Checking the stored metadata matters as much as the keys. Comparing one
    case against another's baseline can otherwise pass: the extra keys look
    like an addition, and the addition check loads them away.
    """

    golden_snapshot: Dict[str, Dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden_snapshot = load_required_golden_snapshot()

    def setUp(self) -> None:
        # Registered before any module is built, so under addCleanup's LIFO
        # order the group is torn down last. Destroying it ahead of a module
        # would make that module's shutdown throw.
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(_single_rank_process_group())

    @contextlib.contextmanager
    def _built(self, case: _GoldenCase) -> Iterator[torch.nn.Module]:
        module = case.build()
        try:
            yield module
        finally:
            if case.cleanup is not None:
                case.cleanup(module)

    def _check_case(self, case: _GoldenCase) -> None:
        with self._built(case) as module:
            self.assertIs(type(module), case.expected_class)

            if case.key not in self.golden_snapshot:
                self.fail(
                    f"No golden entry for {case.key}. Run with --update-golden "
                    "to create it. Writing one here would bless whatever the "
                    "code currently produces, and concurrent tests writing the "
                    "whole file at once corrupt it."
                )

            entry = self.golden_snapshot[case.key]
            self.assertEqual(
                entry["metric_class"],
                case.expected_class.__name__,
                f"Golden entry {case.key} was written by a different class.",
            )
            self.assertEqual(
                entry["variant"],
                case.variant,
                f"Golden entry {case.key} was written by a different variant.",
            )

            current_keys = set(module.state_dict().keys())

        baseline_keys = set(entry["state_dict_keys"])

        removed = baseline_keys - current_keys
        if removed:
            self.fail(
                f"BREAKING CHANGE in {case.key}: state_dict keys removed: "
                f"{sorted(removed)}."
            )

        added = current_keys - baseline_keys
        if not added:
            return

        self.fail(
            f"BREAKING CHANGE in {case.key}: state_dict keys added: "
            f"{sorted(added)}. {self._why_an_older_checkpoint_breaks(case, added)}"
        )

    def _why_an_older_checkpoint_breaks(
        self, case: _GoldenCase, added: Set[str]
    ) -> str:
        """The best available explanation for an added key, not a verdict.

        A strict load that raises names the exact key. One that succeeds proves
        nothing: DCP validates every model FQN against checkpoint metadata
        before load_state_dict runs, and state a strict load cannot see, such
        as anything torchmetrics holds by setattr, is state the planner still
        demands. So an addition always fails and this only picks the wording.

        Build a separate source and destination so neither is a module already
        inspected.
        """
        with self._built(case) as source, self._built(case) as destination:
            older = {k: v for k, v in source.state_dict().items() if k not in added}
            try:
                destination.load_state_dict(older, strict=True)
            except Exception as e:
                return f"A checkpoint without them no longer loads: {e}"
        return (
            "An in-process load still succeeds, so the key is invisible to the "
            "strict check, but DCP rejects the checkpoint before that runs."
        )

    def test_golden_cases(self) -> None:
        for case in _SCHEMA_CASES:
            with self.subTest(case.key):
                self._check_case(case)

    def test_metadata_mismatch_is_rejected(self) -> None:
        """The stored metric_class and variant must match the case asking.

        Without this, a case can compare against an entry another case wrote:
        the extra keys read as an addition and the addition check loads them
        away. Injecting each mismatch pins both guards, which are otherwise
        only exercised when the golden is already wrong.
        """
        case = _CORE_SCHEMA_CASES[0]
        for field_name, wrong_value in (
            ("metric_class", "SomeOtherClass"),
            ("variant", "some_other_variant"),
        ):
            with self.subTest(field_name):
                entry = dict(type(self).golden_snapshot[case.key])
                entry[field_name] = wrong_value
                # Shadows the class attribute for this instance only; the real
                # snapshot is untouched and no restore is needed.
                self.golden_snapshot = {case.key: entry}
                try:
                    with self.assertRaises(AssertionError) as cm:
                        self._check_case(case)
                    self.assertIn("written by a different", str(cm.exception))
                finally:
                    del self.golden_snapshot

    def test_no_orphaned_golden_entries(self) -> None:
        """Every golden entry must belong to something that checks it.

        An entry nobody claims is dead weight that still looks like coverage.
        That happens when a metric is dropped from METRICS_TO_TEST, or a case is
        removed, and the entry is left behind.
        """
        expected = {case.key for case in _SCHEMA_CASES}
        for metric_class, compute_modes, _kwargs, variants in METRICS_TO_TEST:
            for compute_mode in compute_modes:
                for variant in variants:
                    expected.add(
                        get_metric_snapshot_key(metric_class, compute_mode, variant)
                    )

        orphans = sorted(set(self.golden_snapshot) - expected)
        if orphans:
            self.fail(
                f"Golden entries that no test claims: {orphans}.\n"
                "Either restore whatever used to check them, or delete the "
                "entries."
            )

    def test_case_keys_are_unique(self) -> None:
        keys = [case.key for case in _SCHEMA_CASES]
        self.assertEqual(sorted(keys), sorted(set(keys)))

    def test_one_class_per_stable_id(self) -> None:
        """A stable id names one class, however many variants it has.

        The key carries the variant, so two classes could share an id and still
        keep distinct keys. Anything that maps id to class would then silently
        keep whichever row it read last.
        """
        classes_by_id: Dict[str, Set[Type[torch.nn.Module]]] = {}
        for case in _SCHEMA_CASES:
            classes_by_id.setdefault(case.stable_id, set()).add(case.expected_class)

        for stable_id, classes in classes_by_id.items():
            with self.subTest(stable_id):
                self.assertEqual(
                    len(classes),
                    1,
                    f"{stable_id} is claimed by "
                    f"{sorted(c.__name__ for c in classes)}.",
                )

    def test_cases_for_same_id_have_distinct_key_sets(self) -> None:
        """Every case under one id must snapshot a different key set.

        The unnamed case counts too, so this catches a named variant whose
        builder forgot the argument that distinguishes it. The key and the
        stored metadata both agree with themselves in that case; only the
        resulting shape disagrees.
        """
        by_id: Dict[str, List[FrozenSet[str]]] = {}
        for case in _SCHEMA_CASES:
            with self._built(case) as module:
                by_id.setdefault(case.stable_id, []).append(
                    frozenset(module.state_dict().keys())
                )

        for stable_id, entries in by_id.items():
            with self.subTest(stable_id):
                self.assertEqual(
                    len(set(entries)),
                    len(entries),
                    f"{stable_id} has {len(entries)} cases but "
                    f"{len(set(entries))} distinct key sets. One builder is carrying "
                    "another's configuration.",
                )


class ThroughputMetricBackwardCompatibilityTest(unittest.TestCase):
    """Round-trip checks for ThroughputMetric that are not golden comparisons."""

    def test_throughput_metric_state_roundtrip(self) -> None:
        metric = _make_throughput_metric()
        initial_state = metric.state_dict()

        fresh_metric = _make_throughput_metric()
        fresh_metric.load_state_dict(initial_state, strict=True)

        restored_state = fresh_metric.state_dict()
        self.assertEqual(set(initial_state.keys()), set(restored_state.keys()))

        for key in initial_state:
            torch.testing.assert_close(
                initial_state[key],
                restored_state[key],
                msg=f"State mismatch for key {key}",
            )


class RecMetricModuleBackwardCompatibilityTest(unittest.TestCase):
    """Round-trip checks for RecMetricModule that are not golden comparisons."""

    def test_rec_metric_module_state_roundtrip(self) -> None:
        module = _make_rec_metric_module(_make_throughput_metric())
        initial_state = module.state_dict()

        fresh_module = _make_rec_metric_module(_make_throughput_metric())
        fresh_module.load_state_dict(initial_state, strict=True)

        restored_state = fresh_module.state_dict()
        self.assertEqual(set(initial_state.keys()), set(restored_state.keys()))

        for key in initial_state:
            torch.testing.assert_close(
                initial_state[key],
                restored_state[key],
                msg=f"State mismatch for key {key}",
            )

    def test_rec_metric_module_backward_compat_trained_batches(self) -> None:
        module = _make_rec_metric_module()
        state_dict = module.state_dict()

        state_dict["_trained_batches"] = torch.tensor(100)

        fresh_module = _make_rec_metric_module()
        # strict=True matches production. Under strict=False this test passes
        # even without the pop hook.
        fresh_module.load_state_dict(state_dict, strict=True)


class SchemaStableCoverageTest(unittest.TestCase):
    """The decorator registry and the case table must list the same ids.

    A class decorated but absent from the table has no fixture and no golden
    entry. An id in the table that nothing registers is covered by accident
    rather than by declaration.

    Neither can see a decorated class in a module that was never imported;
    registration is import-bound. Adding a case forces the Buck dep, which is
    what makes the class visible in the first place.
    """

    def test_registry_matches_case_table(self) -> None:
        # Comparing the id sets alone would pass if two ids were swapped between
        # classes, so compare the id-to-class mapping. Built by hand because a
        # comprehension lets a second case under one id overwrite the first, and
        # the duplicate-key check in the generator only sees id plus variant.
        declared: Dict[str, Type[torch.nn.Module]] = {}
        for case in _SCHEMA_CASES:
            first = declared.setdefault(case.stable_id, case.expected_class)
            self.assertIs(
                first,
                case.expected_class,
                f"stable_id {case.stable_id!r} is claimed by two classes: "
                f"{first.__name__} and {case.expected_class.__name__}.",
            )
        self.assertEqual(registered_schema_stable_classes(), declared)


class SchemaChangeTest(unittest.TestCase):
    """Regenerating must not quietly bless a changed key set.

    The comparison already fails on an added or removed key. The hole was the
    escape hatch: --update-golden rewrote the file unconditionally, and the
    failing test names that command, so silencing a real break took one step.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden_snapshot = load_golden_snapshot()

    def test_unauthorized_addition_is_reported(self) -> None:
        old = {"Thing": {"state_dict_keys": ["a"]}}
        new = {"Thing": {"state_dict_keys": ["a", "b"]}}
        self.assertEqual(unauthorized_additions(old, new), {"Thing": ["b"]})

    def test_authorized_addition_is_allowed(self) -> None:
        old = {"Thing": {"state_dict_keys": ["a"]}}
        new = {"Thing": {"state_dict_keys": ["a", "b"]}}
        self.assertEqual(
            unauthorized_additions(old, new, (_make_record("Thing", "b"),)), {}
        )

    def test_a_brand_new_entry_needs_no_authorization(self) -> None:
        old: Dict[str, Dict[str, Any]] = {}
        new = {"Thing": {"state_dict_keys": ["a", "b"]}}
        self.assertEqual(unauthorized_additions(old, new), {})

    def test_a_dropped_entry_is_reported(self) -> None:
        old = {"Thing": {"state_dict_keys": ["a"]}}
        new: Dict[str, Dict[str, Any]] = {}
        self.assertEqual(removed_entries(old, new), ["Thing"])

    def test_a_rename_that_adds_a_key_cannot_hide(self) -> None:
        """The gap the addition check alone leaves open.

        Renaming the class moves the entry to a new key, so the added state key
        rides in under the brand-new exemption. Only the dropped old entry
        gives it away.
        """
        old = {"Thing": {"state_dict_keys": ["a"]}}
        new = {"RenamedThing": {"state_dict_keys": ["a", "b"]}}
        self.assertEqual(unauthorized_additions(old, new), {})
        self.assertEqual(removed_entries(old, new), ["Thing"])

    def test_unauthorized_removal_is_reported(self) -> None:
        old = {"Thing": {"state_dict_keys": ["a", "b"]}}
        new = {"Thing": {"state_dict_keys": ["a"]}}
        self.assertEqual(unauthorized_removals(old, new), {"Thing": ["b"]})

    def test_authorized_removal_is_allowed(self) -> None:
        old = {"Thing": {"state_dict_keys": ["a", "b"]}}
        new = {"Thing": {"state_dict_keys": ["a"]}}
        self.assertEqual(
            unauthorized_removals(old, new, (_make_removal("Thing", "b"),)), {}
        )

    def test_a_dropped_entry_is_left_to_removed_entries(self) -> None:
        """Otherwise a rename reports every key twice, once per gate."""
        old = {"Thing": {"state_dict_keys": ["a", "b"]}}
        new: Dict[str, Dict[str, Any]] = {}
        self.assertEqual(unauthorized_removals(old, new), {})
        self.assertEqual(removed_entries(old, new), ["Thing"])

    def test_configured_records_are_not_stale(self) -> None:
        """Both registries checked in one place, so neither can run empty.

        Looping over the registries meant zero assertions while they are empty,
        which reads as coverage. The detectors themselves are pinned by the
        synthetic tests below.
        """
        self.assertEqual(stale_records(_SCHEMA_ADDITIONS, self.golden_snapshot), [])
        self.assertEqual(
            stale_removal_records(_SCHEMA_REMOVALS, self.golden_snapshot), []
        )

    def test_stale_removal_records_are_detected(self) -> None:
        # Staleness inverts: the key must be gone, not present.
        snapshot = {"Thing": {"state_dict_keys": ["a"]}}
        self.assertEqual(
            stale_removal_records((_make_removal("Thing", "b"),), snapshot), []
        )
        self.assertEqual(
            stale_removal_records((_make_removal("Thing", "a"),), snapshot),
            ["Thing still has key 'a'"],
        )
        self.assertEqual(
            stale_removal_records((_make_removal("Gone", "a"),), snapshot),
            ["Gone is not in the golden"],
        )

    def test_removal_record_fields_are_required(self) -> None:
        for blank in ("", "   "):
            with self.subTest(repr(blank)), self.assertRaises(ValueError):
                _make_removal("Thing", "b", hook=blank)

    def test_stale_records_are_detected(self) -> None:
        # _SCHEMA_ADDITIONS is empty, so the check above passes without running
        # anything. These pin the detector itself.
        snapshot = {"Thing": {"state_dict_keys": ["a"]}}
        self.assertEqual(stale_records((_make_record("Thing", "a"),), snapshot), [])
        self.assertEqual(
            stale_records((_make_record("Gone", "a"),), snapshot),
            ["Gone is not in the golden"],
        )
        self.assertEqual(
            stale_records((_make_record("Thing", "b"),), snapshot),
            ["Thing has no key 'b'"],
        )

    def test_record_fields_are_required(self) -> None:
        for blank in ("", "   "):
            with self.subTest(repr(blank)), self.assertRaises(ValueError):
                _make_record("Thing", "b", reason=blank)

    def test_record_added_must_be_a_date(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a datetime.date"):
            # pyre-ignore[6]
            _make_record("Thing", "b", added="2026-01-01")


class GoldenRegenerationTest(unittest.TestCase):
    """The gates on the real generate-validate-save path.

    The tests above call the comparison helpers directly, which says nothing
    about whether `update_golden_snapshot` consults them or writes anyway. Each
    case here doctors a baseline, runs the whole path against a temporary file,
    and requires that file to come back untouched.

    The patch redirects where the snapshot is read and written. It replaces a
    path, not a dependency, so the file I/O under test is real.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Building every case is the slow part, so pay it once and reuse the
        # result as the baseline each case then doctors.
        with _single_rank_process_group():
            cls.current: Dict[str, Dict[str, Any]] = generate_golden_snapshot()
            cls.current.update(generate_schema_case_entries())

    @contextlib.contextmanager
    def _baseline(self, snapshot: Dict[str, Dict[str, Any]]) -> Iterator[Path]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.json"
            path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
            with unittest.mock.patch(
                f"{__name__}.GOLDEN_SNAPSHOT_PATH", path
            ), _single_rank_process_group():
                yield path

    def _assert_refused(self, baseline: Dict[str, Dict[str, Any]], reason: str) -> None:
        with self._baseline(baseline) as path:
            before = path.read_bytes()
            with self.assertRaises((ValueError, RuntimeError)) as cm:
                update_golden_snapshot()
            self.assertIn(reason, str(cm.exception))
            self.assertEqual(path.read_bytes(), before, "the golden was rewritten")

    def _entry_holding_keys(self, baseline: Dict[str, Dict[str, Any]]) -> str:
        """An entry with at least one state key.

        Six entries legitimately have none, the AUC family among them, and they
        sort first. Doctoring one of those changes nothing and the gate has
        nothing to catch.
        """
        for key in sorted(baseline):
            if baseline[key]["state_dict_keys"]:
                return key
        self.fail("no golden entry holds a state key")

    def test_an_empty_baseline_is_refused(self) -> None:
        self._assert_refused({}, "restore it from source control")

    def test_an_unauthorized_addition_is_refused(self) -> None:
        # Drop one key from one entry, so the live code looks like it added it.
        baseline = copy.deepcopy(type(self).current)
        key = self._entry_holding_keys(baseline)
        baseline[key]["state_dict_keys"] = baseline[key]["state_dict_keys"][1:]
        self._assert_refused(baseline, "add state_dict keys that nothing authorizes")

    def test_an_unauthorized_removal_is_refused(self) -> None:
        baseline = copy.deepcopy(type(self).current)
        key = self._entry_holding_keys(baseline)
        baseline[key]["state_dict_keys"] = baseline[key]["state_dict_keys"] + [
            "a_key_the_code_no_longer_produces"
        ]
        self._assert_refused(baseline, "drop state_dict keys that nothing authorizes")

    def test_a_dropped_entry_is_refused(self) -> None:
        baseline = copy.deepcopy(type(self).current)
        baseline["AnEntryTheCodeNoLongerProduces"] = {
            "metric_class": "Gone",
            "variant": "",
            "state_dict_keys": [],
            "persistent_buffer_fqns": [],
            "non_persistent_buffer_fqns": [],
        }
        self._assert_refused(baseline, "remove golden entries")

    def test_an_unchanged_baseline_is_rewritten_identically(self) -> None:
        with self._baseline(type(self).current) as path:
            before = path.read_bytes()
            update_golden_snapshot()
            self.assertEqual(path.read_bytes(), before)


class MetricCoverageTest(unittest.TestCase):
    """
    Test that ensures all RecMetric subclasses are covered by backward compatibility tests.

    This test will FAIL if a new metric is added to torchrec but not added to METRICS_TO_TEST.
    When adding a new metric, users must add it to METRICS_TO_TEST in this file.
    """

    # Metrics that are intentionally excluded from testing (with reason)
    EXCLUDED_METRICS: Dict[str, str] = {
        # Add metrics here that should be excluded, with a reason
        # e.g., "torchrec.metrics.foo.SomeMetric": "deprecated, removed next release",
    }

    def test_rows_produce_distinct_keys(self) -> None:
        """No two rows may claim one golden entry.

        The key drops kwargs, so two rows differing only in configuration
        collide. Generation refuses to write that, but this names the offending
        rows without making anyone run --update-golden to find out.
        """
        keys = [
            get_metric_snapshot_key(metric_class, compute_mode, variant)
            for metric_class, compute_modes, _kwargs, variants in METRICS_TO_TEST
            for compute_mode in compute_modes
            for variant in variants
        ]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        self.assertEqual(
            duplicates, [], f"METRICS_TO_TEST rows collide on {duplicates}"
        )

    @unittest.skipIf(
        sys.version_info < (3, 11),
        "concurrent.futures._base.Future is type but not a class",
    )
    def test_all_recmetrics_are_covered(self) -> None:
        discovered_metrics: Set[str] = {
            cls.__name__ for cls in _discover_all_recmetric_subclasses()
        }

        covered_metrics: Set[str] = {
            metric_class.__name__ for metric_class, _, _, _ in METRICS_TO_TEST
        }

        missing_metrics = (
            discovered_metrics - covered_metrics - set(self.EXCLUDED_METRICS.keys())
        )

        if missing_metrics:
            self.fail(
                f"The following RecMetric subclasses are not covered by backward "
                f"compatibility tests: {sorted(missing_metrics)}.\n\n"
                f"To fix this:\n"
                f"1. Add a METRICS_TO_TEST row. The check runs from the table, "
                f"so the row is the whole registration.\n"
                f"2. Run with --update-golden to write its entry.\n\n"
                f"If the metric should be excluded, add it to EXCLUDED_METRICS with a reason."
            )


# Cross-config state_dict tests: detect config-dependent keys and verify
# cross-config loads succeed with strict=True. New conditional-state params
# must be added to KNOWN_CONDITIONAL_STATE with a proper always-pop hook.

_BATCH_SIZE_STAGES_ALTERNATIVE: List[BatchSizeStage] = [
    BatchSizeStage(batch_size=256, max_iters=1),
    BatchSizeStage(batch_size=512, max_iters=None),
]

_BASE_RECMETRIC_PARAMS: Set[str] = {
    "self",
    "args",
    "kwargs",
    "world_size",
    "my_rank",
    "batch_size",
    "tasks",
    "compute_mode",
    "window_size",
    "fused_update_limit",
    "compute_on_all_ranks",
    "should_validate_update",
    "process_group",
    "enable_pt2_compile",
    "should_clone_update_inputs",
}

_BASE_COMPUTATION_PARAMS: Set[str] = {
    "self",
    "args",
    "kwargs",
    "my_rank",
    "batch_size",
    "n_tasks",
    "tasks",
    "window_size",
    "compute_on_all_ranks",
    "should_validate_update",
    "compute_mode",
    "process_group",
    "fused_update_limit",
    "allow_missing_label_with_zero_weight",
    "session_metric_def",
}

_PARAM_ALTERNATIVES: Dict[str, List[Any]] = {
    "batch_size_stages": [_BATCH_SIZE_STAGES_ALTERNATIVE],
    "description": ["test_description"],
    "is_negative_task_mask": [[True]],
    "label_names": [["label_a", "label_b"]],
}


def _get_metric_specific_params(
    metric_cls: Type[RecMetric],
) -> Dict[str, inspect.Parameter]:
    """Get non-base params from metric and computation class signatures."""
    params: Dict[str, inspect.Parameter] = {}

    for name, param in inspect.signature(metric_cls.__init__).parameters.items():
        if name not in _BASE_RECMETRIC_PARAMS:
            params[name] = param

    comp_cls = getattr(metric_cls, "_computation_class", None)
    if comp_cls is not None:
        for name, param in inspect.signature(comp_cls.__init__).parameters.items():
            if name not in _BASE_COMPUTATION_PARAMS and name not in params:
                params[name] = param

    return params


def _generate_alternatives(
    param: inspect.Parameter,
) -> List[Any]:
    """Auto-generate alternative values for a param based on its default."""
    name = param.name
    default = param.default

    if name in _PARAM_ALTERNATIVES:
        return _PARAM_ALTERNATIVES[name]

    if default is inspect.Parameter.empty:
        return []

    if isinstance(default, bool):
        return [not default]
    if isinstance(default, int):
        return [default + 1]
    if isinstance(default, float):
        return [default + 1.0]
    if isinstance(default, str):
        return [default + "_alt"]
    return []


# Params that produce different state_dict keys. Uses strings (includes ThroughputMetric/nn.Module).
KNOWN_CONDITIONAL_STATE: Set[Tuple[str, str]] = {
    ("MSEMetric", "include_r_squared"),
    ("MultiLabelPrecisionMetric", "label_names"),
    ("MultiLabelPrecisionMetric", "num_labels"),
    ("ThroughputMetric", "batch_size_stages"),
    ("TowerQPSMetric", "batch_size_stages"),
}

# Params that do NOT affect state_dict keys. Every non-base param must be here
# or in KNOWN_CONDITIONAL_STATE.
KNOWN_SAFE_PARAMS: Set[Tuple[str, str]] = {
    ("AUCMetric", "apply_bin"),
    ("AUCMetric", "grouped_auc"),
    ("AUPRCMetric", "grouped_auprc"),
    ("AUPRCMetric", "max_prediction"),
    ("AUPRCMetric", "min_prediction"),
    ("AUPRCMetric", "num_bins"),
    ("AccuracyMetric", "threshold"),
    ("HindsightTargetPRMetric", "target_precision"),
    ("NDCGMetric", "exponential_gain"),
    ("NDCGMetric", "is_negative_task_mask"),
    ("NDCGMetric", "k"),
    ("NDCGMetric", "remove_single_length_sessions"),
    ("NDCGMetric", "report_ndcg_as_decreasing_curve"),
    ("NDCGMetric", "scale_by_weights_tensor"),
    ("NDCGMetric", "session_key"),
    ("NEMetric", "include_logloss"),
    ("PrecisionMetric", "threshold"),
    ("RAUCMetric", "grouped_rauc"),
    ("RecalibratedCalibrationMetric", "recalibration_coefficient"),
    ("RecalibratedNEMetric", "include_logloss"),
    ("RecalibratedNEMetric", "recalibration_coefficient"),
    ("RecallMetric", "threshold"),
    ("SegmentedNEMetric", "cast_keys_to_int"),
    ("SegmentedNEMetric", "grouping_keys"),
    ("SegmentedNEMetric", "include_logloss"),
    ("SegmentedNEMetric", "num_groups"),  # changes tensor shapes, not key names
    ("TensorWeightedAvgMetric", "description"),
    ("TowerQPSMetric", "warmup_steps"),
}

# Subset with proper always-pop hooks (cross-config load tests use these).
RECMETRIC_CONDITIONAL_STATE: Dict[Tuple[Type[RecMetric], str], List[Any]] = {
    (MSEMetric, "include_r_squared"): [True],
    (TowerQPSMetric, "batch_size_stages"): [_BATCH_SIZE_STAGES_ALTERNATIVE],
}

_RECMETRIC_COMMON_KWARGS: Dict[str, Any] = {
    "world_size": 1,
    "my_rank": 0,
    "batch_size": 32,
    "compute_mode": RecComputeMode.UNFUSED_TASKS_COMPUTATION,
    "window_size": 100,
}

_THROUGHPUT_COMMON_KWARGS: Dict[str, Any] = {
    "batch_size": 32,
    "world_size": 1,
    "window_seconds": 100,
    "warmup_steps": 10,
}


_cached_recmetric_subclasses: Optional[Set[Type[RecMetric]]] = None


def _discover_all_recmetric_subclasses() -> Set[Type[RecMetric]]:
    """Discover all RecMetric subclasses in torchrec.metrics."""
    global _cached_recmetric_subclasses
    if _cached_recmetric_subclasses is not None:
        return _cached_recmetric_subclasses

    import importlib
    import pkgutil

    import torchrec.metrics

    subclasses: Set[Type[RecMetric]] = set()
    for _, module_name, _ in pkgutil.iter_modules(torchrec.metrics.__path__):
        try:
            module = importlib.import_module(f"torchrec.metrics.{module_name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, RecMetric)
                    and attr is not RecMetric
                    and not attr_name.startswith("_")
                ):
                    subclasses.add(attr)
        except ImportError:
            continue
    _cached_recmetric_subclasses = subclasses
    return subclasses


def _get_default_keys_cached(
    metric_cls: Type[RecMetric],
    cls_name: str,
    default_keys_cache: Optional[Dict[str, Set[str]]] = None,
) -> Optional[Set[str]]:
    """Get default state_dict keys for a metric, using cache if available."""
    if default_keys_cache is not None and cls_name in default_keys_cache:
        return default_keys_cache[cls_name]
    try:
        default_keys = set(extract_state_dict_keys(metric_cls))
    except (TypeError, ValueError, KeyError, RecMetricException):
        return None
    if default_keys_cache is not None:
        default_keys_cache[cls_name] = default_keys
    return default_keys


def _probe_alternatives(
    metric_cls: Type[RecMetric],
    param_name: str,
    alternatives: List[Any],
    default_keys: Set[str],
) -> Optional[str]:
    """Probe alternative param values. Returns 'misclassified', 'unprobed', or None."""
    tested_any = False
    for alt_value in alternatives:
        try:
            variant_keys = set(
                extract_state_dict_keys(metric_cls, **{param_name: alt_value})
            )
            tested_any = True
        except (TypeError, ValueError, KeyError, RecMetricException):
            continue
        if default_keys != variant_keys:
            return "misclassified"
    if not tested_any:
        return "unprobed"
    return None


def _classify_known_safe_param(
    cls_name: str,
    param_name: str,
    metrics_by_name: Dict[str, Type[RecMetric]],
    default_keys_cache: Optional[Dict[str, Set[str]]] = None,
) -> Optional[str]:
    """Classify a KNOWN_SAFE_PARAMS entry. Returns category or None if verified safe."""
    metric_cls = metrics_by_name.get(cls_name)
    if metric_cls is None:
        return "stale"
    params = _get_metric_specific_params(metric_cls)
    if param_name not in params:
        return "stale"
    alternatives = _generate_alternatives(params[param_name])
    if not alternatives:
        return "unprobed"
    default_keys = _get_default_keys_cached(metric_cls, cls_name, default_keys_cache)
    if default_keys is None:
        return "unprobed"
    return _probe_alternatives(metric_cls, param_name, alternatives, default_keys)


class ConditionalStateRegistryTest(unittest.TestCase):

    @unittest.skipIf(
        sys.version_info < (3, 11),
        "concurrent.futures._base.Future is type but not a class",
    )
    def test_validate_state_affecting_params_recmetrics(self) -> None:
        all_metrics = _discover_all_recmetric_subclasses()
        for metric_cls in all_metrics:
            try:
                default_keys = set(extract_state_dict_keys(metric_cls))
            except (TypeError, ValueError, KeyError, RecMetricException):
                continue
            for param_name, param in _get_metric_specific_params(metric_cls).items():
                alternatives = _generate_alternatives(param)
                if not alternatives:
                    continue
                for alt_value in alternatives:
                    with self.subTest(metric=metric_cls.__name__, param=param_name):
                        try:
                            variant_keys = set(
                                extract_state_dict_keys(
                                    metric_cls, **{param_name: alt_value}
                                )
                            )
                        except (TypeError, ValueError, KeyError, RecMetricException):
                            continue

                        if default_keys != variant_keys:
                            self.assertIn(
                                (metric_cls.__name__, param_name),
                                KNOWN_CONDITIONAL_STATE,
                                f"{metric_cls.__name__}.{param_name} affects "
                                f"state_dict keys but is not in "
                                f"KNOWN_CONDITIONAL_STATE. "
                                f"Added: {variant_keys - default_keys}, "
                                f"Removed: {default_keys - variant_keys}",
                            )

    @unittest.skipIf(
        sys.version_info < (3, 11),
        "concurrent.futures._base.Future is type but not a class",
    )
    def test_all_params_categorized(self) -> None:
        all_metrics = _discover_all_recmetric_subclasses()
        uncategorized = []
        for metric_cls in all_metrics:
            for param_name in _get_metric_specific_params(metric_cls):
                pair = (metric_cls.__name__, param_name)
                if (
                    pair not in KNOWN_CONDITIONAL_STATE
                    and pair not in KNOWN_SAFE_PARAMS
                ):
                    uncategorized.append(pair)

        if uncategorized:
            formatted = "\n".join(
                f'    ("{cls}", "{param}"),' for cls, param in sorted(uncategorized)
            )
            self.fail(
                f"Found {len(uncategorized)} uncategorized metric param(s).\n"
                f"Each param must be in KNOWN_CONDITIONAL_STATE (if it conditionally\n"
                f"registers buffers/state) or KNOWN_SAFE_PARAMS (if it does not).\n"
                f"Add these to the appropriate set:\n{formatted}"
            )

    @unittest.skipIf(
        sys.version_info < (3, 11),
        "concurrent.futures._base.Future is type but not a class",
    )
    def test_none_default_params_have_test_values(self) -> None:
        all_metrics = _discover_all_recmetric_subclasses()
        missing = []
        for metric_cls in all_metrics:
            for param_name, param in _get_metric_specific_params(metric_cls).items():
                if param.default is None and param_name not in _PARAM_ALTERNATIVES:
                    missing.append((metric_cls.__name__, param_name))

        if missing:
            formatted = "\n".join(
                f'    "{p}",' for _, p in sorted(set(missing), key=lambda x: x[1])
            )
            self.fail(
                f"Found None-default params without _PARAM_ALTERNATIVES entries.\n"
                f"These params can't be auto-probed for conditional state.\n"
                f"Add test values to _PARAM_ALTERNATIVES for:\n{formatted}"
            )

    @unittest.skipIf(
        sys.version_info < (3, 11),
        "concurrent.futures._base.Future is type but not a class",
    )
    def test_known_safe_params_are_actually_safe(self) -> None:
        all_metrics = _discover_all_recmetric_subclasses()
        metrics_by_name = {cls.__name__: cls for cls in all_metrics}
        default_keys_cache: Dict[str, Set[str]] = {}
        buckets: Dict[str, List[Tuple[str, str]]] = {
            "stale": [],
            "misclassified": [],
            "unprobed": [],
        }
        for cls_name, param_name in sorted(KNOWN_SAFE_PARAMS):
            category = _classify_known_safe_param(
                cls_name, param_name, metrics_by_name, default_keys_cache
            )
            if category is not None:
                buckets[category].append((cls_name, param_name))

        error_messages = {
            "stale": "Stale entries (param not in any signature)",
            "misclassified": (
                "Misclassified (actually affects state_dict keys, "
                "move to KNOWN_CONDITIONAL_STATE)"
            ),
        }
        errors = []
        for key, label in error_messages.items():
            if buckets[key]:
                formatted = "\n".join(f'    ("{c}", "{p}"),' for c, p in buckets[key])
                errors.append(f"{label}:\n{formatted}")
        if errors:
            self.fail("KNOWN_SAFE_PARAMS issues:\n" + "\n\n".join(errors))

    def test_validate_state_affecting_params_throughput(self) -> None:
        default_metric = ThroughputMetric(**_THROUGHPUT_COMMON_KWARGS)
        variant_metric = ThroughputMetric(
            **_THROUGHPUT_COMMON_KWARGS,
            batch_size_stages=_BATCH_SIZE_STAGES_ALTERNATIVE,
        )
        default_keys = set(default_metric.state_dict().keys())
        variant_keys = set(variant_metric.state_dict().keys())

        if default_keys != variant_keys:
            self.assertIn(
                ("ThroughputMetric", "batch_size_stages"),
                KNOWN_CONDITIONAL_STATE,
                f"ThroughputMetric.batch_size_stages affects state_dict "
                f"keys but is not in KNOWN_CONDITIONAL_STATE. "
                f"Added keys: {variant_keys - default_keys}, "
                f"Removed keys: {default_keys - variant_keys}",
            )


class CrossConfigLoadTest(unittest.TestCase):

    def _make_common_kwargs(self) -> Dict[str, Any]:
        return {**_RECMETRIC_COMMON_KWARGS, "tasks": [create_test_task("task1")]}

    def _assert_cross_config_load(
        self,
        metric_cls: Type[RecMetric],
        param_name: str,
        alt_value: Any,
        direction: str,
    ) -> None:
        common_kwargs = self._make_common_kwargs()
        if direction == "variant_to_default":
            src = metric_cls(**common_kwargs, **{param_name: alt_value})
            dst = metric_cls(**common_kwargs)
        else:
            src = metric_cls(**common_kwargs)
            dst = metric_cls(**common_kwargs, **{param_name: alt_value})
        dst.load_state_dict(src.state_dict(), strict=True)

    def test_cross_config_load_variant_to_default(self) -> None:
        for (
            metric_cls,
            param_name,
        ), alternatives in RECMETRIC_CONDITIONAL_STATE.items():
            for alt_value in alternatives:
                with self.subTest(
                    metric=metric_cls.__name__,
                    param=param_name,
                    direction="variant_to_default",
                ):
                    self._assert_cross_config_load(
                        metric_cls, param_name, alt_value, "variant_to_default"
                    )

    def test_cross_config_load_default_to_variant(self) -> None:
        for (
            metric_cls,
            param_name,
        ), alternatives in RECMETRIC_CONDITIONAL_STATE.items():
            for alt_value in alternatives:
                with self.subTest(
                    metric=metric_cls.__name__,
                    param=param_name,
                    direction="default_to_variant",
                ):
                    self._assert_cross_config_load(
                        metric_cls, param_name, alt_value, "default_to_variant"
                    )

    def test_throughput_cross_config_load_variant_to_default(self) -> None:
        variant = ThroughputMetric(
            **_THROUGHPUT_COMMON_KWARGS,
            batch_size_stages=_BATCH_SIZE_STAGES_ALTERNATIVE,
        )
        default = ThroughputMetric(**_THROUGHPUT_COMMON_KWARGS)
        default.load_state_dict(variant.state_dict(), strict=True)

    def test_throughput_cross_config_load_default_to_variant(self) -> None:
        default = ThroughputMetric(**_THROUGHPUT_COMMON_KWARGS)
        variant = ThroughputMetric(
            **_THROUGHPUT_COMMON_KWARGS,
            batch_size_stages=_BATCH_SIZE_STAGES_ALTERNATIVE,
        )
        variant.load_state_dict(default.state_dict(), strict=True)

    def test_multi_label_precision_cross_config_load_fails_without_hook(self) -> None:
        variant_kwargs: Dict[str, Any] = {
            **self._make_common_kwargs(),
            "num_labels": 3,
        }
        default_kwargs: Dict[str, Any] = {
            **self._make_common_kwargs(),
            "num_labels": 1,
        }
        variant = MultiLabelPrecisionMetric(**variant_kwargs)
        default = MultiLabelPrecisionMetric(**default_kwargs)

        with self.assertRaises(RuntimeError):
            default.load_state_dict(variant.state_dict(), strict=True)

    def test_multi_label_precision_label_names_cross_config_load_fails(self) -> None:
        kwargs_a: Dict[str, Any] = {
            **self._make_common_kwargs(),
            "num_labels": 1,
            "label_names": ["cat"],
        }
        kwargs_b: Dict[str, Any] = {
            **self._make_common_kwargs(),
            "num_labels": 1,
            "label_names": ["dog"],
        }
        variant_a = MultiLabelPrecisionMetric(**kwargs_a)
        variant_b = MultiLabelPrecisionMetric(**kwargs_b)

        with self.assertRaises(RuntimeError):
            variant_b.load_state_dict(variant_a.state_dict(), strict=True)


class DCPCheckpointClientSimulationTest(unittest.TestCase):
    """
    Test suite that simulates DCP CheckpointClient FQN validation behavior.

    This test explicitly simulates what happens at the DCP level when loading
    checkpoints. The DCP CheckpointClient (in aiplatform/modelstore) validates
    that ALL FQNs in the model's state_dict exist in the checkpoint metadata
    BEFORE calling load_state_dict.

    This is different from PyTorch's load_state_dict behavior:
    - PyTorch: Fails if checkpoint has keys model doesn't expect (with strict=True)
    - DCP: Fails if MODEL has keys that CHECKPOINT doesn't have

    When a new buffer is added to a metric:
    - PyTorch load_state_dict: May work (torchmetrics handles missing keys)
    - DCP CheckpointClient: FAILS with InvalidParamQualNameException

    See: aiplatform/modelstore/experimental/DCP/planners/planner_utils.py
    Function: is_loading_param_fqn_in_cp_metadata()
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.golden_snapshot = load_required_golden_snapshot()

    def _simulate_dcp_fqn_validation(
        self,
        model_fqns: Set[str],
        checkpoint_fqns: Set[str],
        allow_partial_load: bool = False,
    ) -> Tuple[bool, Set[str]]:
        """
        Simulate DCP CheckpointClient FQN validation.

        This mirrors the logic in is_loading_param_fqn_in_cp_metadata():
        - For each FQN in the model's state_dict
        - Check if it exists in the checkpoint metadata
        - If not, and allow_partial_load=False, raise InvalidParamQualNameException

        Args:
            model_fqns: FQNs from the current model's state_dict
            checkpoint_fqns: FQNs from the saved checkpoint (golden snapshot)
            allow_partial_load: If True, skip validation for missing FQNs

        Returns:
            Tuple of (would_succeed, missing_fqns)
        """
        missing_in_checkpoint = model_fqns - checkpoint_fqns

        if allow_partial_load:
            # DCP logs a warning but continues
            return True, missing_in_checkpoint

        # DCP raises InvalidParamQualNameException
        would_succeed = len(missing_in_checkpoint) == 0
        return would_succeed, missing_in_checkpoint

    def test_ne_metric_dcp_validation(self) -> None:
        key = get_metric_snapshot_key(
            NEMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION
        )
        if key not in self.golden_snapshot:
            self.fail(
                f"No golden entry for {key}. This simulation needs one; run "
                "with --update-golden."
            )

        current_fqns = set(
            extract_state_dict_keys(NEMetric, RecComputeMode.UNFUSED_TASKS_COMPUTATION)
        )
        checkpoint_fqns = set(self.golden_snapshot[key]["state_dict_keys"])

        would_succeed, missing = self._simulate_dcp_fqn_validation(
            current_fqns, checkpoint_fqns, allow_partial_load=False
        )

        if not would_succeed:
            self.fail(
                f"DCP CheckpointClient would fail for NEMetric!\n"
                f"Model has FQNs not in checkpoint: {sorted(missing)}\n"
                f"This simulates InvalidParamQualNameException at load time."
            )

    def test_rec_metric_module_dcp_validation(self) -> None:
        key = _REC_METRIC_MODULE_DEFAULT.key
        if key not in self.golden_snapshot:
            self.fail(
                f"No golden entry for {key}. This simulation needs one; run "
                "with --update-golden."
            )

        tasks = [create_test_task("task1")]
        ne_metric = NEMetric(
            world_size=1,
            my_rank=0,
            batch_size=32,
            tasks=tasks,
            compute_mode=RecComputeMode.UNFUSED_TASKS_COMPUTATION,
            window_size=100,
        )
        rec_metrics = RecMetricList([ne_metric])

        module = RecMetricModule(
            batch_size=32,
            world_size=1,
            rec_tasks=tasks,
            rec_metrics=rec_metrics,
        )

        current_fqns = set(module.state_dict().keys())
        checkpoint_fqns = set(self.golden_snapshot[key]["state_dict_keys"])

        would_succeed, missing = self._simulate_dcp_fqn_validation(
            current_fqns, checkpoint_fqns, allow_partial_load=False
        )

        if not would_succeed:
            self.fail(
                f"DCP CheckpointClient would fail for RecMetricModule!\n"
                f"Model has FQNs not in checkpoint: {sorted(missing)}\n"
                f"This simulates InvalidParamQualNameException at load time.\n\n"
                f"To fix: Update golden snapshot with --update-golden after "
                f"coordinating with trainer teams."
            )


def generate_schema_case_entries() -> Dict[str, Dict[str, Any]]:
    """Golden entries for the _SCHEMA_CASES table.

    The comparison path fails rather than writing a missing entry, so this is
    the only way a new case gets one.

    Owns the process group rather than leaving it to the caller: the tests get
    one from setUp, and --update-golden runs outside unittest entirely.
    """
    entries: Dict[str, Dict[str, Any]] = {}
    with _single_rank_process_group():
        for case in _SCHEMA_CASES:
            if case.key in entries:
                raise ValueError(
                    f"Two cases share the key {case.key!r}. One would overwrite "
                    "the other's golden entry."
                )
            module = case.build()
            try:
                entries[case.key] = {
                    "metric_class": case.expected_class.__name__,
                    "variant": case.variant,
                    "state_dict_keys": sorted(module.state_dict().keys()),
                    "persistent_buffer_fqns": [],
                    "non_persistent_buffer_fqns": [],
                }
            finally:
                if case.cleanup is not None:
                    case.cleanup(module)
    return entries


@dataclass(frozen=True)
class _SchemaChange:
    """What every deliberate change to a golden entry has to say.

    Neither record changes behavior. Both are review gates and a paper trail,
    and the subclass field naming the migration is the one that matters.
    """

    golden_key: str
    state_key: str
    reason: str
    owner: str

    def _validate(self, extra_text: Tuple[str, ...], date_field: str) -> None:
        for name in ("golden_key", "state_key", "reason", "owner") + extra_text:
            value = getattr(self, name)
            if not value or not value.strip():
                raise ValueError(
                    f"{type(self).__name__}.{name} must be non-empty "
                    f"({self.state_key!r} on {self.golden_key!r})"
                )
        # The annotation alone does not stop a string, and a string date sorts
        # and compares wrong without ever raising.
        when = getattr(self, date_field)
        if not isinstance(when, datetime.date):
            raise ValueError(
                f"{type(self).__name__}.{date_field} must be a datetime.date, "
                f"got {type(when).__name__}"
            )


@dataclass(frozen=True)
class _SchemaAddition(_SchemaChange):
    """An intentional new state_dict key on a class that already has a golden entry.

    Additions are the direction a module hook cannot repair. DCP validates that
    every model FQN exists in checkpoint metadata before load_state_dict runs,
    so a checkpoint written before the key existed is rejected by the planner,
    ahead of any hook. Adding one persistent buffer to RecMetricModule has
    already taken down production training this way.

    `rollout` names how checkpoints that predate the key are handled, usually
    LoadOverride.allow_missing_fqns.
    """

    rollout: str
    added: datetime.date

    def __post_init__(self) -> None:
        self._validate(("rollout",), "added")


@dataclass(frozen=True)
class _SchemaRemoval(_SchemaChange):
    """A state_dict key dropped from a class that keeps its golden entry.

    Removals break a narrower set of loads than additions. torchmetrics pops
    the keys it still declares, so a key it no longer declares reaches
    nn.Module's strict check unclaimed and raises. DCP and the legacy client
    build the load dict from the model, so the leftover key is never requested
    there. Only a checkpoint-derived load reads it back, which today means
    transfer learning.

    `hook` names what pops the key for checkpoints that still carry it.
    """

    hook: str
    removed: datetime.date

    def __post_init__(self) -> None:
        self._validate(("hook",), "removed")


# Every deliberate addition to an existing golden entry. Regenerating the
# snapshot refuses to write an added key that is not listed here.
_SCHEMA_ADDITIONS: Tuple[_SchemaAddition, ...] = ()

# The same, for keys dropped from an entry that survives.
_SCHEMA_REMOVALS: Tuple[_SchemaRemoval, ...] = ()


def _make_record(golden_key: str, state_key: str, **overrides: Any) -> _SchemaAddition:
    """A filled-in record, so tests can vary the one field they care about."""
    fields: Dict[str, Any] = {
        "golden_key": golden_key,
        "state_key": state_key,
        "reason": "test",
        "rollout": "test",
        "owner": "test",
        "added": datetime.date(2026, 1, 1),
    }
    fields.update(overrides)
    return _SchemaAddition(**fields)


def _make_removal(golden_key: str, state_key: str, **overrides: Any) -> _SchemaRemoval:
    """The removal-side twin of _make_record."""
    fields: Dict[str, Any] = {
        "golden_key": golden_key,
        "state_key": state_key,
        "reason": "test",
        "hook": "test",
        "owner": "test",
        "removed": datetime.date(2026, 1, 1),
    }
    fields.update(overrides)
    return _SchemaRemoval(**fields)


def _unclaimed_delta(
    source: Dict[str, Dict[str, Any]],
    target: Dict[str, Dict[str, Any]],
    records: Sequence[_SchemaChange],
) -> Dict[str, List[str]]:
    """For entries both snapshots hold, keys in `source` absent from `target`.

    Keys a record claims are dropped from the result. Entries missing from
    either side are skipped, so a brand-new or dropped entry is somebody else's
    problem.
    """
    claimed: Dict[str, Set[str]] = {}
    for record in records:
        claimed.setdefault(record.golden_key, set()).add(record.state_key)

    offenders: Dict[str, List[str]] = {}
    for golden_key, entry in source.items():
        if golden_key not in target:
            continue
        delta = set(entry["state_dict_keys"]) - set(
            target[golden_key]["state_dict_keys"]
        )
        unclaimed = sorted(delta - claimed.get(golden_key, set()))
        if unclaimed:
            offenders[golden_key] = unclaimed
    return offenders


def unauthorized_additions(
    old: Dict[str, Dict[str, Any]],
    new: Dict[str, Dict[str, Any]],
    records: Sequence[_SchemaAddition] = (),
) -> Dict[str, List[str]]:
    """Keys that regeneration would add without a matching _SchemaAddition.

    Entries absent from `old` are skipped. A genuinely new class or case has no
    baseline to break, so its first snapshot needs no authorization. Pair this
    with removed_entries, which is what stops a rename from reaching that
    exemption while also adding a key.
    """
    return _unclaimed_delta(source=new, target=old, records=records)


def unauthorized_removals(
    old: Dict[str, Dict[str, Any]],
    new: Dict[str, Dict[str, Any]],
    records: Sequence[_SchemaRemoval] = (),
) -> Dict[str, List[str]]:
    """Keys that regeneration would drop without a matching _SchemaRemoval.

    Entries absent from `new` are skipped, because removed_entries already
    refuses those outright.
    """
    return _unclaimed_delta(source=old, target=new, records=records)


def removed_entries(
    old: Dict[str, Dict[str, Any]], new: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Golden entries that regeneration would drop.

    A METRICS_TO_TEST key is built from `__name__`, so renaming a metric moves
    its entry. The old one disappears and the replacement looks brand new, so a
    rename that also adds a state key would walk straight through the addition
    gate. Case-table keys are hand-written ids and do not move on a rename,
    which is what those ids are for.

    Losing an entry is rare and always worth a look, so this refuses rather
    than asking for a record.
    """
    return sorted(set(old) - set(new))


def stale_records(
    records: Sequence[_SchemaAddition], snapshot: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Addition records naming a key the snapshot does not hold.

    A record for a key that never landed, or that was later removed, is a claim
    nobody can check.
    """
    stale: List[str] = []
    for record in records:
        entry = snapshot.get(record.golden_key)
        if entry is None:
            stale.append(f"{record.golden_key} is not in the golden")
        elif record.state_key not in entry["state_dict_keys"]:
            stale.append(f"{record.golden_key} has no key {record.state_key!r}")
    return stale


def stale_removal_records(
    records: Sequence[_SchemaRemoval], snapshot: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Removal records whose key came back.

    Staleness inverts here. An addition record points at a key that must now be
    present; a removal record points at one that must now be absent. Re-adding
    the key leaves a record excusing a removal that no longer happened.
    """
    stale: List[str] = []
    for record in records:
        entry = snapshot.get(record.golden_key)
        if entry is None:
            stale.append(f"{record.golden_key} is not in the golden")
        elif record.state_key in entry["state_dict_keys"]:
            stale.append(f"{record.golden_key} still has key {record.state_key!r}")
    return stale


def update_golden_snapshot() -> None:
    print("Generating golden snapshot...")
    snapshot = generate_golden_snapshot()
    case_entries = generate_schema_case_entries()
    collisions = sorted(set(snapshot) & set(case_entries))
    if collisions:
        raise ValueError(
            f"Case keys collide with metric snapshot keys: {collisions}. "
            "One generator would overwrite the other."
        )
    snapshot.update(case_entries)

    # Not load_golden_snapshot: an absent file reads as {}, which turns every
    # generated entry into a brand-new one and silences all three gates below.
    previous = load_required_golden_snapshot()

    dropped = removed_entries(previous, snapshot)
    if dropped:
        raise ValueError(
            f"Regenerating would remove golden entries: {dropped}.\n"
            "No record authorizes this, deliberately. If the removal is right, "
            f"delete those entries from {GOLDEN_SNAPSHOT_PATH.name} by hand and "
            "run again, in the same diff that renames or drops the class. The "
            "hand edit is the acknowledgement.\n"
            "If it is not right, a case lost its coverage and the table needs "
            "the row back."
        )

    dropped_keys = unauthorized_removals(previous, snapshot, _SCHEMA_REMOVALS)
    if dropped_keys:
        detail = "\n".join(
            f"  {key}: {sorted(keys)}" for key, keys in sorted(dropped_keys.items())
        )
        raise ValueError(
            "Regenerating would drop state_dict keys that nothing authorizes:\n"
            f"{detail}\n"
            "A checkpoint written before the removal still carries the key. "
            "Loads built from the model never ask for it, but a "
            "checkpoint-derived load reads it back, finds nobody claiming it, "
            "and raises. Add a _SchemaRemoval for each, naming the hook that "
            "pops it."
        )

    offenders = unauthorized_additions(previous, snapshot, _SCHEMA_ADDITIONS)
    if offenders:
        detail = "\n".join(
            f"  {key}: {sorted(keys)}" for key, keys in sorted(offenders.items())
        )
        raise ValueError(
            "Regenerating would add state_dict keys that nothing authorizes:\n"
            f"{detail}\n"
            "An added key makes every older checkpoint unloadable, and no "
            "module hook can repair that: the planner rejects the load before "
            "any hook runs. Add a _SchemaAddition for each, naming the reason, "
            "the rollout plan for existing checkpoints, and an owner."
        )

    save_golden_snapshot(snapshot)
    print(f"Golden snapshot saved to {GOLDEN_SNAPSHOT_PATH}")
    print(f"Total metrics captured: {len(snapshot)}")
    for key in sorted(snapshot.keys()):
        info = snapshot[key]
        print(
            f"  - {key}: {len(info['state_dict_keys'])} state_dict keys, "
            f"{len(info['persistent_buffer_fqns'])} persistent buffers"
        )


if __name__ == "__main__":
    if "--update-golden" in sys.argv or os.environ.get("UPDATE_GOLDEN_SNAPSHOT"):
        if "--update-golden" in sys.argv:
            sys.argv.remove("--update-golden")
        update_golden_snapshot()
    else:
        unittest.main()
