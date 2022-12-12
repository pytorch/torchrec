import torch
import unittest
import torch.distributed as dist
import os
import time

class Barrier(object):
    barrier_id = 0

    @classmethod
    def init(cls):
        cls.barrier_id = 0
        barrier_dir = os.path.join(os.environ["TEMP_DIR"], "barrier")
        for f_name in os.listdir(barrier_dir):
            os.unlink(os.path.join(barrier_dir, f_name))

    @classmethod
    def sync(cls, wait_for=None, timeout=10):
        if wait_for is None:
            wait_for = dist.get_world_size()
        cls.barrier_id += 1
        barrier_dir = os.path.join(os.environ["TEMP_DIR"], "barrier")
        pid = str(os.getpid())
        barrier_file = os.path.join(barrier_dir, pid)
        with _lock():
            with open(barrier_file, "w") as f:
                f.write(str(cls.barrier_id))

        start_time = time.time()
        while True:
            arrived = 0
            with _lock():
                for f_name in os.listdir(barrier_dir):
                    with open(os.path.join(barrier_dir, f_name), "r") as f:
                        data = f.read()
                        if int(data) >= cls.barrier_id:
                            arrived += 1
            if arrived == wait_for:
                break

            if time.time() - start_time > timeout:
                raise RuntimeError("barrier timeout")
            time.sleep(0.1)

def init_multigpu_helper(world_size: int, backend: str):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    nGPUs = torch.cuda.device_count()
    visible_devices = range(nGPUs)

    if backend == "nccl":
        # This is a hack for a known NCCL issue using multiprocess
        # in conjunction with multiple threads to manage different GPUs which
        # may cause ncclCommInitRank to fail.
        # http://docs.nvidia.com/deeplearning/sdk/nccl-release-notes/rel_2.1.4.html#rel_2.1.4
        # It slows down the performance of collective operations.
        # Without this setting NCCL might throw unhandled error.
        os.environ["NCCL_MAX_NRINGS"] = "1"

    # If rank is less than or equal to number of available GPU's
    # then each rank can be mapped to corresponding GPU.
    nGPUs_per_process = 1
    if world_size > nGPUs:
        nGPUs_per_process = nGPUs // world_size
    rank_to_GPU = {
        i: list(visible_devices[i * nGPUs_per_process : (i + 1) * nGPUs_per_process])
        for i in range(world_size)
    }
    return rank_to_GPU

class TestAllToAllSingleDistributed(unittest.TestCase):
    def _barrier(self, *args, **kwargs):
        Barrier.sync(*args, **kwargs)

    def call_dist_op(
        self,
        profiling_title_postfix,
        is_async,
        op,
        *args,
        expect_event=True,
        secondary_op_call=None,
        profile_cuda=False,
        tensor_shapes=None,
        **kwargs,
    ):
        op_calls = [lambda: op(*args, **kwargs)]
        if secondary_op_call is not None:
            op_calls.append(secondary_op_call)

        autograd_profiler_ctx = torch.autograd.profiler.profile(
            use_cuda=profile_cuda, record_shapes=True
        )

        # TODO: move this test to use torch.profiler once kineto issues are
        # fixed internally.
        with autograd_profiler_ctx as prof:
            works = [op_call() for op_call in op_calls]
            if is_async:
                for work in works:
                    work.wait()

        if expect_event and dist.get_backend() in PROFILING_SUPPORTED_BACKENDS:
            # We are only interested in the backend's implementation not the dispatcher wrapper.
            events = get_profiling_event(
                dist.get_backend() + profiling_title_postfix, autograd_profiler_ctx
            )
            # DETAIL debug mode can use a pg wrapper that issues more collectives
            # under the hood
            if dist.get_debug_level() != dist.DebugLevel.DETAIL:
                self.assertEqual(len(events), len(op_calls))
            for e in events:
                self.assertTrue(e.is_async)
                self.assertEqual(e.count, 1)
                self.assertGreaterEqual(e.cpu_time, 0)
                # Verify tensor shapes if given
                # DETAIL debug mode can use a pg wrapper that issues more collectives
                # under the hood
                if (
                    tensor_shapes is not None
                    and dist.get_debug_level() != dist.DebugLevel.DETAIL
                ):
                    self.assertEqual(
                        e.input_shapes,
                        tensor_shapes,
                        f"event shape: {e.input_shapes} vs tensor {tensor_shapes}",
                    )

    def _test_all_to_all_single_equal_split_helper(
            self, group, group_id, rank, cuda=False, rank_to_GPU=None, dtype=torch.float
        ):
            if group_id is not None:
                size = len(group)
                in_tensor = torch.ones([size, size], dtype=dtype) * rank
                expected_tensor = torch.cat(
                    [torch.ones([1, size], dtype=dtype) * i for i in group]
                )
                out_tensor = torch.ones([size, size], dtype=dtype) * -1
                if cuda:
                    in_tensor = in_tensor.cuda(rank_to_GPU[rank][0])
                    expected_tensor = expected_tensor.cuda(rank_to_GPU[rank][0])
                    out_tensor = out_tensor.cuda(rank_to_GPU[rank][0])
                if dtype == torch.complex64:
                    tensor_shapes = [torch.view_as_real(in_tensor).shape]
                else:
                    tensor_shapes = [in_tensor.shape]
                self.call_dist_op(
                    ":all_to_all",
                    False,
                    dist.all_to_all_single,
                    out_tensor,
                    in_tensor,
                    group=group_id,
                    tensor_shapes=tensor_shapes,
                )
                self.assertEqual(out_tensor, expected_tensor)
            self._barrier()

    def _test_all_to_all_single_unequal_split_helper(
        self, group, group_id, rank, cuda=False, rank_to_GPU=None, dtype=torch.float
    ):
        if group_id is not None:
            size = len(group)
            in_splits = [i + 1 for i in group]
            out_splits = [rank + 1 for _ in group]
            in_tensor = torch.ones([sum(in_splits), size], dtype=dtype) * rank
            out_tensor = torch.ones([(rank + 1) * size, size], dtype=dtype)
            expected_tensor = torch.cat(
                [torch.ones([rank + 1, size], dtype=dtype) * i for i in group]
            )
            if cuda:
                in_tensor = in_tensor.cuda(rank_to_GPU[rank][0])
                expected_tensor = expected_tensor.cuda(rank_to_GPU[rank][0])
                out_tensor = out_tensor.cuda(rank_to_GPU[rank][0])
            dist.all_to_all_single(
                out_tensor, in_tensor, out_splits, in_splits, group=group_id
            )
            self.assertEqual(out_tensor, expected_tensor)
        self._barrier()

    def _test_all_to_all_helper(
        self,
        group,
        group_id,
        rank,
        cuda=False,
        rank_to_GPU=None,
        dtype=torch.float,
    ):
        if group_id is not None:
            size = len(group)
            in_splits = [i + 1 for i in group]
            in_tensors = [
                torch.ones([in_splits[i], size], dtype=dtype) * rank
                for i, _ in enumerate(group)
            ]
            out_tensors = [
                torch.ones([(rank + 1), size], dtype=dtype) for _ in group
            ]
            expected_tensors = [
                torch.ones([rank + 1, size], dtype=dtype) * i for i in group
            ]
            if cuda:
                in_tensors = [t.cuda(rank_to_GPU[rank][0]) for t in in_tensors]
                expected_tensors = [
                    t.cuda(rank_to_GPU[rank][0]) for t in expected_tensors
                ]
                out_tensors = [t.cuda(rank_to_GPU[rank][0]) for t in out_tensors]
            dist.all_to_all(out_tensors, in_tensors, group=group_id)
            for t1, t2 in zip(out_tensors, expected_tensors):
                self.assertEqual(t1, t2)
        self._barrier()


    def test_all_to_all_single_equal_split_cuda(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
        self._test_all_to_all_single_equal_split_helper(
            group,
            group_id,
            rank,
            True,
            rank_to_GPU,
        )

    def test_all_to_all_single_equal_split_cuda_complex(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
        self._test_all_to_all_single_equal_split_helper(
            group, group_id, rank, True, rank_to_GPU, dtype=torch.cfloat
        )

    def test_all_to_all_single_unequal_split_cuda(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
        self._test_all_to_all_single_unequal_split_helper(
            group,
            group_id,
            rank,
            True,
            rank_to_GPU,
        )

    def test_all_to_all_single_unequal_split_cuda_complex(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
        self._test_all_to_all_single_unequal_split_helper(
            group,
            group_id,
            rank,
            True,
            rank_to_GPU,
            dtype=torch.cfloat,
        )
