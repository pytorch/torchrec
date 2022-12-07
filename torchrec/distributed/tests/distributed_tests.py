class TestAllToAllSingleDistributed(unittest.TestCase):
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

    @sandcastle_skip_if(BACKEND != "nccl", "Only Nccl supports CUDA all_to_all_single")
    @skip_if_no_gpu
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

    @sandcastle_skip_if(BACKEND != "nccl", "Only Nccl supports CUDA all_to_all_single")
    @skip_if_no_gpu
    def test_all_to_all_single_equal_split_cuda_complex(self):
        group, group_id, rank = self._init_global_test()
        rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
        self._test_all_to_all_single_equal_split_helper(
            group, group_id, rank, True, rank_to_GPU, dtype=torch.cfloat
        )

    @sandcastle_skip_if(BACKEND != "nccl", "Only Nccl supports CUDA all_to_all_single")
    @skip_if_no_gpu
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

    @sandcastle_skip_if(BACKEND != "nccl", "Only Nccl supports CUDA all_to_all_single")
    @skip_if_no_gpu
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
