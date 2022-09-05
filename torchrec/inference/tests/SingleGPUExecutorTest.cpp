/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/SingleGPUExecutor.h"
#include <folly/futures/Future.h>
#include <gtest/gtest.h>
#include <multipy/runtime/deploy.h>
#include <torch/cuda.h> // @manual
#include <torch/script.h>
#include <torch/torch.h> // @manual
#include "torchrec/inference/Types.h"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  return rc;
}

const char* path(const char* envname, const char* path) {
  const char* e = std::getenv(envname);
  return e ? e : path;
}

std::vector<c10::IValue> get_input_example(
    torch::deploy::InterpreterSession& model_interpreter_session) {
  auto eg = model_interpreter_session.self
                .attr("load_pickle")({"model", "example.pkl"})
                .toIValue();
  return eg.toTupleRef().elements();
}

void assert_tensors_eq(const at::Tensor& expected, const at::Tensor& got) {
  ASSERT_TRUE(expected.allclose(got, 1e-03, 1e-05));
}

TEST(TorchDeployGPUTest, SimpleModel) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP();
  }

  const char* model_filename = path("TORCH_PACKAGE_SIMPLE", "/tmp/simple");

  auto device = c10::Device(c10::kCUDA, 0);

  auto manager = std::make_shared<torch::deploy::InterpreterManager>(1);
  torch::deploy::Package p = manager->loadPackage(model_filename);
  auto model = p.loadPickle("model", "model.pkl");
  {
    auto M = model.acquireSession();
    M.self.attr("to")({device});
  }

  std::vector<at::IValue> example_inputs;
  {
    auto I = p.acquireSession();
    example_inputs = get_input_example(I);
  }

  std::vector<size_t> interp_idxs = {0};
  auto executor = std::make_unique<torchrec::SingleGPUExecutor>(
      manager, model, device, interp_idxs);

  {
    auto inputs = example_inputs;

    // Calling forward on freshly loaded model
    folly::Promise<std::unique_ptr<torchrec::PredictionResponse>> promise;
    auto future = promise.getSemiFuture();

    executor->schedule(std::make_shared<torchrec::PredictionBatch>(
        "forward", inputs, std::move(promise)));

    auto forward_result_0 = std::move(future).get()->predictions.toTensor();
    auto expected = example_inputs[0].toTensor() + at::ones({10, 20});
    assert_tensors_eq(expected, forward_result_0);
  }

  {
    // Calling set_weight
    folly::Promise<std::unique_ptr<torchrec::PredictionResponse>> promise;
    auto future = promise.getSemiFuture();

    std::vector<c10::IValue> set_weight_inputs = {at::zeros({10, 20})};

    executor->schedule(std::make_shared<torchrec::PredictionBatch>(
        "set_weight", set_weight_inputs, std::move(promise)));

    auto predictionResponse = std::move(future).get();
  }

  {
    auto inputs = example_inputs;

    // Calling forward on changed model
    folly::Promise<std::unique_ptr<torchrec::PredictionResponse>> promise;
    auto future = promise.getSemiFuture();

    executor->schedule(std::make_shared<torchrec::PredictionBatch>(
        "forward", inputs, std::move(promise)));

    auto forward_result_1 = std::move(future).get()->predictions.toTensor();
    auto expected = example_inputs[0].toTensor();
    assert_tensors_eq(expected, forward_result_1);
  }
}

c10::IValue execute(
    torchrec::SingleGPUExecutor& executor,
    const std::string& methodName,
    std::vector<c10::IValue> args) {
  folly::Promise<std::unique_ptr<torchrec::PredictionResponse>> promise;
  auto future = promise.getSemiFuture();

  executor.schedule(std::make_shared<torchrec::PredictionBatch>(
      methodName, std::move(args), std::move(promise)));
  return std::move(future).get()->predictions;
}

TEST(TorchDeployGPUTest, SimpleModel_multiGPU) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP();
  }

  const size_t numGpu = torch::cuda::device_count();
  if (numGpu <= 1) {
    GTEST_SKIP();
  }

  const char* model_filename = path("TORCH_PACKAGE_SIMPLE", "/tmp/simple");

  auto device = c10::Device(c10::kCUDA, 0);

  auto manager =
      std::make_shared<torch::deploy::InterpreterManager>(numGpu + 1);
  torch::deploy::Package package = manager->loadPackage(model_filename);

  std::vector<torch::deploy::ReplicatedObj> models;
  torch::deploy::ReplicatedObj model_control;
  const size_t gpu_rank_control = 0;

  {
    auto I = package.acquireSession();

    auto pyModel = I.fromMovable(package.loadPickle("model", "model.pkl"));

    for (size_t i = 0; i < numGpu; i++) {
      auto model = I.createMovable(
          pyModel.attr("to")(c10::IValue(c10::Device(c10::kCUDA, i))));
      if (i == gpu_rank_control) {
        model_control = model;
      }
      models.push_back(std::move(model));
    }
  }

  std::vector<std::unique_ptr<torchrec::SingleGPUExecutor>> workExecutors;
  for (size_t i = 0; i < numGpu; i++) {
    const std::vector<size_t> interp_idxs = {static_cast<size_t>(i)};
    workExecutors.push_back(std::make_unique<torchrec::SingleGPUExecutor>(
        manager,
        std::move(models[i]),
        c10::Device(c10::kCUDA, i),
        interp_idxs));
  }

  const std::vector<size_t> interp_idxs = {static_cast<size_t>(numGpu)};
  auto controlExecutor = std::make_unique<torchrec::SingleGPUExecutor>(
      manager,
      std::move(model_control),
      c10::Device(c10::kCUDA, gpu_rank_control),
      interp_idxs);

  std::vector<at::IValue> example_inputs;
  {
    auto I = package.acquireSession();
    example_inputs = get_input_example(I);
  }
  auto example_input0 = example_inputs[0].toTensor();
  auto expected_forward0 = example_input0 + at::ones(example_input0.sizes());

  for (size_t i = 0; i < numGpu; i++) {
    auto result =
        execute(*workExecutors[i], "forward", example_inputs).toTensor();
    assert_tensors_eq(expected_forward0, result);
  }

  execute(*controlExecutor, "set_weight", {at::zeros(example_input0.sizes())});
  for (size_t i = 0; i < numGpu; i++) {
    auto result =
        execute(*workExecutors[i], "forward", example_inputs).toTensor();
    if (i == gpu_rank_control) {
      assert_tensors_eq(example_input0, result);
    } else {
      assert_tensors_eq(expected_forward0, result);
    }
  }
}
