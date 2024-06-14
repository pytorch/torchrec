/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <torch/nn/functional/activation.h>
#include <torch/script.h>

#ifdef BAZEL_BUILD
#include "examples/protos/predictor.grpc.pb.h"
#else
#include "predictor.grpc.pb.h"
#endif

#define NUM_BYTES_FLOAT_FEATURES 4
#define NUM_BYTES_SPARSE_FEATURES 4

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

ABSL_FLAG(uint16_t, port, 50051, "Server port for the service");

using predictor::FloatVec;
using predictor::PredictionRequest;
using predictor::PredictionResponse;
using predictor::Predictor;

class PredictorServiceHandler final : public Predictor::Service {
 public:
  PredictorServiceHandler(torch::jit::script::Module& module)
      : module_(module) {}

  Status Predict(
      grpc::ServerContext* context,
      const PredictionRequest* request,
      PredictionResponse* reply) override {
    std::cout << "Predict Called!" << std::endl;
    c10::Dict<std::string, at::Tensor> dict;

    auto floatFeature = request->float_features();
    auto floatFeatureBlob = floatFeature.values();
    auto numFloatFeatures = floatFeature.num_features();
    auto batchSize =
        floatFeatureBlob.size() / (NUM_BYTES_FLOAT_FEATURES * numFloatFeatures);

    std::cout << "Size: " << floatFeatureBlob.size()
              << " Num Features: " << numFloatFeatures << std::endl;
    auto floatFeatureTensor = torch::from_blob(
        floatFeatureBlob.data(),
        {batchSize, numFloatFeatures},
        torch::kFloat32);

    auto idListFeature = request->id_list_features();
    auto numIdListFeatures = idListFeature.num_features();
    auto lengthsBlob = idListFeature.lengths();
    auto valuesBlob = idListFeature.values();

    std::cout << "Lengths Size: " << lengthsBlob.size()
              << " Num Features: " << numIdListFeatures << std::endl;
    assert(
        batchSize ==
        (lengthsBlob.size() / (NUM_BYTES_SPARSE_FEATURES * numIdListFeatures)));

    auto lengthsTensor = torch::from_blob(
        lengthsBlob.data(),
        {lengthsBlob.size() / NUM_BYTES_SPARSE_FEATURES},
        torch::kInt32);
    auto valuesTensor = torch::from_blob(
        valuesBlob.data(),
        {valuesBlob.size() / NUM_BYTES_SPARSE_FEATURES},
        torch::kInt32);

    dict.insert("float_features", floatFeatureTensor.to(torch::kCUDA));
    dict.insert("id_list_features.lengths", lengthsTensor.to(torch::kCUDA));
    dict.insert("id_list_features.values", valuesTensor.to(torch::kCUDA));

    std::vector<c10::IValue> input;
    input.push_back(c10::IValue(dict));

    torch::Tensor output =
        this->module_.forward(input).toGenericDict().at("default").toTensor();

    auto predictions = reply->mutable_predictions();
    FloatVec fv;
    fv.mutable_data()->Add(
        output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
    (*predictions)["default"] = fv;
    return Status::OK;
  }

 private:
  torch::jit::script::Module& module_;
};

void RunServer(uint16_t port, torch::jit::script::Module& module) {
  std::string server_address = absl::StrFormat("0.0.0.0:%d", port);
  PredictorServiceHandler service(module);

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char** argv) {
  // absl::ParseCommandLine(argc, argv);

  if (argc != 2) {
    std::cerr << "usage: ts-infer <path-to-exported-model>\n";
    return -1;
  }

  std::cout << "Loading model...\n";

  // deserialize ScriptModule
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "Error loading model\n";
    return -1;
  }

  torch::NoGradGuard no_grad; // ensures that autograd is off
  module.eval(); // turn off dropout and other training-time layers/functions

  std::cout << "Sanity Check with dummy inputs" << std::endl;
  c10::Dict<std::string, at::Tensor> dict;
  dict.insert(
      "float_features",
      torch::ones(
          {1, 13}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)));
  dict.insert(
      "id_list_features.lengths",
      torch::ones({26}, torch::dtype(torch::kLong).device(torch::kCUDA, 0)));
  dict.insert(
      "id_list_features.values",
      torch::ones({26}, torch::dtype(torch::kLong).device(torch::kCUDA, 0)));

  std::vector<c10::IValue> input;
  input.push_back(c10::IValue(dict));

  // Execute the model and turn its output into a tensor.
  auto output = module.forward(input).toGenericDict().at("default").toTensor();
  std::cout << " Model Forward Completed, Output: " << output.item<float>()
            << std::endl;

  RunServer(absl::GetFlag(FLAGS_port), module);

  return 0;
}
