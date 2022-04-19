// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: predictor.proto
#ifndef GRPC_predictor_2eproto__INCLUDED
#define GRPC_predictor_2eproto__INCLUDED

#include "predictor.pb.h"

#include <grpcpp/impl/codegen/async_generic_service.h>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/client_context.h>
#include <grpcpp/impl/codegen/completion_queue.h>
#include <grpcpp/impl/codegen/message_allocator.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/server_callback_handlers.h>
#include <grpcpp/impl/codegen/server_context.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/stub_options.h>
#include <grpcpp/impl/codegen/sync_stream.h>
#include <functional>

namespace predictor {

// The predictor service definition. Synchronous for now.
class Predictor final {
 public:
  static constexpr char const* service_full_name() {
    return "predictor.Predictor";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status Predict(
        ::grpc::ClientContext* context,
        const ::predictor::PredictionRequest& request,
        ::predictor::PredictionResponse* response) = 0;
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::predictor::PredictionResponse>>
    AsyncPredict(
        ::grpc::ClientContext* context,
        const ::predictor::PredictionRequest& request,
        ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::predictor::PredictionResponse>>(
          AsyncPredictRaw(context, request, cq));
    }
    std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
        ::predictor::PredictionResponse>>
    PrepareAsyncPredict(
        ::grpc::ClientContext* context,
        const ::predictor::PredictionRequest& request,
        ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<::grpc::ClientAsyncResponseReaderInterface<
          ::predictor::PredictionResponse>>(
          PrepareAsyncPredictRaw(context, request, cq));
    }
    class async_interface {
     public:
      virtual ~async_interface() {}
      virtual void Predict(
          ::grpc::ClientContext* context,
          const ::predictor::PredictionRequest* request,
          ::predictor::PredictionResponse* response,
          std::function<void(::grpc::Status)>) = 0;
      virtual void Predict(
          ::grpc::ClientContext* context,
          const ::predictor::PredictionRequest* request,
          ::predictor::PredictionResponse* response,
          ::grpc::ClientUnaryReactor* reactor) = 0;
    };
    typedef class async_interface experimental_async_interface;
    virtual class async_interface* async() {
      return nullptr;
    }
    class async_interface* experimental_async() {
      return async();
    }

   private:
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::predictor::PredictionResponse>*
    AsyncPredictRaw(
        ::grpc::ClientContext* context,
        const ::predictor::PredictionRequest& request,
        ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface<
        ::predictor::PredictionResponse>*
    PrepareAsyncPredictRaw(
        ::grpc::ClientContext* context,
        const ::predictor::PredictionRequest& request,
        ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(
        const std::shared_ptr<::grpc::ChannelInterface>& channel,
        const ::grpc::StubOptions& options = ::grpc::StubOptions());
    ::grpc::Status Predict(
        ::grpc::ClientContext* context,
        const ::predictor::PredictionRequest& request,
        ::predictor::PredictionResponse* response) override;
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::predictor::PredictionResponse>>
    AsyncPredict(
        ::grpc::ClientContext* context,
        const ::predictor::PredictionRequest& request,
        ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::predictor::PredictionResponse>>(
          AsyncPredictRaw(context, request, cq));
    }
    std::unique_ptr<
        ::grpc::ClientAsyncResponseReader<::predictor::PredictionResponse>>
    PrepareAsyncPredict(
        ::grpc::ClientContext* context,
        const ::predictor::PredictionRequest& request,
        ::grpc::CompletionQueue* cq) {
      return std::unique_ptr<
          ::grpc::ClientAsyncResponseReader<::predictor::PredictionResponse>>(
          PrepareAsyncPredictRaw(context, request, cq));
    }
    class async final : public StubInterface::async_interface {
     public:
      void Predict(
          ::grpc::ClientContext* context,
          const ::predictor::PredictionRequest* request,
          ::predictor::PredictionResponse* response,
          std::function<void(::grpc::Status)>) override;
      void Predict(
          ::grpc::ClientContext* context,
          const ::predictor::PredictionRequest* request,
          ::predictor::PredictionResponse* response,
          ::grpc::ClientUnaryReactor* reactor) override;

     private:
      friend class Stub;
      explicit async(Stub* stub) : stub_(stub) {}
      Stub* stub() {
        return stub_;
      }
      Stub* stub_;
    };
    class async* async() override {
      return &async_stub_;
    }

   private:
    std::shared_ptr<::grpc::ChannelInterface> channel_;
    class async async_stub_ {
      this
    };
    ::grpc::ClientAsyncResponseReader<::predictor::PredictionResponse>*
    AsyncPredictRaw(
        ::grpc::ClientContext* context,
        const ::predictor::PredictionRequest& request,
        ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader<::predictor::PredictionResponse>*
    PrepareAsyncPredictRaw(
        ::grpc::ClientContext* context,
        const ::predictor::PredictionRequest& request,
        ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_Predict_;
  };
  static std::unique_ptr<Stub> NewStub(
      const std::shared_ptr<::grpc::ChannelInterface>& channel,
      const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status Predict(
        ::grpc::ServerContext* context,
        const ::predictor::PredictionRequest* request,
        ::predictor::PredictionResponse* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_Predict : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}

   public:
    WithAsyncMethod_Predict() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_Predict() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Predict(
        ::grpc::ServerContext* /*context*/,
        const ::predictor::PredictionRequest* /*request*/,
        ::predictor::PredictionResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestPredict(
        ::grpc::ServerContext* context,
        ::predictor::PredictionRequest* request,
        ::grpc::ServerAsyncResponseWriter<::predictor::PredictionResponse>*
            response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq,
        void* tag) {
      ::grpc::Service::RequestAsyncUnary(
          0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_Predict<Service> AsyncService;
  template <class BaseClass>
  class WithCallbackMethod_Predict : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}

   public:
    WithCallbackMethod_Predict() {
      ::grpc::Service::MarkMethodCallback(
          0,
          new ::grpc::internal::CallbackUnaryHandler<
              ::predictor::PredictionRequest,
              ::predictor::PredictionResponse>(
              [this](
                  ::grpc::CallbackServerContext* context,
                  const ::predictor::PredictionRequest* request,
                  ::predictor::PredictionResponse* response) {
                return this->Predict(context, request, response);
              }));
    }
    void SetMessageAllocatorFor_Predict(
        ::grpc::MessageAllocator<
            ::predictor::PredictionRequest,
            ::predictor::PredictionResponse>* allocator) {
      ::grpc::internal::MethodHandler* const handler =
          ::grpc::Service::GetHandler(0);
      static_cast<::grpc::internal::CallbackUnaryHandler<
          ::predictor::PredictionRequest,
          ::predictor::PredictionResponse>*>(handler)
          ->SetMessageAllocator(allocator);
    }
    ~WithCallbackMethod_Predict() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Predict(
        ::grpc::ServerContext* /*context*/,
        const ::predictor::PredictionRequest* /*request*/,
        ::predictor::PredictionResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* Predict(
        ::grpc::CallbackServerContext* /*context*/,
        const ::predictor::PredictionRequest* /*request*/,
        ::predictor::PredictionResponse* /*response*/) {
      return nullptr;
    }
  };
  typedef WithCallbackMethod_Predict<Service> CallbackService;
  typedef CallbackService ExperimentalCallbackService;
  template <class BaseClass>
  class WithGenericMethod_Predict : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}

   public:
    WithGenericMethod_Predict() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_Predict() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Predict(
        ::grpc::ServerContext* /*context*/,
        const ::predictor::PredictionRequest* /*request*/,
        ::predictor::PredictionResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_Predict : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}

   public:
    WithRawMethod_Predict() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_Predict() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Predict(
        ::grpc::ServerContext* /*context*/,
        const ::predictor::PredictionRequest* /*request*/,
        ::predictor::PredictionResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestPredict(
        ::grpc::ServerContext* context,
        ::grpc::ByteBuffer* request,
        ::grpc::ServerAsyncResponseWriter<::grpc::ByteBuffer>* response,
        ::grpc::CompletionQueue* new_call_cq,
        ::grpc::ServerCompletionQueue* notification_cq,
        void* tag) {
      ::grpc::Service::RequestAsyncUnary(
          0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class WithRawCallbackMethod_Predict : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}

   public:
    WithRawCallbackMethod_Predict() {
      ::grpc::Service::MarkMethodRawCallback(
          0,
          new ::grpc::internal::
              CallbackUnaryHandler<::grpc::ByteBuffer, ::grpc::ByteBuffer>(
                  [this](
                      ::grpc::CallbackServerContext* context,
                      const ::grpc::ByteBuffer* request,
                      ::grpc::ByteBuffer* response) {
                    return this->Predict(context, request, response);
                  }));
    }
    ~WithRawCallbackMethod_Predict() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status Predict(
        ::grpc::ServerContext* /*context*/,
        const ::predictor::PredictionRequest* /*request*/,
        ::predictor::PredictionResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual ::grpc::ServerUnaryReactor* Predict(
        ::grpc::CallbackServerContext* /*context*/,
        const ::grpc::ByteBuffer* /*request*/,
        ::grpc::ByteBuffer* /*response*/) {
      return nullptr;
    }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_Predict : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}

   public:
    WithStreamedUnaryMethod_Predict() {
      ::grpc::Service::MarkMethodStreamed(
          0,
          new ::grpc::internal::StreamedUnaryHandler<
              ::predictor::PredictionRequest,
              ::predictor::PredictionResponse>(
              [this](
                  ::grpc::ServerContext* context,
                  ::grpc::ServerUnaryStreamer<
                      ::predictor::PredictionRequest,
                      ::predictor::PredictionResponse>* streamer) {
                return this->StreamedPredict(context, streamer);
              }));
    }
    ~WithStreamedUnaryMethod_Predict() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status Predict(
        ::grpc::ServerContext* /*context*/,
        const ::predictor::PredictionRequest* /*request*/,
        ::predictor::PredictionResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedPredict(
        ::grpc::ServerContext* context,
        ::grpc::ServerUnaryStreamer<
            ::predictor::PredictionRequest,
            ::predictor::PredictionResponse>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_Predict<Service> StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_Predict<Service> StreamedService;
};

} // namespace predictor

#endif // GRPC_predictor_2eproto__INCLUDED
