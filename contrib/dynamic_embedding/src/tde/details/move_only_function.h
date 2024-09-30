#pragma once

#include <memory>
#include <utility>

namespace tde::details {

template <typename T>
class MoveOnlyFunction;

template <typename R, typename... Args>
class MoveOnlyFunction<R(Args...)> {
 public:
  MoveOnlyFunction() = default;

  template <typename F>
  /*implicit*/ MoveOnlyFunction(F f) : f_(new Derived<F>(std::move(f))) {}

  MoveOnlyFunction(const MoveOnlyFunction<R(Args...)>&) = delete;
  MoveOnlyFunction(MoveOnlyFunction<R(Args...)>&& o) noexcept
      : f_(std::move(o.f_)) {}

  MoveOnlyFunction<R(Args...)>& operator=(const MoveOnlyFunction<R(Args...)>&) =
      delete;
  MoveOnlyFunction<R(Args...)>& operator=(
      MoveOnlyFunction<R(Args...)>&&) noexcept = default;

  template <typename F>
  MoveOnlyFunction<R(Args...)>& operator=(F f) {
    (*this) = MoveOnlyFunction<R(Args...)>(std::move(f));
    return *this;
  }

  R operator()(Args&&... args) {
    return (*f_)(std::forward<Args>(args)...);
  }

  /*implicit*/ operator bool() const {
    return f_ != nullptr;
  }

 private:
  struct Base {
    virtual ~Base() = default;
    virtual R operator()(Args&&...) = 0;
  };

  template <typename F>
  struct Derived final : public Base {
    explicit Derived(F f) : f_(std::move(f)) {}
    R operator()(Args&&... args) override {
      return f_(std::forward<Args>(args)...);
    }
    F f_;
  };
  std::unique_ptr<Base> f_;
};

} // namespace tde::details
