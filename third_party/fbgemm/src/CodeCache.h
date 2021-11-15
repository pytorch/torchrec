/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <condition_variable>
#include <future>
#include <map>

#if __cplusplus >= 201402L && !defined(__APPLE__)
// For C++14, use shared_timed_mutex.
// some macOS C++14 compilers don't support shared_timed_mutex.
#define FBGEMM_USE_SHARED_TIMED_MUTEX
#endif

#ifdef FBGEMM_USE_SHARED_TIMED_MUTEX
#include <shared_mutex>
#else
#include <mutex>
#endif

#ifdef FBCODE_CAFFE2
#include <folly/container/F14Map.h>
#endif


namespace fbgemm {

/**
 * @brief Thread safe cache for microkernels, ensures single creation per key.
 * @tparam Key Type of unique key (typically a tuple)
 * @tparam Value Type of the microkernel function (Typically a function pointer)
 */
template <typename KEY, typename VALUE>
class CodeCache {
 private:

#ifdef FBCODE_CAFFE2
  folly::F14FastMap<KEY, std::shared_future<VALUE>> values_;
#else
  std::map<KEY, std::shared_future<VALUE>> values_;
#endif

#ifdef FBGEMM_USE_SHARED_TIMED_MUTEX
  std::shared_timed_mutex mutex_;
#else
  std::mutex mutex_;
#endif

 public:
  CodeCache(const CodeCache&) = delete;
  CodeCache& operator=(const CodeCache&) = delete;

  CodeCache(){};

  template<typename GENFUNC>
  VALUE getOrCreate(const KEY& key, GENFUNC generatorFunction) {
    // Check for existence of the key
    {
#ifdef FBGEMM_USE_SHARED_TIMED_MUTEX
      std::shared_lock<std::shared_timed_mutex> sharedLock(mutex_);
#else
      std::unique_lock<std::mutex> uniqueLock(mutex_);
#endif

      auto it = values_.find(key);
      if (it != values_.end()) {
        return it->second.get();
      } else {
#ifdef FBGEMM_USE_SHARED_TIMED_MUTEX
        sharedLock.unlock();
        std::unique_lock<std::shared_timed_mutex> uniqueLock(mutex_);

        // Need to look up again because there could be race condition from
        // the time gap between sharedLock.unlock() and creating uniqueLock.
        it = values_.find(key);
        if (it == values_.end()) {
#endif
          std::promise<VALUE> returnPromise;
          values_[key] = returnPromise.get_future().share();

          uniqueLock.unlock();
          // The value (code) generation is not happening under a lock
          VALUE val = generatorFunction();
          returnPromise.set_value(val);
          return val;
#ifdef FBGEMM_USE_SHARED_TIMED_MUTEX
        } else {
          return it->second.get();
        }
#endif
      }
    }
  }
};

} // namespace fbgemm
