/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <benchmark/benchmark.h>
#include <torch/torch.h>
#include <torchrec/csrc/dynamic_embedding/details/mixed_lfu_lru_strategy.h>

namespace torchrec {

class RecordIterator {
  using Container = std::vector<MixedLFULRUStrategy::Record>;

 public:
  RecordIterator(Container::const_iterator begin, Container::const_iterator end)
      : begin_(begin), end_(end) {}
  std::optional<TransformerRecord<uint32_t>> operator()() {
    if (begin_ == end_) {
      return std::nullopt;
    }
    TransformerRecord<uint32_t> record{};
    record.global_id_ = next_global_id_++;
    record.lxu_record_ = *reinterpret_cast<const uint32_t*>(&(*begin_++));
    return record;
  }

 private:
  int64_t next_global_id_{};
  Container::const_iterator begin_;
  Container::const_iterator end_;
};

class RandomizeMixedLXUSet {
 public:
  RandomizeMixedLXUSet(
      size_t n,
      uint8_t max_freq,
      uint32_t max_time,
      uint8_t min_freq = 5) {
    TORCH_CHECK(max_freq > min_freq);
    std::random_device dev;
    std::default_random_engine engine(dev());
    uint8_t freq_size = max_freq - min_freq;
    TORCH_CHECK(max_time > 1);
    records_.reserve(n);

    std::uniform_int_distribution<uint8_t> freq_dist(0, freq_size);
    std::uniform_int_distribution<uint32_t> time_dist(0, max_time - 1);
    for (size_t i = 0; i < n; ++i) {
      MixedLFULRUStrategy::Record record{};
      record.freq_power_ = freq_dist(engine) + min_freq;
      record.time_ = time_dist(engine);
      records_.emplace_back(record);
    }
  }

  [[nodiscard]] RecordIterator Iterator() const {
    return {records_.begin(), records_.end()};
  }

 private:
  std::vector<MixedLFULRUStrategy::Record> records_;
};

void BM_MixedLFULRUStrategyEvict(benchmark::State& state) {
  RandomizeMixedLXUSet lxuSet(state.range(0), state.range(1), state.range(2));
  for (auto _ : state) {
    MixedLFULRUStrategy::evict(lxuSet.Iterator(), state.range(3));
  }
}

BENCHMARK(BM_MixedLFULRUStrategyEvict)
    ->ArgNames({"total", "max_freq", "max_time", "num_to_evict"})
    ->Args({
        300000000 * 2,
        12,
        3000,
        5000000,
    });
} // namespace torchrec
