#pragma once
#include <memory>
#include <optional>
#include "nlohmann/json.hpp"
#include "tcb/span.hpp"
#include "tde/details/move_only_function.h"
#include "tde/details/naive_id_transformer.h"

namespace tde::details {

/**
 * CachelineIDTransformer
 *
 * Transform GlobalID to CacheID by naive flat hash map
 * @tparam LXURecord The extension type used for eviction strategy.
 * @tparam Bitmap The bitmap class to record the free cache ids.
 */
template <
    typename LXURecord,
    int64_t num_cacheline = 8,
    int64_t cacheline_size = 64,
    typename BitMap = Bitmap<uint32_t>,
    typename Hash = std::hash<int64_t>>
class CachelineIDTransformer {
 public:
  static_assert(num_cacheline > 0, "num_cacheline should be positive.");
  static_assert(cacheline_size > 0, "cacheline_size should be positive.");

  using lxu_record_t = LXURecord;
  using record_t = TransformerRecord<lxu_record_t>;
  enum {
    TransformUpdateNeedThreadSafe = 0,
    TransformFetchNeedThreadSafe = 0,
    TransformHasFilter = 1,
    TransformerHasCacheIDTransformer = 1,
    TransformCanContinue = 1,
    IsCompose = 0,
  };
  static constexpr std::string_view type_ = "cacheline";

  explicit CachelineIDTransformer(int64_t num_embedding, int64_t capacity = 0);
  CachelineIDTransformer(const CachelineIDTransformer<
                         LXURecord,
                         num_cacheline,
                         cacheline_size,
                         BitMap,
                         Hash>&) = delete;
  CachelineIDTransformer(CachelineIDTransformer<
                         LXURecord,
                         num_cacheline,
                         cacheline_size,
                         BitMap,
                         Hash>&&) noexcept = default;

  static CachelineIDTransformer<
      LXURecord,
      num_cacheline,
      cacheline_size,
      BitMap,
      Hash>
  Create(int64_t num_embedding, const nlohmann::json& json) {
    return CachelineIDTransformer<
        LXURecord,
        num_cacheline,
        cacheline_size,
        BitMap,
        Hash>(num_embedding);
  }

  /**
   * Transform global ids to cache ids
   * @tparam Filter To filter whether this transformer need
   * process this global id. By default it process all global-ids.
   * This type is used by composed id transformers like
   * `MultiThreadedIDTransformer`.
   *
   * @tparam CacheIDTransformer Transform the result cache id. It is used by
   * composed id transformers like `MultiThreadedIDTransformer`.
   *
   * @tparam Update Update the eviction strategy tag type. Update LXU Record
   * @tparam Fetch Fetch the not existing global-id/cache-id pair. It is used
   * by dynamic embedding parameter server.
   *
   * @param global_ids Global ID vector
   * @param cache_ids [out] Cache ID vector
   * @param filter Filter lambda. See `Filter`'s doc.
   * @param cache_id_transformer cache_id_transformer lambda. See
   * `CacheIDTransformer`'s doc.
   * @param update update lambda. See `Update` doc.
   * @param fetch fetch lambda. See `Fetch` doc.
   * @return offset that has been processed. If offset is not equals to
   * global_ids.size(), it means the Transformer is Full, and need to be evict.
   */
  template <
      typename Filter = decltype(transform_default::All),
      typename CacheIDTransformer = decltype(transform_default::Identity),
      typename Update = decltype(transform_default::NoUpdate<LXURecord>),
      typename Fetch = decltype(transform_default::NoFetch)>
  bool Transform(
      tcb::span<const int64_t> global_ids,
      tcb::span<int64_t> cache_ids,
      Filter filter = transform_default::All,
      CacheIDTransformer cache_id_transformer = transform_default::Identity,
      Update update = transform_default::NoUpdate<LXURecord>,
      Fetch fetch = transform_default::NoFetch);

  template <typename Callback>
  void ForEach(
      Callback callback =
          [](int64_t global_id, int64_t cache_id, LXURecord tag) {});

  void Evict(tcb::span<const int64_t> global_ids);

  MoveOnlyFunction<std::optional<record_t>()> Iterator() const;

 private:
  struct CacheValue {
    int64_t neg_global_id_;
    uint32_t cache_id_;
    LXURecord lxu_record_;

    int64_t global_id() {
      return -(neg_global_id_ + 1);
    }
    uint32_t cache_id() {
      return cache_id_;
    }
    void set_cache_id(uint32_t cache_id) {
      cache_id_ = cache_id;
    }

    bool is_filled() {
      return neg_global_id_ < 0;
    }
    void set_empty() {
      neg_global_id_ = 0;
    }
  };
  static_assert(sizeof(CacheValue) <= 16);

  static constexpr int64_t kFullMask = 1LL << 63;
  static_assert(kFullMask < 0);

  std::tuple<int64_t, int64_t> FindGroupIndex(int64_t val) {
    int64_t hash = hasher_(val);
    return {hash / group_size_ % num_groups_, hash % group_size_};
  }

  static constexpr int64_t group_size_ =
      num_cacheline * cacheline_size / static_cast<int64_t>(sizeof(CacheValue));
  int64_t num_groups_;
  Hash hasher_;
  std::unique_ptr<CacheValue[]> cache_values_;
  BitMap bitmap_;
};

} // namespace tde::details

#include "tde/details/cacheline_id_transformer_impl.h"
