#pragma once
#include <c10/util/flat_hash_map.h>
#include <memory>
#include <optional>
#include "nlohmann/json.hpp"
#include "tcb/span.hpp"
#include "tde/details/move_only_function.h"

namespace tde::details {

namespace transform_default {

inline bool All(int64_t global_id) {
  return true;
}

template <typename LXURecord>
inline LXURecord NoUpdate(
    std::optional<LXURecord> record,
    int64_t global_id,
    int64_t cache_id) {
  return record.value_or(LXURecord{});
};

inline void NoFetch(int64_t global_id, int64_t cache_id) {}

inline int64_t Identity(int64_t cache_id) {
  return cache_id;
}

} // namespace transform_default

template <typename T = uint32_t>
struct Bitmap {
  explicit Bitmap(int64_t num_bits);
  Bitmap(const Bitmap&) = delete;
  Bitmap(Bitmap&&) noexcept = default;

  int64_t NextFreeBit();
  void FreeBit(int64_t offset);
  bool Full() const;

  static constexpr int64_t num_bits_per_value = sizeof(T) * 8;

  const int64_t num_total_bits_;
  const int64_t num_values_;
  std::unique_ptr<T[]> values_;

  int64_t next_free_bit_;
};

template <typename LXURecord>
struct TransformerRecord {
  int64_t global_id_;
  int64_t cache_id_;
  LXURecord lxu_record_;
};

/**
 * NaiveIDTransformer
 *
 * Transform GlobalID to CacheID by naive flat hash map
 * @tparam LXURecord The extension type used for eviction strategy.
 * @tparam Bitmap The bitmap class to record the free cache ids.
 */
template <typename LXURecord, typename Bitmap = Bitmap<uint32_t>>
class NaiveIDTransformer {
 public:
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
  static constexpr std::string_view type_ = "naive";

  explicit NaiveIDTransformer(int64_t num_embedding);
  NaiveIDTransformer(const NaiveIDTransformer<LXURecord, Bitmap>&) = delete;
  NaiveIDTransformer(NaiveIDTransformer<LXURecord, Bitmap>&&) noexcept =
      default;

  static NaiveIDTransformer<LXURecord, Bitmap> Create(
      int64_t num_embedding,
      const nlohmann::json& json) {
    return NaiveIDTransformer<LXURecord, Bitmap>(num_embedding);
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
    int64_t cache_id_;
    LXURecord lxu_record_;
  };

  ska::flat_hash_map<int64_t, CacheValue> global_id2cache_value_;
  Bitmap bitmap_;
};

} // namespace tde::details

#include "tde/details/naive_id_transformer_impl.h"
