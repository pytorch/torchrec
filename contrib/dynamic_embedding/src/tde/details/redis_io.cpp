#include "redis_io.h"
#include "tde/details/io_registry.h"
#include "tde/details/redis_io_v1.h"

namespace tde::details {

void RegisterRedisIO() {
  auto& reg = IORegistry::Instance();

  {
    IOProvider provider{};
    provider.type_ = "redis";
    provider.Initialize = +[](const char* cfg) -> void* {
      auto opt = redis_v1::Option::Parse(cfg);
      return new redis_v1::RedisV1(opt);
    };
    provider.Finalize =
        +[](void* inst) { delete reinterpret_cast<redis_v1::RedisV1*>(inst); };
    provider.Pull = +[](void* inst, IOPullParameter param) {
      reinterpret_cast<redis_v1::RedisV1*>(inst)->Pull(param);
    };
    provider.Push = +[](void* inst, IOPushParameter param) {
      reinterpret_cast<redis_v1::RedisV1*>(inst)->Push(param);
    };
    reg.Register(provider);
  }
}
} // namespace tde::details
