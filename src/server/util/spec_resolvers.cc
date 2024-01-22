/** Copyright 2020-2023 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// #include <cstdlib>
#include <exception>

#include "gflags/gflags.h"

#include "common/util/env.h"
#include "common/util/logging.h"  // IWYU pragma: keep
#include "common/util/macros.h"
#include "server/util/spec_resolvers.h"

namespace vineyard {

// deployment
DEFINE_string(deployment, "local", "Deployment mode: 'local', 'distributed'");

// meta data
DEFINE_string(meta,
#if defined(BUILD_VINEYARDD_ETCD)
              "etcd",
#else
              "local",
#endif
              "Metadata storage, can be one of: 'local'"
#if defined(BUILD_VINEYARDD_ETCD)
              ", 'etcd'"
#endif
#if defined(BUILD_VINEYARDD_REDIS)
              " and 'redis'"
#endif
);
DEFINE_int64(meta_timeout, 60 /* 1 minutes */,
             "Timeout period before waiting the metadata service to be ready, "
             "in seconds");
#if defined(BUILD_VINEYARDD_ETCD)
DEFINE_string(etcd_endpoint, "http://127.0.0.1:2379", "endpoint of etcd");
DEFINE_string(etcd_prefix, "vineyard", "metadata path prefix in etcd");
DEFINE_string(etcd_cmd, "", "path of etcd executable");
DEFINE_string(etcd_data_dir, "default.etcd", "path of etcd's data directory");
#endif

#if defined(BUILD_VINEYARDD_REDIS)
DEFINE_string(redis_endpoint, "redis://127.0.0.1:6379", "endpoint of redis");
DEFINE_string(redis_prefix, "vineyard", "metadata path prefix in redis");
DEFINE_string(redis_cmd, "", "path of redis executable");
#endif

// share memory
DEFINE_string(size, "",
              "shared memory size for vineyardd, the format could be 1024M, "
              "1024000, 1G, or 1Gi.\nDefaults to empty, means not limited");
DEFINE_string(allocator,
#if defined(DEFAULT_ALLOCATOR)
              VINEYARD_TO_STRING(DEFAULT_ALLOCATOR),
#else
              "dlmalloc",
#endif
              "allocator for shared memory allocation, can be one of: "
              "'dlmalloc', 'mimalloc'");

DEFINE_int64(stream_threshold, 80,
             "memory threshold of streams (percentage of total memory)");

// shared memory spilling
DEFINE_string(
    spill_path, "",
    "path to spill temporary files, if not set, spilling will be disabled");
DEFINE_double(spill_lower_rate, 0.3,
              "low watermark of triggering memory spilling");
DEFINE_double(spill_upper_rate, 0.8,
              "high watermark of triggering memory spilling");

// ipc
DEFINE_string(
    socket, "",
    "IPC socket file location. Defaults to '/var/run/vineyard.sock' for root "
    "users and '~/.local/vineyard/vineyard.sock' for non-root users.");

// rpc
DEFINE_bool(rpc, true, "Enable RPC service by default");
DEFINE_int32(rpc_socket_port, 9600, "port to listen in rpc server");

// Kubernetes
DEFINE_bool(sync_crds, false, "Synchronize CRDs when persisting objects");

// IO: spill and migration
DEFINE_bool(compression, true, "Compress before migration or spilling");

// metrics and prometheus
DEFINE_bool(prometheus, false,
            "Whether to print metrics for prometheus or not");
DEFINE_bool(metrics, false,
            "Alias for --prometheus, and takes precedence over --prometheus");

// core dump
DEFINE_bool(coredump, false, "Enable core dump when been aborted");

// auth
DEFINE_string(htpasswd, "", "Location of htpasswd database for authentication");

const Resolver& Resolver::get(std::string name) {
  static auto server_resolver = ServerSpecResolver();
  static auto bulkstore_resolver = BulkstoreSpecResolver();
  static auto metastore_resolver = MetaStoreSpecResolver();
  static auto ipc_server_resolver = IpcSpecResolver();
  static auto rpc_server_resolver = RpcSpecResolver();

  if (name == "server") {
    return server_resolver;
  } else if (name == "bulkstore") {
    return bulkstore_resolver;
  } else if (name == "metastore") {
    return metastore_resolver;
  } else if (name == "ipcserver") {
    return ipc_server_resolver;
  } else if (name == "rpcserver") {
    return rpc_server_resolver;
  } else {
    throw std::exception();
  }
}

json MetaStoreSpecResolver::resolve() const {
  json spec;
  // resolve for meta
  spec["meta"] = FLAGS_meta;
  spec["meta_timeout"] = FLAGS_meta_timeout;

  // resolve for etcd
#if defined(BUILD_VINEYARDD_ETCD)
  spec["etcd_prefix"] = FLAGS_etcd_prefix;
  spec["etcd_endpoint"] = FLAGS_etcd_endpoint;
  spec["etcd_cmd"] = FLAGS_etcd_cmd;
  spec["etcd_data_dir"] = FLAGS_etcd_data_dir;
#endif

  // resolve for redis
#if defined(BUILD_VINEYARDD_REDIS)
  spec["redis_prefix"] = FLAGS_redis_prefix;
  spec["redis_endpoint"] = FLAGS_redis_endpoint;
  spec["redis_cmd"] = FLAGS_redis_cmd;
#endif
  return spec;
}

json BulkstoreSpecResolver::resolve() const {
  json spec;
  size_t bulkstore_limit = parseMemoryLimit(FLAGS_size);
  spec["memory_size"] = bulkstore_limit;
  spec["allocator"] = FLAGS_allocator;
  spec["stream_threshold"] = FLAGS_stream_threshold;
  spec["spill_path"] = FLAGS_spill_path;
  spec["spill_lower_bound_rate"] = FLAGS_spill_lower_rate;
  spec["spill_upper_bound_rate"] = FLAGS_spill_upper_rate;
  return spec;
}

size_t BulkstoreSpecResolver::parseMemoryLimit(
    std::string const& memory_limit) const {
  // Parse human-readable size. Note that any extra character that follows a
  // valid sequence will be ignored.
  //
  // You can express memory as a plain integer or as a fixed-point number using
  // one of these suffixes: E, P, T, G, M, K. You can also use the power-of-two
  // equivalents: Ei, Pi, Ti, Gi, Mi, Ki.
  //
  // For example, the following represent roughly the same value:
  //
  // 128974848, 129k, 129M, 123Mi, 1G, 10Gi, ...
  const int64_t parsed_bytes = parse_memory_size(memory_limit);
  if (parsed_bytes <= 0) {
    // defaults to the physical memory size
    const int64_t system_memory_limit = read_physical_memory_limit();
    if (system_memory_limit <= 0) {
      LOG(WARNING) << "Failed to resolve the system physical memory size. "
                      "Defaults the limit to 16Gi.";
      LOG(WARNING) << "Try specify the memory size manually with the `--size` "
                      "command line argument.";
      return 16L * 1024 * 1024 * 1024;
    } else {
      return system_memory_limit;
    }
  } else {
    return parsed_bytes;
  }
}

json IpcSpecResolver::resolve() const {
  json spec;
  spec["socket"] = FLAGS_socket;
  return spec;
}

json RpcSpecResolver::resolve() const {
  json spec;
  spec["rpc"] = FLAGS_rpc;
  spec["port"] = FLAGS_rpc_socket_port;
  return spec;
}

json ServerSpecResolver::resolve() const {
  json spec;
  spec["deployment"] = FLAGS_deployment;
  spec["compression"] = FLAGS_compression;
  spec["sync_crds"] =
      FLAGS_sync_crds || (read_env("VINEYARD_SYNC_CRDS") == "1");
  spec["metastore_spec"] = Resolver::get("metastore").resolve();
  spec["bulkstore_spec"] = Resolver::get("bulkstore").resolve();
  spec["ipc_spec"] = Resolver::get("ipcserver").resolve();
  spec["rpc_spec"] = Resolver::get("rpcserver").resolve();
  spec["htpasswd"] = FLAGS_htpasswd;
  return spec;
}

}  // namespace vineyard
