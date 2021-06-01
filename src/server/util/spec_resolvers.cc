/** Copyright 2020 Alibaba Group Holding Limited.

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
#include "common/util/logging.h"
#include "server/util/spec_resolvers.h"

// meta data
DEFINE_string(deployment, "local", "deployment mode: local, distributed");
DEFINE_string(etcd_endpoint, "http://127.0.0.1:2379", "endpoint of etcd");
DEFINE_string(etcd_prefix, "vineyard", "path prefix in etcd");
DEFINE_string(etcd_cmd, "", "path of etcd executable");
// share memory
DEFINE_string(size, "256Mi",
              "shared memory size for vineyardd, the format could be 1024M, "
              "1024000, 1G, or 1Gi");
DEFINE_int64(stream_threshold, 80,
             "memory threshold of streams (percentage of total memory)");
// ipc
DEFINE_string(socket, "/var/run/vineyard.sock", "IPC socket file location");
// rpc
DEFINE_int32(rpc_socket_port, 9600, "port to listen in rpc server");
// Kubernetes
DEFINE_bool(sync_crds, false, "Synchronize CRDs when persisting objects");

namespace vineyard {

const Resolver& Resolver::get(std::string name) {
  static auto server_resolver = ServerSpecResolver();
  static auto bulkstore_resolver = BulkstoreSpecResolver();
  static auto etcd_resolver = EtcdSpecResolver();
  static auto ipc_server_resolver = IpcSpecResolver();
  static auto rpc_server_resolver = RpcSpecResolver();

  if (name == "server") {
    return server_resolver;
  } else if (name == "bulkstore") {
    return bulkstore_resolver;
  } else if (name == "etcd") {
    return etcd_resolver;
  } else if (name == "ipcserver") {
    return ipc_server_resolver;
  } else if (name == "rpcserver") {
    return rpc_server_resolver;
  } else {
    throw std::exception();
  }
}

json EtcdSpecResolver::resolve() const {
  json spec;
  // FIXME: get from flags or env
  spec["prefix"] = FLAGS_etcd_prefix;
  spec["etcd_endpoint"] = FLAGS_etcd_endpoint;
  spec["etcd_cmd"] = FLAGS_etcd_cmd;
  return spec;
}

json BulkstoreSpecResolver::resolve() const {
  json spec;
  size_t bulkstore_limit = parseMemoryLimit(FLAGS_size);
  spec["memory_size"] = bulkstore_limit;
  spec["stream_threshold"] = FLAGS_stream_threshold;
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
  const char *start = memory_limit.c_str(),
             *end = memory_limit.c_str() + memory_limit.size();
  char* parsed_end = nullptr;
  double parse_size = std::strtod(start, &parsed_end);
  if (end == parsed_end || *parsed_end == '\0') {
    return static_cast<size_t>(parse_size);
  }
  switch (*parsed_end) {
  case 'k':
  case 'K':
    return static_cast<size_t>(parse_size * (1LL << 10));
  case 'm':
  case 'M':
    return static_cast<size_t>(parse_size * (1LL << 20));
  case 'g':
  case 'G':
    return static_cast<size_t>(parse_size * (1LL << 30));
  case 't':
  case 'T':
    return static_cast<size_t>(parse_size * (1LL << 40));
  case 'P':
  case 'p':
    return static_cast<size_t>(parse_size * (1LL << 50));
  case 'e':
  case 'E':
    return static_cast<size_t>(parse_size * (1LL << 60));
  default:
    return static_cast<size_t>(parse_size);
  }
}

json IpcSpecResolver::resolve() const {
  json spec;
  spec["socket"] = FLAGS_socket;
  return spec;
}

json RpcSpecResolver::resolve() const {
  json spec;
  spec["port"] = FLAGS_rpc_socket_port;
  return spec;
}

json ServerSpecResolver::resolve() const {
  json spec;
  spec["deployment"] = FLAGS_deployment;
  spec["sync_crds"] =
      FLAGS_sync_crds || (read_env("VINEYARD_SYNC_CRDS") == "1");
  spec["metastore_spec"] = Resolver::get("etcd").resolve();
  spec["bulkstore_spec"] = Resolver::get("bulkstore").resolve();
  spec["ipc_spec"] = Resolver::get("ipcserver").resolve();
  spec["rpc_spec"] = Resolver::get("rpcserver").resolve();
  return spec;
}

}  // namespace vineyard
