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

ptree EtcdSpecResolver::resolve() const {
  ptree spec;
  // FIXME: get from flags or env
  spec.put("prefix", FLAGS_etcd_prefix + ".");
  spec.put("etcd_endpoint", FLAGS_etcd_endpoint);
  spec.put("etcd_cmd", FLAGS_etcd_cmd);
  return spec;
}

ptree BulkstoreSpecResolver::resolve() const {
  ptree spec;
  size_t bulkstore_limit = parseMemoryLimit(FLAGS_size);
  spec.put("memory_size", bulkstore_limit);
  spec.put("stream_threshold", std::to_string(FLAGS_stream_threshold));
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

ptree IpcSpecResolver::resolve() const {
  ptree spec;
  spec.put("socket", FLAGS_socket);
  return spec;
}

ptree RpcSpecResolver::resolve() const {
  ptree spec;
  spec.put("port", FLAGS_rpc_socket_port);
  return spec;
}

ptree ServerSpecResolver::resolve() const {
  ptree spec;
  spec.put("deployment", FLAGS_deployment);
  spec.add_child("metastore_spec", Resolver::get("etcd").resolve());
  spec.add_child("bulkstore_spec", Resolver::get("bulkstore").resolve());
  spec.add_child("ipc_spec", Resolver::get("ipcserver").resolve());
  spec.add_child("rpc_spec", Resolver::get("rpcserver").resolve());
  return spec;
}

}  // namespace vineyard
