/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#ifndef SRC_SERVER_UTIL_SPEC_RESOLVERS_H_
#define SRC_SERVER_UTIL_SPEC_RESOLVERS_H_

#include <string>

#include "common/util/flags.h"
#include "common/util/json.h"

namespace vineyard {

// Whether to print metrics for prometheus or not, default value is false.
DECLARE_bool(prometheus);
DECLARE_bool(metrics);

/**
 * @brief Resolver is the base class of different kinds of
 * specification resolvers adopted in vineyard
 *
 */
class Resolver {
 public:
  static const Resolver& get(std::string name);
  virtual json resolve() const = 0;
};

/**
 * @brief MetaStoreSpecResolver aims to resolve the metastore specification
 *
 */
class MetaStoreSpecResolver : public Resolver {
 public:
  json resolve() const;
};

/**
 * @brief BulkstoreSpecResolver resolves the bulkstore specification
 *
 */
class BulkstoreSpecResolver : public Resolver {
 public:
  json resolve() const;

 private:
  size_t parseMemoryLimit(std::string const& memory_limit) const;
};

/**
 * @brief IpcSpecResolver is designed for resolving the specification of
 * inter-process communication
 *
 */
class IpcSpecResolver : public Resolver {
 public:
  json resolve() const;
};

/**
 * @brief RpcSpecResolver is designed for resolving the specification of remote
 * procedure call
 *
 */
class RpcSpecResolver : public Resolver {
 public:
  json resolve() const;
};

/**
 * @brief ServerSpecResolver resolves the server specification
 *
 */
class ServerSpecResolver : public Resolver {
 public:
  json resolve() const;
};

}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_SPEC_RESOLVERS_H_
