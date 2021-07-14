/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#ifndef SRC_SERVER_UTIL_ETCD_LAUNCHER_H_
#define SRC_SERVER_UTIL_ETCD_LAUNCHER_H_

#include <netdb.h>

#include <memory>
#include <set>
#include <string>

#include "boost/filesystem.hpp"
#include "boost/process.hpp"
#include "etcd/Client.hpp"

#include "common/util/logging.h"
#include "common/util/status.h"

namespace vineyard {
class EtcdLauncher {
 public:
  explicit EtcdLauncher(const json& etcd_spec) : etcd_spec_(etcd_spec) {}

  Status LaunchEtcdServer(std::unique_ptr<etcd::Client>& etcd_client,
                          std::string& sync_lock,
                          std::unique_ptr<boost::process::child>& etcd_proc);

  // Check if the etcd server available, return True if succeed, otherwise
  // False.
  static bool probeEtcdServer(std::unique_ptr<etcd::Client>& etcd_client,
                              std::string const& key);

 private:
  void parseEndpoint();

  void initHostInfo();

  const json etcd_spec_;
  std::string endpoint_host_;
  int endpoint_port_;
  std::set<std::string> local_hostnames_;
  std::set<std::string> local_ip_addresses_;
};
}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_ETCD_LAUNCHER_H_
