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

#ifndef SRC_SERVER_UTIL_ETCD_LAUNCHER_H_
#define SRC_SERVER_UTIL_ETCD_LAUNCHER_H_

#include <netdb.h>

#include <memory>
#include <set>
#include <string>
#include <vector>

#if defined(BUILD_VINEYARDD_ETCD)

#include "boost/process/child.hpp"  // IWYU pragma: keep
#include "etcd/Client.hpp"

#include "common/util/status.h"

namespace vineyard {

class EtcdLauncher {
 public:
  explicit EtcdLauncher(const json& etcd_spec);
  ~EtcdLauncher();

  Status LaunchEtcdServer(std::unique_ptr<etcd::Client>& etcd_client,
                          std::string& sync_lock);

  // Check if the etcd server available, return True if succeed, otherwise
  // False.
  static bool probeEtcdServer(std::unique_ptr<etcd::Client>& etcd_client,
                              std::string const& key);

 private:
  Status removeMember(std::string& member_id, int max_retries = 5);

  Status parseEndpoint();

  std::string GetMemberID() { return etcd_member_id_; }

  Status addMember(std::string& member_name, std::string& peer_endpoint,
                   const std::string& etcd_endpoints, int max_retries = 5);

  std::vector<json> listMembers(const std::string& etcd_endpoints);
  std::vector<std::string> listPeerURLs(const std::vector<json>& members);
  std::vector<std::string> listClientURLs(const std::vector<json>& members);

  std::vector<std::string> listMembersName(const std::vector<json>& members);

  std::string generateMemberName(
      std::vector<std::string> const& existing_members_name);

  Status UpdateEndpoint();

  Status initHostInfo();

  const json etcd_spec_;
  std::string endpoint_host_;
  std::string etcd_data_dir_;
  int endpoint_port_;
  std::set<std::string> local_hostnames_;
  std::set<std::string> local_ip_addresses_;

  std::string etcdctl_cmd_;
  std::string etcd_member_id_;
  std::string etcd_endpoints_;

  std::unique_ptr<boost::process::child> etcd_proc_;

  friend class EtcdMetaService;
};

}  // namespace vineyard

#endif  // BUILD_VINEYARDD_ETCD

#endif  // SRC_SERVER_UTIL_ETCD_LAUNCHER_H_
