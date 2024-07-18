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

#ifndef SRC_SERVER_UTIL_ETCDCTL_H_
#define SRC_SERVER_UTIL_ETCDCTL_H_

#include <string>
#include <vector>

#include "common/util/status.h"

namespace vineyard {

class Etcdctl {
 public:
  explicit Etcdctl(const std::string& etcdctl_cmd)
      : etcdctl_cmd_(etcdctl_cmd) {}

  Status removeMember(const std::string& member_id,
                      const std::string& etcd_endpoints, int max_retries = 5);

  std::string findMemberID(const std::string& peer_urls,
                           const std::string& etcd_endpoints);

  bool checkMemberStatus(const std::string& client_endpoint);

  Status addMember(const std::string& member_name,
                   const std::string& peer_endpoint,
                   const std::string& etcd_endpoints, int max_retries = 5);

  std::vector<json> listMembers(const std::string& etcd_endpoints);

  std::vector<json> listHealthyMembers(const std::vector<json>& members);

  std::vector<std::string> listPeerURLs(const std::vector<json>& members);

  std::vector<std::string> listClientURLs(const std::vector<json>& members);

  std::vector<std::string> listMembersName(const std::vector<json>& members);

 private:
  std::string etcdctl_cmd_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_ETCDCTL_H_
