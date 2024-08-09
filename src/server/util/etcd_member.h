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

#ifndef SRC_SERVER_UTIL_ETCD_MEMBER_H_
#define SRC_SERVER_UTIL_ETCD_MEMBER_H_

#include <memory>
#include <string>
#include <vector>

#include "etcd/Client.hpp"

#include "common/util/status.h"

namespace vineyard {

Status removeMember(std::unique_ptr<etcd::Client>& etcd_client,
                    const uint64_t& member_id, int max_retries = 5);

uint64_t findMemberID(std::unique_ptr<etcd::Client>& etcd_client,
                      const std::string& peer_urls);

Status addMember(std::unique_ptr<etcd::Client>& etcd_client,
                 const std::string& peer_endpoint, bool is_learner = false,
                 int max_retries = 5);

std::vector<json> listMembers(std::unique_ptr<etcd::Client>& etcd_client);

std::vector<json> listHealthyMembers(const std::vector<json>& members);

std::vector<std::string> listPeerURLs(const std::vector<json>& members);

std::vector<std::string> listClientURLs(const std::vector<json>& members);

std::vector<std::string> listMembersName(const std::vector<json>& members);

}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_ETCD_MEMBER_H_
