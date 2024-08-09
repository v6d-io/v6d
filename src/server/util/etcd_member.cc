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

#include "server/util/etcd_member.h"

#include <memory>
#include <string>
#include <vector>
#include "etcd/Response.hpp"

#if defined(BUILD_VINEYARDD_ETCD)

#include "boost/process.hpp"  // IWYU pragma: keep

#include "common/util/logging.h"  // IWYU pragma: keep
#include "common/util/status.h"

namespace vineyard {

json member_to_json(const etcdv3::Member& m) {
  json j = json{};

  if (m.get_id() != 0) {
    j["ID"] = m.get_id();
  }

  if (!m.get_name().empty()) {
    j["name"] = m.get_name();
  }

  if (!m.get_peerURLs().empty()) {
    j["peerURLs"] = m.get_peerURLs();
  }

  if (!m.get_clientURLs().empty()) {
    j["clientURLs"] = m.get_clientURLs();
  }

  j["isLearner"] = m.get_learner();
  return j;
}

Status addMember(std::unique_ptr<etcd::Client>& etcd_client,
                 const std::string& peer_endpoint, bool is_learner,
                 int max_retries) {
  int retries = 0;
  while (retries < max_retries) {
    etcd::Response res =
        etcd_client->add_member(peer_endpoint, is_learner).get();
    if (!res.is_ok()) {
      LOG(ERROR) << "Failed to add etcd member: " << res.error_message();
      retries += 1;
      sleep(1);
      continue;
    } else {
      return Status::OK();
    }
  }
  return Status::EtcdError("Failed to add etcd member after " +
                           std::to_string(max_retries) + " retries");
}

Status removeMember(std::unique_ptr<etcd::Client>& etcd_client,
                    const uint64_t& member_id, int max_retries) {
  int retries = 0;

  auto members = listMembers(etcd_client);
  bool member_exist = false;
  for (const auto& member : members) {
    if (member["ID"].get<uint64_t>() == member_id) {
      member_exist = true;
      break;
    }
  }
  if (!member_exist) {
    LOG(INFO) << "The member id " << std::to_string(member_id)
              << " has been removed";
    return Status::OK();
  }

  if (members.size() == 1) {
    LOG(INFO) << "The last member can not be removed";
    return Status::OK();
  }

  while (retries < max_retries) {
    etcd::Response res = etcd_client->remove_member(member_id).get();
    if (!res.is_ok()) {
      LOG(ERROR) << "Failed to remove etcd member: " << res.error_message();
      retries += 1;
      sleep(1);
      continue;
    } else {
      return Status::OK();
    }
  }
  return Status::EtcdError("Failed to remove etcd member " +
                           std::to_string(member_id) + " after " +
                           std::to_string(max_retries) + " retries");
}

uint64_t findMemberID(std::unique_ptr<etcd::Client>& etcd_client,
                      const std::string& peer_urls) {
  uint64_t member_id = 0;

  auto members = listMembers(etcd_client);
  for (const auto& member : members) {
    auto peers = member["peerURLs"];
    for (const auto& peer : peers) {
      if (peer.get<std::string>() == peer_urls) {
        member_id = member["ID"].get<uint64_t>();
        break;
      }
    }
  }
  LOG(INFO) << "Find member id: " << member_id << " for peer urls "
            << peer_urls;
  return member_id;
}

std::vector<json> listMembers(std::unique_ptr<etcd::Client>& etcd_client) {
  std::vector<json> members;

  etcd::Response res = etcd_client->list_member().get();
  if (!res.is_ok()) {
    LOG(ERROR) << "Failed to list etcd members: " << res.error_message();
    return members;
  }

  for (const auto& member : res.members()) {
    json member_json = member_to_json(member);
    members.emplace_back(member_json);
  }

  return members;
}

std::vector<json> listHealthyMembers(const std::vector<json>& members) {
  std::vector<json> healthy_members;
  for (const auto& member : members) {
    if (member.find("clientURLs") == member.end()) {
      continue;
    }
    healthy_members.emplace_back(member);
  }
  return healthy_members;
}

std::vector<std::string> listPeerURLs(const std::vector<json>& members) {
  std::vector<std::string> peerURLs;

  for (const auto& member : members) {
    if (member.find("peerURLs") == member.end()) {
      continue;
    }
    auto peers = member["peerURLs"];
    for (const auto& peer : peers) {
      peerURLs.emplace_back(peer.get<std::string>());
    }
  }
  return peerURLs;
}

std::vector<std::string> listClientURLs(const std::vector<json>& members) {
  std::vector<std::string> clientURLs;

  for (const auto& member : members) {
    if (member.find("clientURLs") == member.end()) {
      continue;
    }
    auto clients = member["clientURLs"];
    for (const auto& client : clients) {
      clientURLs.emplace_back(client.get<std::string>());
    }
  }
  return clientURLs;
}

std::vector<std::string> listMembersName(const std::vector<json>& members) {
  std::vector<std::string> members_name;
  for (const auto& member : members) {
    if (member.find("name") == member.end()) {
      continue;
    }
    auto name = member["name"];
    members_name.emplace_back(name.get<std::string>());
  }
  return members_name;
}

}  // namespace vineyard

#endif  // BUILD_VINEYARDD_ETCD
