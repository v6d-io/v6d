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

#include "server/util/etcdctl.h"

#include <memory>
#include <string>
#include <vector>

#if defined(BUILD_VINEYARDD_ETCD)

#include "boost/process.hpp"  // IWYU pragma: keep

#include "common/util/logging.h"  // IWYU pragma: keep
#include "common/util/status.h"

namespace vineyard {

Status Etcdctl::addMember(const std::string& member_name,
                          const std::string& peer_endpoint,
                          const std::string& etcd_endpoints, int max_retries) {
  int retries = 0;
  while (retries < max_retries) {
    std::error_code ec;
    std::unique_ptr<boost::process::child> etcdctl_proc_ =
        std::make_unique<boost::process::child>(
            etcdctl_cmd_, "member", "add", member_name,
            "--peer-urls=" + peer_endpoint, "--endpoints=" + etcd_endpoints,
            "--command-timeout=30s", "--keepalive-timeout=30s",
            "--dial-timeout=20s", boost::process::std_out > stdout,
            boost::process::std_err > stderr, ec);
    if (!etcdctl_proc_) {
      LOG(ERROR) << "Failed to start etcdctl";
      return Status::EtcdError("Failed to start etcdctl");
    }
    if (ec) {
      LOG(ERROR) << "Failed to add etcd member: " << ec.message();
      return Status::EtcdError("Failed to add etcd member: " + ec.message());
    }

    // wait for the etcdctl to finish the add member operation
    etcdctl_proc_->wait();
    int exit_code = etcdctl_proc_->exit_code();

    if (exit_code != 0) {
      LOG(ERROR) << "Failed to add etcd member: exit code: " << exit_code
                 << ", retries: " << retries << "/" << max_retries;
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

Status Etcdctl::removeMember(const std::string& member_id,
                             const std::string& etcd_endpoints,
                             int max_retries) {
  int retries = 0;

  auto members = listMembers(etcd_endpoints);
  bool member_exist = false;
  for (const auto& member : members) {
    std::stringstream ss;
    ss << std::hex << member["ID"].get<uint64_t>();
    if (ss.str() == member_id) {
      member_exist = true;
      break;
    }
  }
  if (!member_exist) {
    LOG(INFO) << "The member id " << member_id << " has been removed";
    return Status::OK();
  }

  if (members.size() == 1) {
    LOG(INFO) << "The last member can not be removed";
    return Status::OK();
  }

  while (retries < max_retries) {
    std::error_code ec;
    std::unique_ptr<boost::process::child> etcdctl_proc_ =
        std::make_unique<boost::process::child>(
            etcdctl_cmd_, "member", "remove", member_id,
            "--endpoints=" + etcd_endpoints, boost::process::std_out > stdout,
            boost::process::std_err > stderr, ec);
    if (!etcdctl_proc_) {
      LOG(ERROR) << "Failed to start etcdctl";
      return Status::EtcdError("Failed to start etcdctl");
    }
    if (ec) {
      LOG(ERROR) << "Failed to remove etcd member: " << ec.message();
      return Status::EtcdError("Failed to remove etcd member: " + ec.message());
    }
    // wait for the etcdctl to finish the remove member operation
    etcdctl_proc_->wait();
    int exit_code = etcdctl_proc_->exit_code();

    if (exit_code != 0) {
      LOG(ERROR) << "Failed to remove etcd member: exit code: " << exit_code
                 << ", retries: " << retries << "/" << max_retries;
      retries += 1;
      sleep(1);
      continue;
    } else {
      return Status::OK();
    }
  }
  return Status::EtcdError("Failed to remove etcd member after " +
                           std::to_string(max_retries) + " retries");
}

std::string Etcdctl::findMemberID(const std::string& member_name,
                                  const std::string& etcd_endpoints) {
  std::string member_id = "";
  auto members = listMembers(etcd_endpoints);
  for (const auto& member : members) {
    if (member["name"].get<std::string>() == member_name) {
      std::stringstream ss;
      ss << std::hex << member["ID"].get<uint64_t>();
      member_id = ss.str();
      break;
    }
  }
  return member_id;
}

std::vector<json> Etcdctl::listMembers(const std::string& etcd_endpoints) {
  std::vector<json> members;
  boost::process::ipstream output_stream;
  std::error_code ec;

  std::unique_ptr<boost::process::child> etcdctl_proc_ =
      std::make_unique<boost::process::child>(
          etcdctl_cmd_, "member", "list", "--endpoints=" + etcd_endpoints,
          "--write-out=json", boost::process::std_out > output_stream,
          boost::process::std_err > stderr, ec);

  if (!etcdctl_proc_) {
    LOG(ERROR) << "Failed to start etcdctl";
    return members;
  }
  if (ec) {
    LOG(ERROR) << "Failed to list etcd members: " << ec.message();
    return members;
  }

  std::stringstream buffer;
  std::string line;
  while (std::getline(output_stream, line)) {
    buffer << line << '\n';
  }

  std::string output = buffer.str();
  auto result = json::parse(output);
  for (const auto& member : result["members"]) {
    members.emplace_back(member);
  }
  return members;
}

std::vector<std::string> Etcdctl::listPeerURLs(
    const std::vector<json>& members) {
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

std::vector<std::string> Etcdctl::listClientURLs(
    const std::vector<json>& members) {
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

std::vector<std::string> Etcdctl::listMembersName(
    const std::vector<json>& members) {
  std::vector<std::string> members_name;
  for (const auto& member : members) {
    auto name = member["name"];
    members_name.emplace_back(name.get<std::string>());
  }
  return members_name;
}

}  // namespace vineyard

#endif  // BUILD_VINEYARDD_ETCD
