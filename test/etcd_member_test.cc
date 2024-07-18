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

#include <map>
#include <string>

#include "boost/process.hpp"  // IWYU pragma: keep

#include "client/client.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int getStartedMembersSize(const std::string& etcdctl_cmd,
                          const std::string& etcd_endpoints) {
  int members_size = 0;
  boost::process::ipstream output_stream;
  std::error_code ec;

  std::unique_ptr<boost::process::child> etcdctl_proc_ =
      std::make_unique<boost::process::child>(
          etcdctl_cmd, "member", "list", "--endpoints=" + etcd_endpoints,
          "--write-out=json", boost::process::std_out > output_stream,
          boost::process::std_err > stderr, ec);

  if (!etcdctl_proc_ || ec) {
    LOG(ERROR) << "Failed to run etcdctl member list: " << ec.message();
    return members_size;
  }

  std::stringstream buffer;
  std::string line;
  while (std::getline(output_stream, line)) {
    buffer << line << '\n';
  }

  std::string output = buffer.str();
  auto result = json::parse(output);
  for (const auto& member : result["members"]) {
    if (member.find("clientURLs") == member.end()) {
      continue;
    }
    members_size++;
  }
  return members_size;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf(
        "usage ./etcd_member_test <ipc_socket> <etcdctl_path> "
        "<etcd_endpoints>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);
  std::string etcdctl_cmd = std::string(argv[2]);
  std::string etcd_endpoints = std::string(argv[3]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  std::map<InstanceID, json> cluster;
  VINEYARD_CHECK_OK(client.ClusterInfo(cluster));
  CHECK(!cluster.empty());

  int members_size = getStartedMembersSize(etcdctl_cmd, etcd_endpoints);
  CHECK_EQ(members_size, cluster.size());
  LOG(INFO) << "Passed etcd member tests...";

  client.Disconnect();

  return 0;
}
