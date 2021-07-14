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

#include "server/util/etcd_launcher.h"

#include <netdb.h>

#include <sys/types.h>

#include <memory>
#include <string>
#include <vector>

#include "common/util/env.h"

namespace vineyard {

constexpr int max_probe_retries = 15;

bool validate_advertise_hostname(std::string const& hostname,
                                 std::string const& port_string) {
  struct addrinfo hints = {}, *addrs = nullptr;
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  if (getaddrinfo(hostname.c_str(), port_string.c_str(), &hints, &addrs) != 0) {
    return false;
  } else {
    freeaddrinfo(addrs);
    return true;
  }
}

Status EtcdLauncher::LaunchEtcdServer(
    std::unique_ptr<etcd::Client>& etcd_client, std::string& sync_lock,
    std::unique_ptr<boost::process::child>& etcd_proc) {
  std::string const& etcd_endpoint =
      etcd_spec_["etcd_endpoint"].get_ref<std::string const&>();
  etcd_client.reset(new etcd::Client(etcd_endpoint));

  if (probeEtcdServer(etcd_client, sync_lock)) {
    return Status::OK();
  }

  LOG(INFO) << "Starting the etcd server";

  // resolve etcd binary
  std::string etcd_cmd = etcd_spec_["etcd_cmd"].get_ref<std::string const&>();
  if (etcd_cmd.empty()) {
    setenv("LC_ALL", "C", 1);  // makes boost's path works as expected.
    etcd_cmd = boost::process::search_path("etcd").string();
  }
  LOG(INFO) << "Found etcd at: " << etcd_cmd;

  parseEndpoint();
  initHostInfo();

  std::vector<std::string> local_host_names = {"localhost", "127.0.0.1",
                                               "0.0.0.0"};

  bool try_launch = false;
  if (local_hostnames_.find(endpoint_host_) != local_hostnames_.end() ||
      local_ip_addresses_.find(endpoint_host_) != local_ip_addresses_.end()) {
    try_launch = true;
  }

  if (!try_launch) {
    LOG(INFO) << "Will not launch an etcd instance.";
    int retries = 0;
    while (retries < max_probe_retries) {
      if (probeEtcdServer(etcd_client, sync_lock)) {
        break;
      }
      retries += 1;
      sleep(1);
    }
    if (retries >= max_probe_retries) {
      return Status::EtcdError(
          "Etcd has been launched but failed to connect to it");
    } else {
      return Status::OK();
    }
  }

  std::string host_to_advertise;
  if (host_to_advertise.empty()) {
    for (auto const& h : local_hostnames_) {
      if (h != "localhost" &&
          validate_advertise_hostname(h, std::to_string(endpoint_port_))) {
        host_to_advertise = h;
        break;
      }
    }
  }
  if (host_to_advertise.empty()) {
    for (auto const& h : local_ip_addresses_) {
      if (h != "127.0.0.1" && h != "0.0.0.0" &&
          validate_advertise_hostname(h, std::to_string(endpoint_port_))) {
        host_to_advertise = h;
        break;
      }
    }
  }
  if (host_to_advertise.empty()) {
    host_to_advertise = "127.0.0.1";
  }

  std::string client_endpoint =
      "http://" + host_to_advertise + ":" + std::to_string(endpoint_port_);
  std::string peer_endpoint =
      "http://" + host_to_advertise + ":" + std::to_string(endpoint_port_ + 1);

  std::vector<std::string> args;
  args.emplace_back("--listen-client-urls");
  args.emplace_back("http://0.0.0.0:" + std::to_string(endpoint_port_));
  args.emplace_back("--advertise-client-urls");
  args.emplace_back(client_endpoint);
  args.emplace_back("--listen-peer-urls");
  args.emplace_back("http://0.0.0.0:" + std::to_string(endpoint_port_ + 1));
  args.emplace_back("--initial-cluster");
  args.emplace_back("default=" + peer_endpoint);
  args.emplace_back("--initial-advertise-peer-urls");
  args.emplace_back(peer_endpoint);

  if (VLOG_IS_ON(10)) {
    args.emplace_back("--log-level");
    args.emplace_back("debug");
  }

  auto env = boost::this_process::environment();
  std::error_code ec;
  etcd_proc = std::make_unique<boost::process::child>(
      etcd_cmd, boost::process::args(args), boost::process::std_out > stdout,
      boost::process::std_err > stderr, env, ec);
  if (ec) {
    LOG(ERROR) << "Failed to launch etcd: " << ec.message();
    return Status::EtcdError("Failed to launch etcd: " + ec.message());
  } else {
    LOG(INFO) << "etcd launched: pid = " << etcd_proc->id() << ", listen on "
              << endpoint_port_;

    int retries = 0;
    std::error_code err;
    while (etcd_proc && etcd_proc->running(err) && !err &&
           retries < max_probe_retries) {
      etcd_client.reset(new etcd::Client(etcd_endpoint));
      if (probeEtcdServer(etcd_client, sync_lock)) {
        break;
      }
      retries += 1;
      sleep(1);
    }
    if (!etcd_proc) {
      return Status::IOError(
          "Failed to wait until etcd ready: operation has been interrupted");
    } else if (err) {
      return Status::IOError("Failed to check the process status: " +
                             err.message());
    } else if (retries >= max_probe_retries) {
      return Status::EtcdError(
          "Etcd has been launched but failed to connect to it");
    } else {
      return Status::OK();
    }
  }
}

void EtcdLauncher::parseEndpoint() {
  std::string const& etcd_endpoint =
      etcd_spec_["etcd_endpoint"].get_ref<std::string const&>();
  size_t port_pos = etcd_endpoint.find_last_of(':');
  endpoint_port_ = std::stoi(
      etcd_endpoint.substr(port_pos + 1, etcd_endpoint.size() - port_pos - 1));
  size_t prefix = etcd_endpoint.find_last_of('/');
  endpoint_host_ = etcd_endpoint.substr(prefix + 1, port_pos - prefix - 1);
  return;
}

void EtcdLauncher::initHostInfo() {
  local_hostnames_.emplace("localhost");
  local_ip_addresses_.emplace("127.0.0.1");
  local_ip_addresses_.emplace("0.0.0.0");
  std::string hostname_value = get_hostname();
  local_hostnames_.emplace(hostname_value);
  struct hostent* host_entry = gethostbyname(hostname_value.c_str());
  if (host_entry == nullptr) {
    LOG(ERROR) << "Failed in gethostbyname: " << hstrerror(h_errno);
    return;
  }
  {
    size_t index = 0;
    while (true) {
      if (host_entry->h_aliases[index] == nullptr) {
        break;
      }
      local_hostnames_.emplace(host_entry->h_aliases[index++]);
    }
  }
  {
    size_t index = 0;
    while (true) {
      if (host_entry->h_addr_list[index] == nullptr) {
        break;
      }
      local_ip_addresses_.emplace(
          inet_ntoa(*((struct in_addr*) host_entry->h_addr_list[index++])));
    }
  }
}

bool EtcdLauncher::probeEtcdServer(std::unique_ptr<etcd::Client>& etcd_client,
                                   std::string const& key) {
  // probe: as a 1-limit range request
  return etcd_client && etcd_client->ls(key, 1).get().is_ok();
}

}  // namespace vineyard
