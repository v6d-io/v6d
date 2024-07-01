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

#include "server/util/etcd_launcher.h"

#include <netdb.h>
#include <sys/types.h>

#include <memory>
#include <string>
#include <vector>
#include "common/util/status.h"

#if defined(BUILD_VINEYARDD_ETCD)

#include "boost/process.hpp"  // IWYU pragma: keep
#include "gulrak/filesystem.hpp"

#include "common/util/asio.h"  // IWYU pragma: keep
#include "common/util/env.h"
#include "common/util/logging.h"  // IWYU pragma: keep

namespace vineyard {

constexpr int max_probe_retries = 15;
constexpr int first_probe_retries = 5;

static bool validate_advertise_hostname(std::string& ipaddress,
                                        std::string const& hostname) {
  struct addrinfo hints = {}, *addrs = nullptr;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  // hints.ai_protocol = IPPROTO_TCP;
  hints.ai_flags |= AI_CANONNAME;

  if (getaddrinfo(hostname.c_str(), NULL, &hints, &addrs) != 0) {
    return false;
  }

  struct addrinfo* addr = addrs;
  // 16 should be ok, but we leave more buffer to keep it safer.
  char ipaddr[32] = {'\0'};
  // see also:
  // https://gist.github.com/jirihnidek/bf7a2363e480491da72301b228b35d5d
  while (addr != nullptr) {
    if (addr->ai_family == AF_INET) {
      inet_ntop(
          addr->ai_family,
          &(reinterpret_cast<struct sockaddr_in*>(addr->ai_addr))->sin_addr,
          ipaddr, 32);
      ipaddress = ipaddr;
      break;
    }
    addr = addr->ai_next;
  }
  freeaddrinfo(addrs);
  return true;
}

static bool check_port_in_use(boost::asio::io_context& context,
                              unsigned short port) {
  boost::system::error_code ec;

  boost::asio::ip::tcp::acceptor acceptor(context);
  acceptor.open(boost::asio::ip::tcp::v4(), ec) ||
      acceptor.bind({boost::asio::ip::tcp::v4(), port}, ec);
  return ec == boost::asio::error::address_in_use;
}

std::string lookupCommand(const json& etcd_spec, const std::string& command) {
  std::string cmd = etcd_spec.value(command + "_cmd", "");
  if (cmd.empty()) {
    setenv("LC_ALL", "C", 1);
    cmd = boost::process::search_path(command).string();
  }

  if (cmd.empty()) {
    setenv("LC_ALL", "en_US.UTF-8", 1);
    cmd = boost::process::search_path(command).string();
  }

  return cmd;
}

Status checkEtcdCmd(const std::string& etcd_cmd) {
  if (etcd_cmd.empty()) {
    std::string error_message =
        "Failed to find etcd binary, please specify its path using the "
        "`--etcd_cmd` argument and try again.";
    LOG(WARNING) << error_message;
    return Status::EtcdError("Failed to find etcd binary");
  }
  if (!ghc::filesystem::exists(ghc::filesystem::path(etcd_cmd))) {
    std::string error_message =
        "The etcd binary '" + etcd_cmd +
        "' does not exist, please specify the correct path using "
        "the `--etcd_cmd` argument and try again.";
    LOG(WARNING) << error_message;
    return Status::EtcdError("The etcd binary does not exist");
  }
  return Status::OK();
}

Status checkEtcdctlCommand(const std::string& etcdctl_cmd) {
  if (etcdctl_cmd.empty()) {
    std::string error_message =
        "Failed to find etcdctl binary, please specify its path using the "
        "`--etcdctl_cmd` argument and try again.";
    LOG(WARNING) << error_message;
    return Status::EtcdError("Failed to find etcdctl binary");
  }
  if (!ghc::filesystem::exists(ghc::filesystem::path(etcdctl_cmd))) {
    std::string error_message =
        "The etcd binary '" + etcdctl_cmd +
        "' does not exist, please specify the correct path using "
        "the `--etcdctl_cmd` argument and try again.";
    LOG(WARNING) << error_message;
    return Status::EtcdError("The etcdctl binary does not exist");
  }
  return Status::OK();
}

EtcdLauncher::EtcdLauncher(const json& etcd_spec,
                           const bool create_new_instance)
    : etcd_spec_(etcd_spec), create_new_instance_(create_new_instance) {}

EtcdLauncher::~EtcdLauncher() {
  if (etcd_proc_) {
    std::error_code err;
    etcd_proc_->terminate(err);
    kill(etcd_proc_->id(), SIGTERM);
    etcd_proc_->wait(err);
  }
  if (!etcd_data_dir_.empty()) {
    std::error_code err;
    ghc::filesystem::remove_all(ghc::filesystem::path(etcd_data_dir_), err);
  }
}

Status EtcdLauncher::LaunchEtcdServer(
    std::unique_ptr<etcd::Client>& etcd_client, std::string& sync_lock) {
  std::string const& etcd_endpoint =
      etcd_spec_["etcd_endpoint"].get_ref<std::string const&>();
  RETURN_ON_ERROR(parseEndpoint());

  std::string etcd_endpoint_ip;
  if (!validate_advertise_hostname(etcd_endpoint_ip, endpoint_host_)) {
    // resolving failure means we need to wait the srv name becomes ready in the
    // DNS side
    return Status::OK();
  }

  // resolve etcdctl binary
  std::string etcdctl_cmd = etcd_spec_.value("etcdctl_cmd", "");
  if (etcdctl_cmd.empty()) {
    etcdctl_cmd = lookupCommand(etcd_spec_, "etcdctl");
  }
  RETURN_ON_ERROR(checkEtcdctlCommand(etcdctl_cmd));
  etcdctl_cmd_ = etcdctl_cmd;
  LOG(INFO) << "Found etcdctl at: " << etcdctl_cmd;

  bool etcd_cluster_existing = false;
  etcd_client.reset(new etcd::Client(etcd_endpoint));
  int retries = 0;
  while (retries < first_probe_retries) {
    etcd_client.reset(new etcd::Client(etcd_endpoint));
    if (probeEtcdServer(etcd_client, sync_lock)) {
      etcd_cluster_existing = true;
      if (!create_new_instance_) {
        etcd_endpoints_ = etcd_endpoint;
        std::cout << "Etcd cluster already exists at, not to add new instance: "
                  << etcd_endpoint << std::endl;
        return Status::OK();
      }
      break;
    }
    retries += 1;
    sleep(1);
  }

  RETURN_ON_ERROR(initHostInfo());
  bool try_launch = false;
  if (local_hostnames_.find(endpoint_host_) != local_hostnames_.end() ||
      local_ip_addresses_.find(endpoint_host_) != local_ip_addresses_.end()) {
    try_launch = true;
  }

  if (!try_launch) {
    return Status::OK();
  }

  LOG(INFO) << "Starting the etcd server";

  // resolve etcd binary
  std::string etcd_cmd = etcd_spec_.value("etcd_cmd", "");
  if (etcd_cmd.empty()) {
    etcd_cmd = lookupCommand(etcd_spec_, "etcd");
  }
  RETURN_ON_ERROR(checkEtcdCmd(etcd_cmd));
  LOG(INFO) << "Found etcd at: " << etcd_cmd;

  std::string host_to_advertise;
  if (host_to_advertise.empty()) {
    for (auto const& h : local_hostnames_) {
      std::string ipaddress{};
      if (h != "localhost" && validate_advertise_hostname(ipaddress, h)) {
        host_to_advertise = ipaddress;
        break;
      }
    }
  }
  if (host_to_advertise.empty()) {
    for (auto const& h : local_ip_addresses_) {
      std::string ipaddress{};
      if (h != "127.0.0.1" && h != "0.0.0.0" &&
          validate_advertise_hostname(ipaddress, h)) {
        host_to_advertise = h;
        break;
      }
    }
  }
  if (host_to_advertise.empty()) {
    host_to_advertise = "127.0.0.1";
  }

  boost::asio::io_context context;
  if (etcd_cluster_existing) {
    while (check_port_in_use(context, endpoint_port_)) {
      endpoint_port_ += 1;
    }
  }

  unsigned int etcd_peer_port = endpoint_port_ + 1;
  while (check_port_in_use(context, etcd_peer_port)) {
    etcd_peer_port += 1;
  }

  std::string etcd_endpoints;
  std::vector<std::string> existing_members;
  std::vector<std::string> peer_urls;
  std::vector<std::string> client_urls;
  std::string new_member_name = generateMemberName(existing_members);

  std::string client_endpoint =
      "http://" + host_to_advertise + ":" + std::to_string(endpoint_port_);
  std::string peer_endpoint =
      "http://" + host_to_advertise + ":" + std::to_string(etcd_peer_port);

  std::vector<std::string> args;
  std::string endpoint;

  if (etcd_cluster_existing) {
    std::string cluster_name;

    std::vector<json> members = listMembers(etcd_endpoint);
    if (members.size() == 0) {
      return Status::EtcdError("No members found via etcdctl");
    }

    existing_members = listMembersName(members);
    new_member_name = generateMemberName(existing_members);
    peer_urls = listPeerURLs(members);
    if (peer_urls.size() == 0) {
      return Status::EtcdError("No peer urls found via etcdctl");
    }
    std::vector<std::string> client_urls = listClientURLs(members);
    if (peer_urls.size() == 0) {
      return Status::EtcdError("No client urls found via etcdctl");
    }

    endpoint = boost::algorithm::join(client_urls, ",");
    if (!addMember(new_member_name, peer_endpoint, endpoint).ok()) {
      return Status::EtcdError("Failed to add new member to the etcd cluster");
    }

    args.emplace_back("--initial-cluster-state");
    args.emplace_back("existing");
    args.emplace_back("--initial-cluster");
    if (existing_members.size() != peer_urls.size()) {
      return Status::EtcdError(
          "The number of existing members is not equal to the number of peer "
          "urls");
    }
    for (size_t i = 0; i < existing_members.size(); i++) {
      cluster_name += existing_members[i] + "=" + peer_urls[i] + ",";
    }
    cluster_name += new_member_name + "=" + peer_endpoint;
    args.emplace_back(cluster_name);
  } else {
    args.emplace_back("--initial-cluster-state");
    args.emplace_back("new");
    args.emplace_back("--initial-cluster");
    args.emplace_back(new_member_name + "=" + peer_endpoint);
  }
  args.emplace_back("--name");
  args.emplace_back(new_member_name);
  args.emplace_back("--listen-client-urls");
  args.emplace_back("http://0.0.0.0:" + std::to_string(endpoint_port_));
  args.emplace_back("--advertise-client-urls");
  args.emplace_back(client_endpoint);
  args.emplace_back("--listen-peer-urls");
  args.emplace_back("http://0.0.0.0:" + std::to_string(etcd_peer_port));
  args.emplace_back("--initial-advertise-peer-urls");
  args.emplace_back(peer_endpoint);

  if (endpoint == "") {
    etcd_endpoints_ = client_endpoint;
  } else {
    etcd_endpoints_ = endpoint + "," + client_endpoint;
  }

  etcd_data_dir_ = etcd_spec_.value("etcd_data_dir", "");
  if (etcd_data_dir_.empty()) {
    std::string file_template = "/tmp/" + new_member_name + ".etcd.XXXXXX";
    char* data_dir = mkdtemp(const_cast<char*>(file_template.c_str()));
    if (data_dir == nullptr) {
      return Status::EtcdError(
          "Failed to create a temporary directory for etcd data");
    }
    etcd_data_dir_ = data_dir;
  }
  // prepare the data dir
  if (!ghc::filesystem::exists(ghc::filesystem::path(etcd_data_dir_))) {
    std::error_code err;
    ghc::filesystem::create_directories(ghc::filesystem::path(etcd_data_dir_),
                                        err);
    if (err) {
      return Status::EtcdError("Failed to create etcd data directory: " +
                               err.message());
    }
  }
  LOG(INFO) << "Vineyard will use '" << etcd_data_dir_
            << "' as the data directory of etcd";
  args.emplace_back("--data-dir");
  args.emplace_back(etcd_data_dir_);

  auto env = boost::this_process::environment();
  // n.b., avoid using `[]` operator to update env, see boostorg/process#122.
#ifndef NDEBUG
  if (VLOG_IS_ON(100)) {
    env.set("ETCD_LOG_LEVEL", "debug");
  } else if (VLOG_IS_ON(10)) {
    env.set("ETCD_LOG_LEVEL", "info");
  } else {
    env.set("ETCD_LOG_LEVEL", "warn");
  }
#else
  env.set("ETCD_LOG_LEVEL", "error");
#endif

  // leave a log here for getting the etcd endpoints of each member
  LOG(INFO) << "Launching etcd with: " << boost::algorithm::join(args, " ");
  std::error_code ec;
  etcd_proc_ = std::unique_ptr<boost::process::child>(new boost::process::child(
      etcd_cmd, boost::process::args(args), boost::process::std_out > stdout,
      boost::process::std_err > stderr, env, ec));
  if (ec) {
    LOG(ERROR) << "Failed to launch etcd: " << ec.message();
    return Status::EtcdError("Failed to launch etcd: " + ec.message());
  } else {
    LOG(INFO) << "Etcd launched: pid = " << etcd_proc_->id() << ", listen on "
              << endpoint_port_;

    int retries = 0;
    std::error_code err;
    while (etcd_proc_ && etcd_proc_->running(err) && !err &&
           retries < max_probe_retries) {
      etcd_client.reset(new etcd::Client(client_endpoint));
      if (probeEtcdServer(etcd_client, sync_lock)) {
        etcd_member_id_ = findMemberID(new_member_name, etcd_endpoints_);
        // reset the etcd watcher
        break;
      }
      retries += 1;
      sleep(1);
    }
    if (!etcd_proc_) {
      return handleEtcdFailure(
          new_member_name,
          "Failed to wait until etcd ready: operation has been interrupted");
    } else if (err) {
      return handleEtcdFailure(
          new_member_name, "Failed to wait until etcd ready: " + err.message());
    } else if (retries >= max_probe_retries) {
      return handleEtcdFailure(
          new_member_name,
          "Etcd has been launched but failed to connect to it");
    } else {
      return Status::OK();
    }
  }
}

Status EtcdLauncher::handleEtcdFailure(const std::string& member_name,
                                       const std::string& errMessage) {
  auto member_id = findMemberID(member_name, etcd_endpoints_);
  RETURN_ON_ERROR(removeMember(etcd_member_id_));
  etcd_member_id_.clear();
  return Status::IOError(errMessage);
}

Status EtcdLauncher::addMember(std::string& member_name,
                               std::string& peer_endpoint,
                               const std::string& etcd_endpoints,
                               int max_retries) {
  int retries = 0;

  while (retries < max_retries) {
    std::error_code ec;
    std::unique_ptr<boost::process::child> etcdctl_proc_ =
        std::make_unique<boost::process::child>(
            etcdctl_cmd_, "member", "add", member_name,
            "--peer-urls=" + peer_endpoint, "--endpoints=" + etcd_endpoints,
            boost::process::std_out > stdout, boost::process::std_err > stderr,
            ec);
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

Status EtcdLauncher::removeMember(std::string& member_id, int max_retries) {
  int retries = 0;

  auto members = listMembers(etcd_endpoints_);
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
            "--endpoints=" + etcd_endpoints_, boost::process::std_out > stdout,
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

std::string EtcdLauncher::findMemberID(const std::string& member_name,
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

bool EtcdLauncher::memberIsStarted(const std::string& member_id,
                                   const std::string& etcd_endpoints) {
  auto members = listMembers(etcd_endpoints);
  for (const auto& member : members) {
    std::stringstream ss;
    ss << std::hex << member["ID"].get<uint64_t>();
    if (ss.str() == member_id && member.find("clientURLs") != member.end()) {
      return true;
    }
  }
  return false;
}

std::vector<json> EtcdLauncher::listMembers(const std::string& etcd_endpoints) {
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

std::vector<std::string> EtcdLauncher::listPeerURLs(
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

std::vector<std::string> EtcdLauncher::listClientURLs(
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

std::vector<std::string> EtcdLauncher::listMembersName(
    const std::vector<json>& members) {
  std::vector<std::string> members_name;
  for (const auto& member : members) {
    auto name = member["name"];
    members_name.emplace_back(name.get<std::string>());
  }
  return members_name;
}

std::string EtcdLauncher::generateMemberName(
    std::vector<std::string> const& existing_members_name) {
  // by default, the member name is the hostname + the current timestamp
  while (true) {
    std::string member_name =
        get_hostname() + "-" + std::to_string(std::time(nullptr));
    if (std::find(existing_members_name.begin(), existing_members_name.end(),
                  member_name) == existing_members_name.end()) {
      return member_name;
    }
  }
}

Status EtcdLauncher::parseEndpoint() {
  std::string const& etcd_endpoint =
      etcd_spec_["etcd_endpoint"].get_ref<std::string const&>();
  size_t port_pos = etcd_endpoint.find_last_of(':');
  std::string endpoint_port_string =
      etcd_endpoint.substr(port_pos + 1, etcd_endpoint.size() - port_pos - 1);
  if (endpoint_port_string.empty()) {
    return Status::Invalid("The etcd endpoint '" + etcd_endpoint +
                           "' is invalid");
  }
  endpoint_port_ = std::atoi(endpoint_port_string.c_str());
  size_t prefix = etcd_endpoint.find_last_of('/');
  endpoint_host_ = etcd_endpoint.substr(prefix + 1, port_pos - prefix - 1);
  if (endpoint_host_.empty()) {
    return Status::Invalid("The etcd endpoint '" + etcd_endpoint +
                           "' is invalid");
  }
  return Status::OK();
}

Status EtcdLauncher::initHostInfo() {
  local_hostnames_.emplace("localhost");
  local_ip_addresses_.emplace("127.0.0.1");
  local_ip_addresses_.emplace("0.0.0.0");
  std::string hostname_value = get_hostname();
  local_hostnames_.emplace(hostname_value);
  struct hostent* host_entry = gethostbyname(hostname_value.c_str());
  if (host_entry == nullptr) {
    LOG(WARNING) << "Failed in gethostbyname: " << hstrerror(h_errno);
    return Status::OK();
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
  return Status::OK();
}

bool EtcdLauncher::probeEtcdServer(std::unique_ptr<etcd::Client>& etcd_client,
                                   std::string const& key) {
  // probe: as a 1-limit range request

  // don't use `etcd_client->ls(key, 1).get().is_ok()` to avoid a random
  // stack-buffer-overflow error when asan is enabled.

  auto task = etcd_client->ls(key, 1);
  auto response = task.get();
  return etcd_client && response.is_ok();
}

Status EtcdLauncher::UpdateEndpoint() {
  auto members = listMembers(etcd_endpoints_);
  auto client_urls = listClientURLs(members);
  etcd_endpoints_ = boost::algorithm::join(client_urls, ",");
  return Status::OK();
}

}  // namespace vineyard

#endif  // BUILD_VINEYARDD_ETCD
