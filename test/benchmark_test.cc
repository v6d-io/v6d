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

#include <getopt.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/object_meta.h"
#include "client/rpc_client.h"
#include "common/util/status.h"

using namespace vineyard;  // NOLINT(build/namespaces)
using namespace std;       // NOLINT(build/namespaces)

#define MEASURE_AND_PRINT_STATS(operation_name, operation_code)        \
  {                                                                    \
    auto start = std::chrono::high_resolution_clock::now();            \
    operation_code;                                                    \
    auto end = std::chrono::high_resolution_clock::now();              \
    std::chrono::duration<double, std::micro> duration = end - start;  \
    collectLatencies(local_latencies, latencies);                      \
    printStats(operation_name, requests_num, num_threads, clients_num, \
               data_size, latencies, duration.count());                \
  }

enum class OperationType {
  PUT_BLOB,
  GET_BLOB,
  PUT_BLOBS,
  GET_BLOBS,
  PUT_REMOTE_BLOB,
  GET_REMOTE_BLOB,
  PUT_REMOTE_BLOBS,
  GET_REMOTE_BLOBS,
  UNKNOWN
};

void printHelp(const char* program_name) {
  std::cout
      << "Usage: " << program_name << " [OPTIONS]\n"
      << "Options:\n"
      << "  -h, --help                     Show this help message and exit\n"
      << "  -i, --ipc_socket=IPC_SOCKET    Specify the IPC socket path "
         "(required)\n"
      << "  -r, --rpc_endpoint=RPC_ENDPOINT  Specify the RPC endpoint "
         "(required)\n"
      << "  -d, --rdma_endpoint=RDMA_ENDPOINT  Specify the RDMA endpoint "
         "(required)\n"
      << "  -c, --clients_num=NUM          Number of clients (required)\n"
      << "  -s, --data_size=SIZE           Data size (e.g., 1KB, 1MB) "
         "(required)\n"
      << "  -n, --requests_num=NUM         Number of requests (required)\n"
      << "  -t, --num_threads=NUM          Number of threads (required)\n"
      << "  -o, --operation=TYPE           Operation type (put_blob, get_blob, "
         "put_blobs, get_blobs, put_remote_blob, get_remote_blob, "
         "put_remote_blobs, get_remote_blobs) (required)\n";
}

OperationType parseOperationType(const string& op_str) {
  if (op_str == "put_blob")
    return OperationType::PUT_BLOB;
  if (op_str == "get_blob")
    return OperationType::GET_BLOB;
  if (op_str == "put_blobs")
    return OperationType::PUT_BLOBS;
  if (op_str == "get_blobs")
    return OperationType::GET_BLOBS;
  if (op_str == "put_remote_blob")
    return OperationType::PUT_REMOTE_BLOB;
  if (op_str == "get_remote_blob")
    return OperationType::GET_REMOTE_BLOB;
  if (op_str == "put_remote_blobs")
    return OperationType::PUT_REMOTE_BLOBS;
  if (op_str == "get_remote_blobs")
    return OperationType::GET_REMOTE_BLOBS;
  return OperationType::UNKNOWN;
}

size_t parseDataSize(const std::string& sizeStr) {
  std::istringstream is(sizeStr);
  double size;
  is >> size;
  std::string unit;
  if (is.fail()) {
    throw std::invalid_argument("Invalid data size format");
  }
  is >> std::ws;
  if (is.peek() != std::istringstream::traits_type::eof()) {
    is >> unit;
  }
  if (unit.empty() || unit == "B" || unit == "b") {
    return static_cast<size_t>(size);
  } else if (unit == "K" || unit == "k" || unit == "KB" || unit == "kb" ||
             unit == "KILOBYTE" || unit == "kilobyte") {
    return static_cast<size_t>(size * pow(1024, 1));
  } else if (unit == "M" || unit == "m" || unit == "MB" || unit == "mb" ||
             unit == "MEGABYTE" || unit == "megabyte") {
    return static_cast<size_t>(size * pow(1024, 2));
  } else if (unit == "G" || unit == "g" || unit == "GB" || unit == "gb" ||
             unit == "GIGABYTE" || unit == "gigabyte") {
    return static_cast<size_t>(size * pow(1024, 3));
  } else if (unit == "T" || unit == "t" || unit == "TB" || unit == "tb" ||
             unit == "TERABYTE" || unit == "terabyte") {
    return static_cast<size_t>(size * pow(1024, 4));
  } else {
    throw std::invalid_argument("Unsupported data size unit");
  }
}

template <typename ClientType>
vector<shared_ptr<ClientType>> generateClients(int clients_num,
                                               string ipc_socket,
                                               string endpoint,
                                               string rdma_endpoint = "") {
  vector<shared_ptr<ClientType>> clients;
  for (int i = 0; i < clients_num; i++) {
    auto client = make_shared<ClientType>();
    if constexpr (is_same_v<ClientType, Client>) {
      VINEYARD_CHECK_OK(client->Connect(ipc_socket));
    } else if constexpr (is_same_v<ClientType, RPCClient>) {
      VINEYARD_CHECK_OK(client->Connect(endpoint, "", "", rdma_endpoint));
    }
    clients.push_back(client);
  }
  return clients;
}

template <typename ClientType>
vector<vector<shared_ptr<ClientType>>> generateClientsForThreads(
    string ipc_socket, string endpoint, int clients_num, int num_threads,
    string rdma_endpoint = "") {
  vector<shared_ptr<ClientType>> clients = generateClients<ClientType>(
      clients_num, ipc_socket, endpoint, rdma_endpoint);
  vector<vector<shared_ptr<ClientType>>> clients_per_thread(
      num_threads,
      std::vector<shared_ptr<ClientType>>(clients_num / num_threads));
  clients_per_thread[0].resize(clients_num / num_threads +
                               clients_num % num_threads);

  size_t index = 0;
  for (size_t i = 0; i < clients_per_thread.size(); i++) {
    for (size_t j = 0; j < clients_per_thread[i].size(); j++) {
      clients_per_thread[i][j] = clients[index++];
    }
  }

  return clients_per_thread;
}

template <typename ClientType>
void clearBlobs(std::shared_ptr<ClientType> client,
                std::vector<std::vector<ObjectID>> object_ids) {
  for (size_t i = 0; i < object_ids.size(); i++) {
    for (auto& obj : object_ids[i]) {
      VINEYARD_CHECK_OK(client->DelData(obj));
    }
  }
}

template <typename ClientType>
void clearClients(vector<vector<shared_ptr<ClientType>>>& clients) {
  for (size_t i = 0; i < clients.size(); ++i) {
    for (size_t j = 0; j < clients[i].size(); ++j) {
      clients[i][j]->Disconnect();
    }
  }
}

std::string generateRandomData(int data_size) {
  const char charset[] =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  std::string value;
  value.resize(data_size);
  unsigned int seed = 1234;
  for (int i = 0; i < data_size; i++) {
    value[i] = charset[rand_r(&seed) % (sizeof(charset) - 1)];
  }
  return value;
}

void CreateBlob(std::vector<std::shared_ptr<Client>>& clients,
                std::string value, int requests_num,
                std::vector<double>& latencies,
                std::vector<ObjectID>& blob_ids) {
  int data_size = value.size();
  for (int i = 0; i < requests_num; i++) {
    int client_index = i % clients.size();
    std::shared_ptr<Client> client = clients[client_index];
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_ptr<BlobWriter> blob_writer;
    VINEYARD_CHECK_OK(client->CreateBlob(data_size, blob_writer));
    std::memcpy(blob_writer->data(), value.c_str(), data_size);
    blob_writer->Seal(*client.get());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    latencies.push_back(duration.count());
    blob_ids.push_back(blob_writer->id());
  }
}

void CreateBlobs(std::vector<std::shared_ptr<Client>>& clients,
                 std::string value, int requests_num,
                 std::vector<double>& latencies,
                 std::vector<ObjectID>& blob_ids) {
  int requests_per_client = requests_num / clients.size();
  int remain_requests = requests_num % clients.size();

  auto create_blobs = [&](std::shared_ptr<Client> client, int num_blobs) {
    std::vector<size_t> data_sizes(num_blobs, value.size());
    std::vector<std::unique_ptr<BlobWriter>> blob_writers;
    blob_writers.reserve(num_blobs);

    auto start = std::chrono::high_resolution_clock::now();
    VINEYARD_CHECK_OK(client->CreateBlobs(data_sizes, blob_writers));
    for (int i = 0; i < num_blobs; ++i) {
      std::memcpy(blob_writers[i]->data(), value.c_str(), value.size());
      blob_writers[i]->Seal(*client.get());
      blob_ids.push_back(blob_writers[i]->id());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;

    latencies.push_back(duration.count());
  };

  for (size_t client_index = 0;
       client_index < clients.size() && requests_per_client > 0;
       ++client_index) {
    create_blobs(clients[client_index], requests_per_client);
  }

  if (remain_requests > 0) {
    int client_index = 0;
    create_blobs(clients[client_index], remain_requests);
  }
}

void GetBlob(std::vector<ObjectID> blob_ids,
             std::vector<std::shared_ptr<Client>>& clients, int requests_num,
             std::vector<double>& latencies) {
  for (int i = 0; i < requests_num; i++) {
    int client_index = i % clients.size();
    std::shared_ptr<Client> client = clients[client_index];
    auto start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<Blob> blob;
    VINEYARD_CHECK_OK(client->GetBlob(blob_ids[0], blob));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    latencies.push_back(duration.count());
  }
}

void GetBlobs(std::vector<ObjectID> blob_ids,
              std::vector<std::shared_ptr<Client>>& clients, int requests_num,
              std::vector<double>& latencies) {
  int requests_per_client = requests_num / clients.size();
  int remain_requests = requests_num % clients.size();

  auto get_blobs = [&](std::shared_ptr<Client> client, int num_requests) {
    std::vector<std::shared_ptr<Blob>> blobs;
    blobs.reserve(blob_ids.size());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_requests; ++i) {
      VINEYARD_CHECK_OK(client->GetBlobs(blob_ids, blobs));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;

    latencies.push_back(duration.count());
  };

  for (size_t client_index = 0;
       client_index < clients.size() && requests_per_client > 0;
       ++client_index) {
    get_blobs(clients[client_index], requests_per_client);
  }

  if (remain_requests > 0) {
    int client_index = 0;
    get_blobs(clients[client_index], remain_requests);
  }
}

void CreateRemoteBlob(
    std::vector<std::shared_ptr<RPCClient>>& rpc_clients,
    std::vector<std::shared_ptr<RemoteBlobWriter>>& remote_blob_writers,
    std::vector<double>& latencies, std::vector<ObjectID>& blob_ids) {
  for (size_t i = 0; i < remote_blob_writers.size(); i++) {
    int client_index = i % rpc_clients.size();
    std::shared_ptr<RPCClient> rpc_client = rpc_clients[client_index];
    ObjectMeta meta;
    auto start = std::chrono::high_resolution_clock::now();
    VINEYARD_CHECK_OK(
        rpc_client->CreateRemoteBlob(remote_blob_writers[i], meta));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    latencies.push_back(duration.count());
    blob_ids.push_back(meta.GetId());
  }
}

void CreateRemoteBlobs(
    std::vector<std::shared_ptr<RPCClient>>& rpc_clients,
    std::vector<std::shared_ptr<RemoteBlobWriter>>& remote_blob_writers,
    std::vector<double>& latencies, std::vector<ObjectID>& blob_ids) {
  int requests_num = remote_blob_writers.size();
  int requests_per_client = requests_num / rpc_clients.size();
  int remain_requests = requests_num % rpc_clients.size();

  auto create_remote_blobs = [&](std::shared_ptr<RPCClient> rpc_client,
                                 int num_requests, int start_index) {
    std::vector<ObjectMeta> blob_metas;
    blob_metas.reserve(num_requests);

    auto start_it = remote_blob_writers.begin() + start_index;
    auto end_it = start_it + num_requests;
    std::vector<std::shared_ptr<RemoteBlobWriter>> sub_blob_writers(start_it,
                                                                    end_it);

    auto start = std::chrono::high_resolution_clock::now();
    VINEYARD_CHECK_OK(
        rpc_client->CreateRemoteBlobs(sub_blob_writers, blob_metas));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    latencies.push_back(duration.count());

    for (auto& meta : blob_metas) {
      blob_ids.push_back(meta.GetId());
    }
  };

  int start_index = 0;
  for (size_t client_index = 0;
       client_index < rpc_clients.size() && requests_per_client > 0;
       ++client_index) {
    create_remote_blobs(rpc_clients[client_index], requests_per_client,
                        start_index);
    start_index += requests_per_client;
  }

  if (remain_requests > 0) {
    int client_index = 0;
    create_remote_blobs(rpc_clients[client_index], remain_requests,
                        start_index);
  }
}

void GetRemoteBlob(std::vector<ObjectID>& blob_ids,
                   std::vector<std::shared_ptr<RPCClient>>& rpc_clients,
                   int& requests_num, std::vector<double>& latencies) {
  for (int i = 0; i < requests_num; i++) {
    int client_index = i % rpc_clients.size();
    std::shared_ptr<RPCClient> rpc_client = rpc_clients[client_index];
    auto start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<RemoteBlob> remote_blob;
    VINEYARD_CHECK_OK(rpc_client->GetRemoteBlob(blob_ids[0], remote_blob));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    latencies.push_back(duration.count());
  }
}

void GetRemoteBlobs(std::vector<ObjectID>& blob_ids,
                    std::vector<std::shared_ptr<RPCClient>>& rpc_clients,
                    int& requests_num, std::vector<double>& latencies) {
  int requests_per_client = requests_num / rpc_clients.size();
  int remain_requests = requests_num % rpc_clients.size();

  auto get_remote_blobs = [&](std::shared_ptr<RPCClient> rpc_client,
                              int num_requests, int start_index) {
    std::vector<ObjectMeta> blob_metas;
    std::vector<std::shared_ptr<RemoteBlob>> remote_blobs;
    remote_blobs.reserve(num_requests);

    auto start_it = blob_ids.begin() + start_index;
    auto end_it = start_it + num_requests;
    std::vector<ObjectID> sub_blob_ids(start_it, end_it);

    auto start = std::chrono::high_resolution_clock::now();
    VINEYARD_CHECK_OK(rpc_client->GetRemoteBlobs(sub_blob_ids, remote_blobs));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    latencies.push_back(duration.count());

    for (const auto& blob : remote_blobs) {
      blob_ids.push_back(blob->id());
    }
  };

  int start_index = 0;
  for (size_t client_index = 0;
       client_index < rpc_clients.size() && requests_per_client > 0;
       ++client_index) {
    get_remote_blobs(rpc_clients[client_index], requests_per_client,
                     start_index);
    start_index += requests_per_client;
  }

  if (remain_requests > 0) {
    int client_index = 0;
    get_remote_blobs(rpc_clients[client_index], remain_requests, start_index);
  }
}

std::vector<std::vector<ObjectID>> PutBlobs(std::shared_ptr<Client> client,
                                            int requests_num, int data_size,
                                            int num_threads) {
  std::vector<std::vector<ObjectID>> object_ids(num_threads);
  std::string value = generateRandomData(data_size);
  std::vector<std::unique_ptr<BlobWriter>> blob_writers;
  blob_writers.reserve(requests_num);
  const std::vector<size_t> data_sizes(requests_num, value.size());
  VINEYARD_CHECK_OK(client->CreateBlobs(data_sizes, blob_writers));
  for (int i = 0; i < requests_num; i++) {
    std::memcpy(blob_writers[i]->data(), value.c_str(), data_size);
    blob_writers[i]->Seal(*client.get());
    object_ids[i % num_threads].push_back(blob_writers[i]->id());
  }

  return object_ids;
}

std::vector<std::vector<ObjectID>> PutRemoteBlobs(
    std::shared_ptr<RPCClient> rpc_client, int requests_num, int data_size,
    int num_threads) {
  std::string value = generateRandomData(data_size);
  std::vector<std::shared_ptr<RemoteBlobWriter>> remote_blob_writers;
  for (int i = 0; i < requests_num; i++) {
    std::shared_ptr<RemoteBlobWriter> remote_blob_writer(
        new RemoteBlobWriter(data_size));
    std::memcpy(remote_blob_writer->data(), value.c_str(), data_size);
    remote_blob_writers.push_back(remote_blob_writer);
  }

  std::vector<ObjectMeta> metas;
  metas.reserve(remote_blob_writers.size());
  VINEYARD_CHECK_OK(rpc_client->CreateRemoteBlobs(remote_blob_writers, metas));
  std::vector<std::vector<ObjectID>> object_ids(num_threads);
  for (size_t i = 0; i < metas.size(); i++) {
    object_ids[i % num_threads].push_back(metas[i].GetId());
  }

  return object_ids;
}

template <typename Func>
void TestCreate(
    Func&& create_func,
    std::vector<std::vector<std::shared_ptr<Client>>>& clients_per_thread,
    std::string& value, std::vector<int>& requests_num,
    std::vector<std::vector<double>>& local_latencies,
    std::vector<std::vector<ObjectID>>& object_ids) {
  std::vector<std::thread> threads;
  int num_threads = clients_per_thread.size();
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(
        create_func, std::ref(clients_per_thread[i]), value, requests_num[i],
        std::ref(local_latencies[i]), std::ref(object_ids[i])));
  }
  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }
}

template <typename Func>
void TestGet(
    Func&& get_func,
    std::vector<std::vector<std::shared_ptr<Client>>>& clients_per_thread,
    std::vector<int>& requests_num,
    std::vector<std::vector<double>>& local_latencies,
    std::vector<std::vector<ObjectID>>& blobs_id) {
  std::vector<std::thread> threads;
  int num_threads = clients_per_thread.size();
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(
        get_func, std::ref(blobs_id[i]), std::ref(clients_per_thread[i]),
        std::ref(requests_num[i]), std::ref(local_latencies[i])));
  }
  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }
}

template <typename Func>
void TestCreateRemote(
    Func&& create_remote_func,
    std::vector<std::vector<std::shared_ptr<RPCClient>>>& clients_per_thread,
    std::vector<std::vector<std::shared_ptr<RemoteBlobWriter>>>&
        remote_blob_writers,
    std::vector<std::vector<double>>& local_latencies,
    std::vector<std::vector<ObjectID>>& object_ids) {
  std::vector<std::thread> threads;
  int num_threads = clients_per_thread.size();
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(
        std::thread(create_remote_func, std::ref(clients_per_thread[i]),
                    std::ref(remote_blob_writers[i]),
                    std::ref(local_latencies[i]), std::ref(object_ids[i])));
  }
  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }
}

template <typename Func>
void TestGetRemote(
    Func&& get_remote_func, std::vector<std::vector<ObjectID>>& blob_ids,
    std::vector<std::vector<std::shared_ptr<RPCClient>>>& clients_per_thread,
    std::vector<int> requests_num,
    std::vector<std::vector<double>>& local_latencies) {
  std::vector<std::thread> threads;
  int num_threads = clients_per_thread.size();
  for (int i = 0; i < num_threads; i++) {
    threads.push_back(std::thread(
        get_remote_func, std::ref(blob_ids[i]), std::ref(clients_per_thread[i]),
        std::ref(requests_num[i]), std::ref(local_latencies[i])));
  }
  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }
}

void printStats(const std::string& op_name, int requests_num, int num_threads,
                int clients_num, size_t data_size,
                const std::vector<double>& latencies, double total_time) {
  double min_time = *std::min_element(latencies.begin(), latencies.end());
  double max_time = *std::max_element(latencies.begin(), latencies.end());
  double accumulate_total_time =
      std::accumulate(latencies.begin(), latencies.end(), 0.0);
  double accumulate_avg_time = accumulate_total_time / latencies.size();
  double throughput = requests_num / (total_time / 1e6);
  std::vector<double> sorted_latencies = latencies;
  std::sort(sorted_latencies.begin(), sorted_latencies.end());
  auto percentile = [&](int p) {
    return sorted_latencies[p * latencies.size() / 100];
  };
  std::cout << "====== " << op_name << " ======" << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "  " << requests_num << " requests completed in "
            << (total_time / 1e6) << " seconds" << std::endl;
  std::cout << "  " << clients_num << " clients parallel in " << num_threads
            << " threads." << std::endl;
  std::cout << "  " << data_size << " bytes payload" << std::endl;
  std::cout << "  min / avg / max latencies: " << min_time << " / "
            << accumulate_avg_time << " / " << max_time << " μs" << std::endl;
  std::cout << "  throughput: " << throughput << " requests per second"
            << std::endl;
  std::cout << "  latencies percentiles:" << std::endl;
  std::cout << "    p50: " << percentile(50) << " μs" << std::endl;
  std::cout << "    p95: " << percentile(95) << " μs" << std::endl;
  std::cout << "    p99: " << percentile(99) << " μs" << std::endl;
}

void collectLatencies(const std::vector<std::vector<double>>& local_latencies,
                      std::vector<double>& latencies) {
  for (const auto& l : local_latencies) {
    latencies.insert(latencies.end(), l.begin(), l.end());
  }
}

void generateRemoteBlobWriters(
    std::vector<std::vector<std::shared_ptr<RemoteBlobWriter>>>&
        remote_blob_writers,
    size_t data_size, const std::string& random_data) {
  for (auto& remote_blob_writer : remote_blob_writers) {
    for (auto& writer : remote_blob_writer) {
      writer = std::make_shared<RemoteBlobWriter>(data_size);
      std::memcpy(writer->data(), random_data.c_str(), data_size);
    }
  }
}

int main(int argc, char* argv[]) {
  // Define long options
  static struct option long_options[] = {
      {"help", no_argument, nullptr, 'h'},
      {"ipc_socket", required_argument, nullptr, 'i'},
      {"rpc_endpoint", required_argument, nullptr, 'r'},
      {"rdma_endpoint", required_argument, nullptr, 'd'},
      {"clients_num", required_argument, nullptr, 'c'},
      {"data_size", required_argument, nullptr, 's'},
      {"requests_num", required_argument, nullptr, 'n'},
      {"num_threads", required_argument, nullptr, 't'},
      {"operation", required_argument, nullptr, 'o'},
      {nullptr, 0, nullptr, 0}};

  string ipc_socket;
  string rpc_endpoint;
  string rdma_endpoint;
  int clients_num = 0;
  string data_size_str;
  int requests_num = 0;
  int num_threads = 0;
  string operation_str;
  OperationType operation;

  int opt;
  int long_index = 0;
  while ((opt = getopt_long(argc, argv, "hi:r:d:c:s:n:t:o:", long_options,
                            &long_index)) != -1) {
    switch (opt) {
    case 'h':
      printHelp(argv[0]);
      return 0;
    case 'i':
      ipc_socket = optarg;
      break;
    case 'r':
      rpc_endpoint = optarg;
      break;
    case 'd':
      rdma_endpoint = optarg;
      break;
    case 'c':
      clients_num = std::stoi(optarg);
      break;
    case 's':
      data_size_str = optarg;
      break;
    case 'n':
      requests_num = std::stoi(optarg);
      break;
    case 't':
      num_threads = std::stoi(optarg);
      break;
    case 'o':
      operation_str = optarg;
      break;
    default:
      printHelp(argv[0]);
      return -1;
    }
  }

  // Validate mandatory arguments
  if ((ipc_socket.empty() && rpc_endpoint.empty() && rdma_endpoint.empty()) ||
      clients_num <= 0 || data_size_str.empty() || requests_num <= 0 ||
      num_threads <= 0 || operation_str.empty()) {
    std::cerr << "Missing required arguments or invalid values\n";
    printHelp(argv[0]);
    return -1;
  }

  size_t data_size = parseDataSize(data_size_str);
  operation = parseOperationType(operation_str);
  std::string random_data = generateRandomData(data_size);
  std::vector<int> requests_num_per_thread(num_threads,
                                           requests_num / num_threads);
  requests_num_per_thread[0] += requests_num % num_threads;

  std::vector<std::vector<std::shared_ptr<Client>>> clients;
  std::vector<std::vector<std::shared_ptr<RPCClient>>> rpc_clients;
  std::vector<std::vector<ObjectID>> blob_ids(num_threads);
  std::vector<std::vector<ObjectID>> remote_blob_ids(num_threads);
  std::vector<std::vector<ObjectID>> put_remote_blob_ids, blob_id_list;

  std::shared_ptr<Client> client = make_shared<Client>();
  std::shared_ptr<RPCClient> rpc_client = make_shared<RPCClient>();

  std::vector<std::vector<std::shared_ptr<RemoteBlobWriter>>>
      remote_blob_writers(num_threads,
                          std::vector<std::shared_ptr<RemoteBlobWriter>>(
                              requests_num / num_threads));
  remote_blob_writers[0].resize(requests_num / num_threads +
                                requests_num % num_threads);

  std::vector<std::vector<double>> local_latencies(num_threads);
  std::vector<double> latencies;
  std::vector<ObjectMeta> metas11;
  try {
    switch (operation) {
    case OperationType::PUT_BLOB:
      clients = generateClientsForThreads<Client>(
          ipc_socket, rpc_endpoint, clients_num, num_threads, rdma_endpoint);

      MEASURE_AND_PRINT_STATS(
          "PutBlob",
          TestCreate(CreateBlob, clients, random_data, requests_num_per_thread,
                     local_latencies, blob_ids));
      clearBlobs<Client>(clients[0][0], blob_ids);
      clearClients<Client>(clients);
      break;
    case OperationType::GET_BLOB:
      clients = generateClientsForThreads<Client>(
          ipc_socket, rpc_endpoint, clients_num, num_threads, rdma_endpoint);

      VINEYARD_CHECK_OK(client->Connect(ipc_socket));
      // only create `num_thread` blobs
      blob_id_list = PutBlobs(client, num_threads, data_size, num_threads);

      MEASURE_AND_PRINT_STATS("GetBlob",
                              TestGet(GetBlob, clients, requests_num_per_thread,
                                      local_latencies, blob_id_list));

      clearBlobs<Client>(client, blob_id_list);
      clearClients<Client>(clients);
      client->Disconnect();
      break;
    case OperationType::PUT_BLOBS:
      clients = generateClientsForThreads<Client>(
          ipc_socket, rpc_endpoint, clients_num, num_threads, rdma_endpoint);

      MEASURE_AND_PRINT_STATS(
          "PutBlobs",
          TestCreate(CreateBlobs, clients, random_data, requests_num_per_thread,
                     local_latencies, blob_ids));

      clearBlobs<Client>(clients[0][0], blob_ids);
      clearClients<Client>(clients);
      break;
    case OperationType::GET_BLOBS:
      clients = generateClientsForThreads<Client>(
          ipc_socket, rpc_endpoint, clients_num, num_threads, rdma_endpoint);

      VINEYARD_CHECK_OK(client->Connect(ipc_socket));
      blob_id_list = PutBlobs(client, requests_num, data_size, num_threads);

      MEASURE_AND_PRINT_STATS(
          "GetBlobs", TestGet(GetBlobs, clients, requests_num_per_thread,
                              local_latencies, blob_id_list));

      clearBlobs<Client>(client, blob_id_list);
      clearClients<Client>(clients);
      break;
    case OperationType::PUT_REMOTE_BLOB:
      rpc_clients = generateClientsForThreads<RPCClient>(
          ipc_socket, rpc_endpoint, clients_num, num_threads, rdma_endpoint);

      generateRemoteBlobWriters(remote_blob_writers, data_size, random_data);

      MEASURE_AND_PRINT_STATS(
          "PutRemoteBlob",
          TestCreateRemote(CreateRemoteBlob, rpc_clients, remote_blob_writers,
                           local_latencies, remote_blob_ids));

      clearBlobs<RPCClient>(rpc_clients[0][0], remote_blob_ids);
      clearClients<RPCClient>(rpc_clients);
      break;
    case OperationType::GET_REMOTE_BLOB:
      rpc_clients = generateClientsForThreads<RPCClient>(
          ipc_socket, rpc_endpoint, clients_num, num_threads, rdma_endpoint);

      VINEYARD_CHECK_OK(
          rpc_client->Connect(rpc_endpoint, "", "", rdma_endpoint));
      // only create `num_thread` blobs
      put_remote_blob_ids =
          PutRemoteBlobs(rpc_client, num_threads, data_size, num_threads);

      MEASURE_AND_PRINT_STATS(
          "GetRemoteBlob",
          TestGetRemote(GetRemoteBlob, put_remote_blob_ids, rpc_clients,
                        requests_num_per_thread, local_latencies));

      clearBlobs<RPCClient>(rpc_client, put_remote_blob_ids);
      clearClients<RPCClient>(rpc_clients);
      rpc_client->Disconnect();
      break;
    case OperationType::PUT_REMOTE_BLOBS:
      rpc_clients = generateClientsForThreads<RPCClient>(
          ipc_socket, rpc_endpoint, clients_num, num_threads, rdma_endpoint);

      VINEYARD_CHECK_OK(
          rpc_client->Connect(rpc_endpoint, "", "", rdma_endpoint));
      generateRemoteBlobWriters(remote_blob_writers, data_size, random_data);

      MEASURE_AND_PRINT_STATS(
          "PutRemoteBlobs",
          TestCreateRemote(CreateRemoteBlobs, rpc_clients, remote_blob_writers,
                           local_latencies, remote_blob_ids));

      clearBlobs<RPCClient>(rpc_clients[0][0], remote_blob_ids);
      clearClients<RPCClient>(rpc_clients);
      break;
    case OperationType::GET_REMOTE_BLOBS:
      rpc_clients = generateClientsForThreads<RPCClient>(
          ipc_socket, rpc_endpoint, clients_num, num_threads, rdma_endpoint);

      VINEYARD_CHECK_OK(
          rpc_client->Connect(rpc_endpoint, "", "", rdma_endpoint));
      put_remote_blob_ids =
          PutRemoteBlobs(rpc_client, requests_num, data_size, num_threads);

      MEASURE_AND_PRINT_STATS(
          "GetRemoteBlobs",
          TestGetRemote(GetRemoteBlobs, put_remote_blob_ids, rpc_clients,
                        requests_num_per_thread, local_latencies));

      clearBlobs<RPCClient>(rpc_client, put_remote_blob_ids);
      clearClients<RPCClient>(rpc_clients);
      rpc_client->Disconnect();
      break;
    default:
      std::cerr << "Unknown operation type: " << operation_str << "\n";
      printHelp(argv[0]);
      return -1;
    }
  } catch (const exception& e) {
    std::cout << "Caught exception: " << e.what();
    return -1;
  }

  std::cout << "Passed benchmark suite test.";
  return 0;
}
