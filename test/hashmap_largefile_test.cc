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

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/array.h"
#include "basic/ds/hashmap.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/env.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage ./hashmap_test <ipc_socket_name> <datafile>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);
  std::string datafile = std::string(argv[2]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  std::vector<int64_t> ns;
  {
    int64_t n;
    std::FILE* fp = fopen(datafile.c_str(), "r");
    while (fscanf(fp, "%" SCNd64, &n) != EOF) {
      ns.emplace_back(n);
    }
    fclose(fp);
  }

  LOG(INFO) << "number size: " << ns.size();

  HashmapBuilder<int64_t, uint64_t> builder(client);
  for (int64_t n : ns) {
    builder.emplace(n, n);
    if (builder.bucket_count() > ns.size() * 16) {
      LOG(ERROR) << "bucket count exceed limit: " << builder.bucket_count();
      exit(-1);
    }
  }

  LOG(INFO) << "hashmap size: " << builder.size()
            << ", bucket count: " << builder.bucket_count()
            << ", load factor: " << builder.load_factor();

  auto sealed_hashmap = std::dynamic_pointer_cast<Hashmap<int64_t, uint64_t>>(
      builder.Seal(client));

  LOG(INFO) << "sealed hashmap size: " << sealed_hashmap->size()
            << ", bucket count: " << sealed_hashmap->bucket_count()
            << ", load factor: " << sealed_hashmap->load_factor()
            << ", memory usage: "
            << prettyprint_memory_size(sealed_hashmap->meta().MemoryUsage());

  for (auto const& kv : *sealed_hashmap) {
    CHECK_EQ(kv.first, kv.second);
  }

  LOG(INFO) << "Passed double hashmap tests...";

  client.Disconnect();

  return 0;
}
