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
#include "basic/ds/hashmap_mvcc.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

void testHashmapMVCC(Client& client) {
  using hashmap_t = HashmapMVCC<int64_t, double>;
  LOG(INFO) << "entry element size: " << sizeof(hashmap_t::Entry) << " bytes";

  std::shared_ptr<hashmap_t> hashmap;
  VINEYARD_CHECK_OK(hashmap_t::Make(client, 1, hashmap));

  const int n = 4096;
  std::vector<std::shared_ptr<hashmap_t>> hashmaps;
  hashmaps.push_back(hashmap);

  for (int i = 1; i <= n; ++i) {
    std::shared_ptr<hashmap_t> hmap;
    VINEYARD_CHECK_OK(hashmaps[i - 1]->emplace(hmap, i, i * 100.0));
    if (hmap == nullptr) {
      hmap = hashmaps[i - 1];
    }
    hashmaps.emplace_back(hmap);
  }

  for (int i = 1; i <= n; ++i) {
    auto& hmap = hashmaps[i];
    for (int j = 1; j <= i; ++j) {
      auto r = hmap->find(j);
      CHECK(r != hmap->end());
      CHECK_EQ(r->first, j);
      CHECK_DOUBLE_EQ(r->second, j * 100.0);
    }
  }
  LOG(INFO) << "Passed hashmap mvcc tests...";
}

void testHashmapMVCCLarge(Client& client) {
  using hashmap_t = HashmapMVCC<int, double>;

  std::shared_ptr<hashmap_t> hashmap;
  VINEYARD_CHECK_OK(hashmap_t::Make(client, 1, hashmap));

  const int n = 1048576;
  std::vector<std::shared_ptr<hashmap_t>> hashmaps;
  hashmaps.push_back(hashmap);

  for (int i = 1; i <= n; ++i) {
    std::shared_ptr<hashmap_t> hmap;
    VINEYARD_CHECK_OK(hashmaps[i - 1]->emplace(hmap, i, i * 100.0));
    if (hmap == nullptr) {
      hmap = hashmaps[i - 1];
    }
    hashmaps.emplace_back(hmap);
  }

  for (int i = 1; i <= n; ++i) {
    auto& hmap = hashmaps[i];
    auto r = hmap->find(i);
    CHECK(r != hmap->end());
    CHECK_EQ(r->first, i);
    CHECK_DOUBLE_EQ(r->second, i * 100.0);
  }
  LOG(INFO) << "Passed hashmap mvcc large tests...";
}

void testHashmapMVCCView(Client& client) {
  using hashmap_t = HashmapMVCC<int, double>;

  std::shared_ptr<hashmap_t> hashmap;
  VINEYARD_CHECK_OK(hashmap_t::Make(client, 1, hashmap));

  const int n = 1048576;
  std::vector<std::shared_ptr<hashmap_t>> hashmaps;
  hashmaps.push_back(hashmap);

  for (int i = 1; i <= n; ++i) {
    std::shared_ptr<hashmap_t> hmap;
    VINEYARD_CHECK_OK(hashmaps[i - 1]->emplace(hmap, i, i * 100.0));
    if (hmap == nullptr) {
      hmap = hashmaps[i - 1];
    }
    hashmaps.emplace_back(hmap);
  }

  auto& blob_writer = hashmaps[n]->blob_writer();
  auto blob = std::dynamic_pointer_cast<Blob>(blob_writer->Seal(client));
  std::shared_ptr<const hashmap_t> hmapview;
  VINEYARD_CHECK_OK(hashmap_t::View(client, blob, hmapview));

  for (int i = 1; i <= n; ++i) {
    auto r = hmapview->find(i);
    CHECK(r != hmapview->end());
    CHECK_EQ(r->first, i);
    CHECK_DOUBLE_EQ(r->second, i * 100.0);
  }
  LOG(INFO) << "Passed hashmap mvcc view tests...";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./hashmap_mvcc_test <ipc_socket_name>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  testHashmapMVCC(client);
  testHashmapMVCCLarge(client);
  testHashmapMVCCView(client);

  LOG(INFO) << "Passed double hashmap mvcc tests...";

  client.Disconnect();

  return 0;
}
