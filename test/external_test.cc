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

#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/tensor.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./external_test.cc <ipc_socket>");
    return 1;
  }

  std::string ipc_socket = std::string(argv[1]);

  auto randStr = [](size_t length) {
    std::string templ =
        "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`~!@"
        "#$%^&*()_+-=?<>,.;':";
    std::string buffer;
    std::random_device rd;
    std::default_random_engine random(rd());
    for (size_t i = 0; i < length; ++i) {
      auto idx = random() % templ.size();
      buffer += templ[idx];
    }
    return buffer;
  };

  auto generated_test_data = [&](std::map<std::string, std::string>& db,
                                 size_t test_size) {
    for (size_t i = 0; i < test_size; ++i) {
      db.insert(std::make_pair(randStr(10), randStr(50)));
    }
  };

  auto create_external_object = [](ExternalClient& client,
                                   std::string const& oid,
                                   std::string const& data) {
    ExternalID eid = ExternalIDFromString(oid);
    std::unique_ptr<vineyard::BlobWriter> blob;
    VINEYARD_CHECK_OK(client.CreateBlob(eid, data.size(), 0, blob));
    auto buffer = reinterpret_cast<uint8_t*>(blob->data());
    memcpy(buffer, data.c_str(), data.size());
    return eid;
  };

  auto get_external_objects = [](ExternalClient& client,
                                 std::vector<ExternalID>& eids) {
    std::vector<std::string> results;
    std::map<ExternalID, ExternalPayload> payloads;
    std::map<ExternalID, std::shared_ptr<arrow::Buffer>> buffers;
    VINEYARD_CHECK_OK(client.GetBlobs(
        std::set<ExternalID>(eids.begin(), eids.end()), payloads, buffers));
    for (size_t i = 0; i < eids.size(); ++i) {
      std::shared_ptr<arrow::Buffer> buff = buffers.find(eids[i])->second;
      ExternalPayload payload = payloads.find(eids[i])->second;
      char* data = reinterpret_cast<char*>(const_cast<uint8_t*>(buff->data()));
      results.emplace_back(std::string(data, buff->size()));
    }
    return results;
  };

  auto check_results = [&](std::vector<ExternalID>& eids,
                           std::vector<std::string> results,
                           std::map<std::string, std::string> answer) {
    CHECK_EQ(eids.size(), results.size());
    for (size_t i = 0; i < eids.size(); ++i) {
      auto search = answer.find(ExternalIDToString(eids[i]));
      CHECK(search != answer.end());
      CHECK(search->second == results[i]);
    }
  };

  {
    std::map<std::string, std::string> answer;
    generated_test_data(answer, 64);

    ExternalClient client;
    VINEYARD_CHECK_OK(client.Open(ipc_socket));

    std::vector<std::string> eids;
    for (auto it = answer.begin(); it != answer.end(); ++it) {
      eids.emplace_back(create_external_object(client, it->first, it->second));
    }

    auto results = get_external_objects(client, eids);

    check_results(eids, results, answer);

    client.CloseSession();
  }
}
