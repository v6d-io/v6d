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
#include <set>
#include <string>
#include <vector>

#include "arrow/buffer_builder.h"
#include "common/util/logging.h"

#include "fuse/adaptors/chunk_buffer/chunk_buffer.h"

uint rand_seed = 1;
std::string randomString(int64_t len) {
  std::vector<char> alpha = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                             'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                             's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
  std::string result = "";
  for (int i = 0; i < len; i++)
    result = result + alpha[rand_r(&rand_seed) % alpha.size()];

  return result;
}

std::vector<int64_t> randomChunks(int n, const int64_t len) {
  std::set<int64_t> res;
  for (int i = 0; i < n; i++) {
    int t = rand_r(&rand_seed) % len;
    res.emplace(t);
  }
  res.emplace(0);
  res.emplace(len - 1);

  return std::vector<int64_t>(res.begin(), res.end());
}
namespace vfi = vineyard::fuse::internal;
int main() {
  constexpr int64_t len = 1000;
  char buf[1000];

  int chunks_num = 20;
  std::string test = randomString(len);

  // test Write(const void* data, int64_t nbytes)
  {
    auto cb = std::make_shared<vfi::ChunkBuffer>();
    auto chunks = randomChunks(chunks_num, len);

    for (size_t i = 1; i < chunks.size(); i++) {
      cb->Write(&test[chunks[i - 1]], chunks[i] - chunks[i - 1] + 1);
    }
    for (int64_t i = 0; i < (int64_t) test.size(); ++i) {
      for (int64_t j = 0; j < (int64_t) test.size(); ++i) {
        int64_t chunk_len = i - j + 1;
        cb->readAt(i, chunk_len, buf);
        for (int64_t k = 0; k < chunk_len; k++) {
          CHECK_EQ(buf[k], test[i + k]);
        }
      }
    }
  }

  // test Write(const std::shared_ptr<arrow::Buffer>& data)
  {
    auto cb = std::make_shared<vfi::ChunkBuffer>();
    auto chunks = randomChunks(chunks_num, len);

    for (int64_t i = 1; i < (int64_t) chunks.size(); i++) {
      auto b = arrow::Buffer::Wrap(&test[chunks[i - 1]],
                                   chunks[i] - chunks[i - 1] + 1);
      cb->Write(b);
    }
    for (int64_t i = 0; i < (int64_t) test.size(); ++i) {
      for (int64_t j = 0; j < (int64_t) test.size(); ++i) {
        int chunk_len = i - j + 1;
        cb->readAt(i, chunk_len, buf);
        for (int k = 0; k < (int64_t) chunk_len; k++) {
          if (buf[k] != test[i + k]) {
            CHECK_EQ(buf[k], test[i + k]);
          }
        }
      }
    }
  }
  return 0;
}
