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

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/compression/compressor.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

std::string generate_random(const size_t len) {
  static const char alphanum[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  std::stringstream ss;
  for (size_t i = 0; i < len; ++i) {
    ss << alphanum[rand() % (sizeof(alphanum) - 1)];
  }

  return ss.str();
}

void CompressSingleBlobTest() {
  // const size_t length = 1024 * 1024 * 256;
  const size_t length = 1024 * 1024;

  std::string data = generate_random(length);
  std::vector<std::pair<uint8_t*, size_t>> compressed;
  std::string decompressed(length, '\0');

  // compression
  {
    std::shared_ptr<Compressor> compressor = std::make_shared<Compressor>();
    VINEYARD_CHECK_OK(compressor->Compress(data.data(), length));

    while (true) {
      void* data = nullptr;
      size_t size;
      auto s = compressor->Pull(data, size);
      LOG(INFO) << "size: " << size << ", s: " << s;
      if (s.IsStreamDrained()) {
        break;
      }
      if (size > 0) {
        uint8_t* chunk = static_cast<uint8_t*>(malloc(size));
        memcpy(chunk, data, size);
        compressed.emplace_back(chunk, size);
      }
    }
    LOG(INFO) << "finish compression, chunk size = " << compressed.size();
  }

  // decompression
  {
    std::shared_ptr<Decompressor> decompressor =
        std::make_shared<Decompressor>();
    size_t offset = 0;
    for (auto& chunk : compressed) {
      void* data = nullptr;
      size_t size;
      VINEYARD_CHECK_OK(decompressor->Buffer(data, size));
      LOG(INFO) << "decompress: size = " << size
                << ", input size = " << chunk.second;
      VINEYARD_ASSERT(size >= chunk.second);
      memcpy(data, chunk.first, chunk.second);
      VINEYARD_CHECK_OK(decompressor->Decompress(chunk.second));
      size_t decompressed_size = 0, decompressed_offset = offset;
      while (true) {
        if (decompressed_offset >= decompressed.length()) {
          break;
        }
        uint8_t* pointer =
            reinterpret_cast<uint8_t*>(const_cast<char*>(decompressed.data())) +
            decompressed_offset;
        size_t size = 0;
        auto s = decompressor->Pull(
            pointer, decompressed.length() - decompressed_offset, size);
        LOG(INFO) << "size = " << size << ", status = " << s
                  << ", decompressed.length() - decompressed_offset = "
                  << (decompressed.length() - decompressed_offset);
        if (s.IsStreamDrained()) {
          break;
        }
        if (size > 0) {
          decompressed_offset += size;
          decompressed_size += size;
        }
      }
      offset += decompressed_size;
    }
    LOG(INFO) << "finish decompression, result size = " << decompressed.size();
  }

  // validate
  VINEYARD_ASSERT(data == decompressed);
}

int main(int argc, char** argv) {
  CompressSingleBlobTest();

  LOG(INFO) << "Passed compressor tests...";
  return 0;
}
