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

#ifndef SRC_SERVER_UTIL_FILE_IO_H_
#define SRC_SERVER_UTIL_FILE_IO_H_

#include <cstddef>
#include <cstdint>
#include "common/util/status.h"
#include "common/util/uuid.h"
#include "io/io/i_io_adaptor.h"
#include "io/io/io_factory.h"
#include "io/io/local_io_adaptor.h"

namespace util {
constexpr auto spill_path_prefix = "tmp/spilled/";
class SpillWriteFile {
 public:
  SpillWriteFile(const std::string& file_name) {
    spill_path_ = spill_path_prefix + file_name;
  };

  SpillWriteFile(const SpillWriteFile&) = delete;
  SpillWriteFile& operator=(const SpillWriteFile&) = delete;
  ~SpillWriteFile() = default;

  vineyard::Status Open();
  vineyard::Status Write(const char* data, const size_t& size,
                         const vineyard::ObjectID& object_id);
  vineyard::Status Sync();

 private:
  std::string spill_path_;
  std::unique_ptr<vineyard::IIOAdaptor> io_adaptor_ = nullptr;
};

class SpillReadFile {
 public:
  SpillReadFile(const std::string& file_name) {
    spill_path_ = spill_path_prefix + file_name;
  }

  SpillReadFile(const SpillReadFile&) = delete;
  SpillReadFile& operator=(const SpillReadFile&) = delete;
  ~SpillReadFile();

  vineyard::Status Open();
  vineyard::Status Read(size_t n, char* result);

 private:
  std::string spill_path_;
  std::unique_ptr<vineyard::IIOAdaptor> io_adaptor_ = nullptr;
};

void PutFixed32(std::string* dst, uint32_t value);

void PutFixed64(std::string* dst, uint64_t value);

inline void EncodeFixed32(char* dst, uint32_t value) {
  uint8_t* const buffer = reinterpret_cast<uint8_t*>(dst);

  buffer[0] = static_cast<uint8_t>(value);
  buffer[1] = static_cast<uint8_t>(value >> 8);
  buffer[2] = static_cast<uint8_t>(value >> 16);
  buffer[3] = static_cast<uint8_t>(value >> 24);
}

inline void EncodeFixed64(char* dst, uint64_t value) {
  uint8_t* const buffer = reinterpret_cast<uint8_t*>(dst);

  buffer[0] = static_cast<uint8_t>(value);
  buffer[1] = static_cast<uint8_t>(value >> 8);
  buffer[2] = static_cast<uint8_t>(value >> 16);
  buffer[3] = static_cast<uint8_t>(value >> 24);
  buffer[4] = static_cast<uint8_t>(value >> 32);
  buffer[5] = static_cast<uint8_t>(value >> 40);
  buffer[6] = static_cast<uint8_t>(value >> 48);
  buffer[7] = static_cast<uint8_t>(value >> 56);
}

inline uint32_t DecodeFixed32(const char* src) {
  const uint8_t* const buffer = reinterpret_cast<const uint8_t*>(src);

  return (static_cast<uint32_t>(buffer[0])) |
         (static_cast<uint32_t>(buffer[1]) << 8) |
         (static_cast<uint32_t>(buffer[2]) << 16) |
         (static_cast<uint32_t>(buffer[3]) << 24);
}

inline uint64_t DecodeFixed64(const char* ptr) {
  const uint8_t* const buffer = reinterpret_cast<const uint8_t*>(ptr);

  return (static_cast<uint64_t>(buffer[0])) |
         (static_cast<uint64_t>(buffer[1]) << 8) |
         (static_cast<uint64_t>(buffer[2]) << 16) |
         (static_cast<uint64_t>(buffer[3]) << 24) |
         (static_cast<uint64_t>(buffer[4]) << 32) |
         (static_cast<uint64_t>(buffer[5]) << 40) |
         (static_cast<uint64_t>(buffer[6]) << 48) |
         (static_cast<uint64_t>(buffer[7]) << 56);
}

}  // namespace util
#endif