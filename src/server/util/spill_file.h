/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#ifndef SRC_SERVER_UTIL_SPILL_FILE_H_
#define SRC_SERVER_UTIL_SPILL_FILE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "common/memory/payload.h"
#include "common/util/arrow.h"
#include "common/util/status.h"
#include "common/util/uuid.h"
#include "server/util/file_io_adaptor.h"

namespace util {

/*
  For each spilled file, the disk-format is:
    - object_id: uint64
    - data_size: uint64
    - content: uint8[data_size]
*/
class SpillWriteFile {
 public:
  SpillWriteFile() = delete;

  explicit SpillWriteFile(const std::string& spill_path)
      : spill_path_(spill_path) {}

  SpillWriteFile(const SpillWriteFile&) = delete;

  SpillWriteFile& operator=(const SpillWriteFile&) = delete;

  ~SpillWriteFile() {
    if (io_adaptor_) {
      DISCARD_ARROW_ERROR(io_adaptor_->Flush());
    }
  }

  vineyard::Status Write(const std::shared_ptr<vineyard::Payload>& payload);
  vineyard::Status Sync();

 private:
  vineyard::Status Init(uint64_t object_id);
  // TODO(ZjuYTW): change to string_view
  std::string spill_path_;
  std::unique_ptr<util::FileIOAdaptor> io_adaptor_ = nullptr;
};

class SpillReadFile {
 public:
  SpillReadFile() = delete;

  explicit SpillReadFile(const std::string& spill_path)
      : spill_path_(spill_path) {}

  SpillReadFile(const SpillReadFile&) = delete;

  SpillReadFile& operator=(const SpillReadFile&) = delete;

  ~SpillReadFile() = default;

  vineyard::Status Read(std::shared_ptr<vineyard::Payload>& payload,
                        std::shared_ptr<vineyard::BulkStore> bulk_store_ptr);

 private:
  vineyard::Status Init(uint64_t object_id);

  // Delete should be called after Init()
  vineyard::Status Delete_(const vineyard::ObjectID& id);
  std::string spill_path_;
  std::unique_ptr<util::FileIOAdaptor> io_adaptor_ = nullptr;
};

void PutFixed64(std::string* dst, uint64_t value);

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
#endif  // SRC_SERVER_UTIL_SPILL_FILE_H_
