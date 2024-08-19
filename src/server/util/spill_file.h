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

#ifndef SRC_SERVER_UTIL_SPILL_FILE_H_
#define SRC_SERVER_UTIL_SPILL_FILE_H_

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "common/memory/payload.h"
#include "common/util/arrow.h"
#include "common/util/status.h"
#include "common/util/uuid.h"
#include "server/util/file_io_adaptor.h"

namespace vineyard {
namespace io {

class FileLocker {
 public:
  static FileLocker& getInstance() {
    static FileLocker instance;
    return instance;
  }

  void lockForWrite(const ObjectID& id) {
    std::unique_lock<std::shared_mutex> mapLock(mapMutex_);
    fileLocks_[id].lock();
  }

  void unlockForWrite(const ObjectID& id) {
    std::unique_lock<std::shared_mutex> mapLock(mapMutex_);
    fileLocks_[id].unlock();
  }

  void lockForRead(const ObjectID& id) {
    std::unique_lock<std::shared_mutex> mapLock(mapMutex_);
    fileLocks_[id].lock_shared();
  }

  void unlockForRead(const ObjectID& id) {
    std::unique_lock<std::shared_mutex> mapLock(mapMutex_);
    fileLocks_[id].unlock_shared();
  }

 private:
  FileLocker() = default;
  std::shared_mutex mapMutex_;
  std::unordered_map<ObjectID, std::shared_mutex> fileLocks_;
};

/*
  For each spilled file, the disk-format is:
    - object_id: uint64
    - data_size: uint64
    - content: uint8[data_size]
*/
class SpillFileWriter {
 public:
  SpillFileWriter() = delete;

  explicit SpillFileWriter(const std::string& spill_path)
      : spill_path_(spill_path) {}

  SpillFileWriter(const SpillFileWriter&) = delete;

  SpillFileWriter& operator=(const SpillFileWriter&) = delete;

  ~SpillFileWriter() {
    if (io_adaptor_) {
      DISCARD_ARROW_ERROR(io_adaptor_->Flush());
    }
  }

  Status Write(const std::shared_ptr<Payload>& payload);
  Status Sync();

 private:
  Status Init(const ObjectID object_id);

  std::string spill_path_;
  std::mutex io_mutex_;
  std::unique_ptr<FileIOAdaptor> io_adaptor_ = nullptr;
};

class SpillFileReader {
 public:
  SpillFileReader() = delete;

  explicit SpillFileReader(const std::string& spill_path)
      : spill_path_(spill_path) {}

  SpillFileReader(const SpillFileReader&) = delete;

  SpillFileReader& operator=(const SpillFileReader&) = delete;

  ~SpillFileReader() = default;

  Status Read(const std::shared_ptr<Payload>& payload,
              const std::shared_ptr<BulkStore>& bulk_store);

 private:
  Status Init(const ObjectID object_id);

  // Delete should be called after Init()
  Status Delete_(const ObjectID id);

  std::string spill_path_;
  std::unique_ptr<FileIOAdaptor> io_adaptor_ = nullptr;
};

}  // namespace io
}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_SPILL_FILE_H_
