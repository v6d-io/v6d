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

#ifndef MODULES_LLM_CACHE_DS_VINEYARD_FILE_H_
#define MODULES_LLM_CACHE_DS_VINEYARD_FILE_H_

#include <memory>
#include <regex>
#include <string>

#include "client/ds/remote_blob.h"
#include "client/rpc_client.h"

namespace vineyard {

class VineyardFileBuilder;

class VineyardFileLock {
 public:
  explicit VineyardFileLock(RPCClient& client, std::string path)
      : path_(path), client_(client) {}

  ~VineyardFileLock() { Unlock(); }

  Status TryLock() {
    bool result = false;
    std::string origin_path =
        std::regex_replace(path_, std::regex("/+"), "\\/");
    client_.TryAcquireLock(origin_path, result, lock_path_);
    if (!result) {
      return Status::Invalid("Failed to acquire lock for file: " + path_);
    }
    return Status::OK();
  }

 private:
  Status Unlock() {
    if (!lock_path_.empty()) {
      // unlock
      bool result = false;
      do {
        client_.TryReleaseLock(lock_path_, result);
      } while (!result);
    }
    return Status::OK();
  }

 private:
  std::string path_;
  std::string lock_path_;
  RPCClient& client_;
};

class VineyardFile : public vineyard::Registered<VineyardFile> {
 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::unique_ptr<Object>(new VineyardFile());
  }

  void Construct(const ObjectMeta& meta) override;

  Status Read(void* buffer, size_t size, size_t offset);

  static Status Make(std::shared_ptr<VineyardFile>& file, RPCClient& client,
                     std::string path);

  size_t Size() { return blob_->size(); }

  uint64_t AccessTime() { return access_time_; }

 private:
  std::shared_ptr<RemoteBlob> blob_;
  std::string path_;
  uint64_t access_time_;

  friend class VineyardFileBuilder;
};

class VineyardFileBuilder {
 public:
  static Status Make(std::shared_ptr<VineyardFileBuilder>& builder,
                     RPCClient& client, std::string path, size_t size);

  ~VineyardFileBuilder() {}

  Status Build(RPCClient& client) { return Status::OK(); }

  std::shared_ptr<Object> SealAndPersist(RPCClient& client);

  Status Write(const void* buffer, size_t size, size_t offset);

  explicit VineyardFileBuilder(std::string path) : path_(path) {}

  size_t Size() { return writer_->size(); }

 private:
  std::shared_ptr<RemoteBlobWriter> writer_;
  std::string path_;
  std::unique_ptr<VineyardFileLock> lock;
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_DS_VINEYARD_FILE_H_
