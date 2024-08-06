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

#include <map>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/remote_blob.h"
#include "client/rpc_client.h"

namespace vineyard {

class VineyardFileBuilder;

class VineyardFile : public vineyard::Registered<VineyardFile> {
 public:
  VineyardFile() = default;

  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::unique_ptr<Object>(new VineyardFile());
  }

  void Construct(const ObjectMeta& meta) override;

  Status Read(void* buffer, size_t size, size_t offset);

  static Status Make(std::shared_ptr<VineyardFile>& file, RPCClient& rpc_client,
                     Client& ipc_client, std::string path);

  static Status BatchedMake(std::vector<std::shared_ptr<VineyardFile>>& files,
                            RPCClient& rpc_client, Client& ipc_client,
                            const std::vector<std::string>& path);

  size_t Size() { return buffer_->size(); }

  uint64_t AccessTime() { return access_time_; }

 private:
  static Status BatchedGetObjects(
      Client& client, RPCClient& rpc_client,
      std::map<InstanceID, std::vector<ObjectMeta>>& instance_to_metas,
      std::unordered_map<ObjectID, std::shared_ptr<VineyardFile>>& id_to_files);

  std::shared_ptr<Buffer> buffer_;
  std::string path_;
  uint64_t access_time_;

  friend class VineyardFileBuilder;
};

class VineyardFileBuilder {
 public:
  static Status Make(std::shared_ptr<VineyardFileBuilder>& builder,
                     RPCClient& rpc_client, Client& ipc_client,
                     std::string path, size_t size);

  ~VineyardFileBuilder() {}

  Status Build(RPCClient& rpc_client, Client& ipc_client) {
    return Status::OK();
  }

  std::shared_ptr<Object> SealAndPersist(RPCClient& rpc_client,
                                         Client& ipc_client);

  Status Write(const void* buffer, size_t size, size_t offset);

  explicit VineyardFileBuilder(std::string path) : path_(path) {}

  static std::vector<std::shared_ptr<Object>> BatchedSealAndPersist(
      RPCClient& rpc_client, Client& ipc_client,
      std::vector<std::shared_ptr<VineyardFileBuilder>>& builders);

  size_t Size() {
    if (writer_) {
      return writer_->size();
    }
    return remote_writer_->size();
  }

 private:
  std::shared_ptr<RemoteBlobWriter> remote_writer_;
  std::unique_ptr<BlobWriter> writer_;
  std::string path_;
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_DS_VINEYARD_FILE_H_
