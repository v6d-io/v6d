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
#include <regex>
#include <string>

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/remote_blob.h"
#include "client/rpc_client.h"
#include "common/util/logging.h"
#include "llm-cache/ds/vineyard_file.h"

namespace vineyard {

void VineyardFile::Construct(const ObjectMeta& meta) {
  Object::Construct(meta);
  if (meta_.GetTypeName() != type_name<VineyardFile>()) {
    return;
  }
  this->path_ = meta_.GetKeyValue("path");
  this->access_time_ = meta_.GetKeyValue<uint64_t>("access_time");
  ObjectID blob_id = meta_.GetMember("buffer")->id();
  RPCClient* rpc_client = reinterpret_cast<RPCClient*>(meta.GetClient());
  if (rpc_client->GetRemoteBlob(blob_id, blob_).ok()) {
    return;
  }
}

Status VineyardFile::Read(void* buffer, size_t size, size_t offset) {
  if (offset + size > blob_->size()) {
    return Status::Invalid("Read out of range");
  }
  memcpy(buffer, blob_->data() + offset, size);
  return Status::OK();
}

Status VineyardFile::Make(std::shared_ptr<VineyardFile>& file,
                          RPCClient& rpc_client, std::string path) {
  std::string origin_path = std::regex_replace(path, std::regex("/+"), "\\/");
  VineyardFileLock lock(rpc_client, path);
  RETURN_ON_ERROR(lock.TryLock());
  ObjectID file_id;
  if (!rpc_client.GetName(origin_path, file_id, false).ok()) {
    return Status::IOError("File " + path + " is not exist.");
  }
  std::shared_ptr<Object> object;
  RETURN_ON_ERROR(rpc_client.GetObject(file_id, object));
  file = std::dynamic_pointer_cast<VineyardFile>(object);
  if (file->blob_ == nullptr) {
    return Status::IOError("File " + path + " is not exist.");
  }
  return Status::OK();
}

Status VineyardFileBuilder::Make(std::shared_ptr<VineyardFileBuilder>& builder,
                                 RPCClient& client, std::string path,
                                 size_t size) {
  std::string actural_path;
  std::string origin_path = std::regex_replace(path, std::regex("/+"), "\\/");
  builder = std::make_shared<VineyardFileBuilder>(origin_path);
  builder->lock = std::make_unique<VineyardFileLock>(client, path);
  RETURN_ON_ERROR(builder->lock->TryLock());
  ObjectID id;
  if (client.GetName(origin_path, id).ok()) {
    return Status::Invalid("File already exists");
  }
  builder->writer_ = std::make_shared<RemoteBlobWriter>(size);
  return Status::OK();
}

std::shared_ptr<Object> VineyardFileBuilder::SealAndPersist(RPCClient& client) {
  VINEYARD_CHECK_OK(this->Build(client));
  RPCClient& rpc_client = reinterpret_cast<RPCClient&>(client);

  std::shared_ptr<VineyardFile> vineyardFile = std::make_shared<VineyardFile>();
  ObjectMeta blob_meta;
  rpc_client.CreateRemoteBlob(writer_, blob_meta);
  rpc_client.Persist(blob_meta.GetId());
  vineyardFile->meta_.AddMember("buffer", blob_meta);
  vineyardFile->meta_.AddKeyValue("path", path_);
  vineyardFile->meta_.SetTypeName(type_name<VineyardFile>());

  auto access_time = std::chrono::system_clock::now().time_since_epoch();
  vineyardFile->meta_.AddKeyValue(
      "access_time",
      std::chrono::duration_cast<std::chrono::nanoseconds>(access_time)
          .count());
  VINEYARD_CHECK_OK(
      client.CreateMetaData(vineyardFile->meta_, vineyardFile->id_));
  client.Persist(vineyardFile->id_);
  Status status = client.PutName(vineyardFile->id_, path_);

  lock.reset();
  return vineyardFile;
}

Status VineyardFileBuilder::Write(const void* buffer, size_t size,
                                  size_t offset) {
  if (writer_ == nullptr) {
    return Status::Invalid("VineyardFileBuilder has not been initialized");
  }
  if (offset + size > writer_->size()) {
    return Status::Invalid("Write out of range");
  }
  memcpy(writer_->data() + offset, buffer, size);
  return Status::OK();
}

}  // namespace vineyard
