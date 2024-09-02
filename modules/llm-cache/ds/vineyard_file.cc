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

#include <algorithm>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/object_meta.h"
#include "client/ds/remote_blob.h"
#include "client/rpc_client.h"
#include "common/util/logging.h"
#include "llm-cache/ds/vineyard_file.h"
#include "llm-cache/thread_group.h"

namespace vineyard {

void VineyardFile::Construct(const ObjectMeta& meta) {
  Object::Construct(meta);
  if (meta_.GetTypeName() != type_name<VineyardFile>()) {
    return;
  }
  this->path_ = meta_.GetKeyValue("path");
  this->access_time_ = meta_.GetKeyValue<uint64_t>("access_time");
  ObjectMeta blob_meta;
  VINEYARD_CHECK_OK(meta_.GetMemberMeta("buffer", blob_meta));
  ObjectID blob_id = blob_meta.GetId();
  VINEYARD_CHECK_OK(meta.GetBuffer(blob_id, buffer_));
}

Status VineyardFile::Read(void* buffer, size_t size, size_t offset) {
  if (buffer == nullptr) {
    return Status::Invalid("Buffer is nullptr");
  }
  if (static_cast<int64_t>(offset + size) > buffer_->size()) {
    return Status::Invalid("Read out of range");
  }
  memcpy(buffer, buffer_->data() + offset, size);
  return Status::OK();
}

Status VineyardFile::Make(std::shared_ptr<VineyardFile>& file,
                          RPCClient& rpc_client, Client& ipc_client,
                          std::string path) {
  std::string origin_path = std::regex_replace(path, std::regex("/+"), "\\/");
  ObjectID file_id;
  ObjectMeta meta;
  ObjectMeta object_meta;
  std::shared_ptr<Object> object;
  if (ipc_client.Connected()) {
    if (!ipc_client.GetName(origin_path, file_id, false).ok()) {
      return Status::IOError("File " + path + " is not exist.");
    }
    RETURN_ON_ERROR(ipc_client.GetMetaData(file_id, meta, true));
    if (meta.GetInstanceId() == ipc_client.instance_id()) {
      object = ipc_client.GetObject(file_id);
      file = std::dynamic_pointer_cast<VineyardFile>(object);
      if (file->buffer_ == nullptr) {
        return Status::IOError("File " + path + " is not exist.");
      }
      return Status::OK();
    } else {
      RETURN_ON_ERROR(rpc_client.GetMetaData(file_id, object_meta, true));
    }
  } else {
    if (!rpc_client.GetName(origin_path, file_id, false).ok()) {
      return Status::IOError("File " + path + " is not exist.");
    }
    RETURN_ON_ERROR(rpc_client.GetMetaData(file_id, object_meta, true));
  }

  std::map<InstanceID, json> cluster_info;
  RETURN_ON_ERROR(rpc_client.ClusterInfo(cluster_info));
  if (object_meta.GetInstanceId() == rpc_client.remote_instance_id()) {
    object = rpc_client.GetObject(file_id);
  } else {
    std::string rpc_endpoint =
        cluster_info[object_meta.GetInstanceId()]["rpc_endpoint"]
            .get<std::string>();
    std::string rdma_endpoint =
        cluster_info[object_meta.GetInstanceId()]["rdma_endpoint"]
            .get<std::string>();
    RPCClient remote_rpc_client;
    RETURN_ON_ERROR(
        remote_rpc_client.Connect(rpc_endpoint, "", "", rdma_endpoint));
    object = remote_rpc_client.GetObject(file_id);
    ObjectID buffer_id = object_meta.GetMember("buffer")->id();
    std::shared_ptr<RemoteBlob> blob;
    RETURN_ON_ERROR(remote_rpc_client.GetRemoteBlob(buffer_id, blob));
    std::dynamic_pointer_cast<VineyardFile>(object)->buffer_ = blob->Buffer();
  }

  file = std::dynamic_pointer_cast<VineyardFile>(object);
  if (file->buffer_ == nullptr) {
    return Status::IOError("File " + path + " is not exist.");
  }
  return Status::OK();
}

Status VineyardFile::BatchedGetObjects(
    Client& client, RPCClient& rpc_client,
    std::map<InstanceID, std::vector<ObjectMeta>>& instance_to_metas,
    std::unordered_map<ObjectID, std::shared_ptr<VineyardFile>>& id_to_files) {
  std::map<InstanceID, json> cluster_info;
  RETURN_ON_ERROR(rpc_client.ClusterInfo(cluster_info));
  auto fn = [&](std::pair<const InstanceID, std::vector<ObjectMeta>>&
                    instance_to_meta) -> Status {
    std::vector<std::shared_ptr<Object>> file_objects;
    if (client.Connected() && instance_to_meta.first == client.instance_id()) {
      std::vector<ObjectID> ids(instance_to_meta.second.size());
      for (size_t i = 0; i < instance_to_meta.second.size(); ++i) {
        ids[i] = instance_to_meta.second[i].GetId();
      }
      instance_to_meta.second.clear();
      RETURN_ON_ERROR(client.GetMetaData(ids, instance_to_meta.second, false));
      file_objects = client.GetObjects(instance_to_meta.second);
    } else {
      if (rpc_client.remote_instance_id() == instance_to_meta.first) {
        std::vector<ObjectID> ids(instance_to_meta.second.size());
        for (size_t i = 0; i < instance_to_meta.second.size(); ++i) {
          ids[i] = instance_to_meta.second[i].GetId();
        }
        instance_to_meta.second.clear();
        RETURN_ON_ERROR(
            rpc_client.GetMetaData(ids, instance_to_meta.second, false));
        RETURN_ON_ERROR(rpc_client.BatchedGetObjects(instance_to_meta.second,
                                                     file_objects));
      } else {
        std::vector<ObjectID> ids(instance_to_meta.second.size());
        for (size_t i = 0; i < instance_to_meta.second.size(); ++i) {
          ids[i] = instance_to_meta.second[i].GetId();
        }
        std::string rpc_endpoint =
            cluster_info[instance_to_meta.first]["rpc_endpoint"]
                .get<std::string>();
        std::string rdma_endpoint =
            cluster_info[instance_to_meta.first]["rdma_endpoint"]
                .get<std::string>();
        RPCClient remote_rpc_client;
        RETURN_ON_ERROR(
            remote_rpc_client.Connect(rpc_endpoint, "", "", rdma_endpoint));

        /*
         * Because the GetMeta will not set buffer that is not created by the
         * caller rpc_client, so we need to get meta again.
         */
        instance_to_meta.second.clear();
        RETURN_ON_ERROR(
            remote_rpc_client.GetMetaData(ids, instance_to_meta.second, false));
        RETURN_ON_ERROR(remote_rpc_client.BatchedGetObjects(
            instance_to_meta.second, file_objects));
      }
    }
    for (size_t i = 0; i < instance_to_meta.second.size(); ++i) {
      id_to_files[instance_to_meta.second[i].GetId()] =
          std::dynamic_pointer_cast<VineyardFile>(file_objects[i]);
    }
    return Status::OK();
  };

  parallel::ThreadGroup tg(
      std::min(instance_to_metas.size(),
               static_cast<size_t>(std::thread::hardware_concurrency())));
  std::vector<parallel::ThreadGroup::tid_t> tids(instance_to_metas.size());
  int index = 0;
  for (auto& instance_to_meta : instance_to_metas) {
    tids[index] = tg.AddTask(fn, instance_to_meta);
    index++;
  }

  std::vector<Status> taskResults(instance_to_metas.size(), Status::OK());
  for (size_t i = 0; i < instance_to_metas.size(); ++i) {
    taskResults[i] = tg.TaskResult(tids[i]);
  }

  return Status::OK();
}

Status VineyardFile::BatchedMake(
    std::vector<std::shared_ptr<VineyardFile>>& files, RPCClient& rpc_client,
    Client& ipc_client, const std::vector<std::string>& paths) {
  std::vector<std::string> origin_paths;
  std::vector<ObjectID> file_ids;

  for (auto const& path : paths) {
    origin_paths.push_back(std::regex_replace(path, std::regex("/+"), "\\/"));
  }

  std::vector<ObjectMeta> file_metas;
  std::map<InstanceID, vineyard::json> clusterInfo;
  RETURN_ON_ERROR(rpc_client.ClusterInfo(clusterInfo));
  std::map<InstanceID, std::vector<ObjectMeta>> instance_to_metas;
  if (ipc_client.Connected()) {
    for (auto const& path : origin_paths) {
      ObjectID file_id;
      if (ipc_client.GetName(path, file_id, false).ok()) {
        file_ids.push_back(file_id);
      } else {
        break;
      }
      RETURN_ON_ERROR(ipc_client.GetMetaData(file_ids, file_metas, true));
    }
  } else {
    // RPC
    for (auto const& path : origin_paths) {
      ObjectID file_id;
      if (rpc_client.GetName(path, file_id, false).ok()) {
        file_ids.push_back(file_id);
      } else {
        break;
      }
    }
    RETURN_ON_ERROR(rpc_client.GetMetaData(file_ids, file_metas, true));
  }
  for (const auto& meta : file_metas) {
    instance_to_metas[meta.GetInstanceId()].push_back(meta);
  }
  std::unordered_map<ObjectID, std::shared_ptr<VineyardFile>> id_to_files;
  RETURN_ON_ERROR(BatchedGetObjects(ipc_client, rpc_client, instance_to_metas,
                                    id_to_files));
  for (auto const& meta : file_metas) {
    if (id_to_files.find(meta.GetId()) != id_to_files.end()) {
      files.push_back(id_to_files[meta.GetId()]);
    } else {
      break;
    }
  }
  return Status::OK();
}

Status VineyardFileBuilder::Make(std::shared_ptr<VineyardFileBuilder>& builder,
                                 RPCClient& rpc_client, Client& ipc_client,
                                 std::string path, size_t size) {
  std::string actural_path;
  std::string origin_path = std::regex_replace(path, std::regex("/+"), "\\/");
  builder = std::make_shared<VineyardFileBuilder>(origin_path);
  ObjectID id;
  if (ipc_client.Connected()) {
    if (ipc_client.GetName(origin_path, id).ok()) {
      return Status::Invalid("File already exists");
    }
    RETURN_ON_ERROR(ipc_client.CreateBlob(size, builder->writer_));
  } else {
    if (rpc_client.GetName(origin_path, id).ok()) {
      return Status::Invalid("File already exists");
    }
    builder->remote_writer_ = std::make_shared<RemoteBlobWriter>(size);
  }
  return Status::OK();
}

std::shared_ptr<Object> VineyardFileBuilder::SealAndPersist(
    RPCClient& rpc_client, Client& ipc_client) {
  VINEYARD_CHECK_OK(this->Build(rpc_client, ipc_client));

  std::shared_ptr<VineyardFile> vineyardFile = std::make_shared<VineyardFile>();
  ObjectMeta blob_meta;
  if (ipc_client.Connected()) {
    std::shared_ptr<Object> object;
    VINEYARD_DISCARD(writer_->Shrink(ipc_client, writer_->size()));
    VINEYARD_CHECK_OK(writer_->Seal(ipc_client, object));
    blob_meta = object->meta();
    VINEYARD_CHECK_OK(ipc_client.Persist(blob_meta.GetId()));
  } else {
    VINEYARD_CHECK_OK(rpc_client.CreateRemoteBlob(remote_writer_, blob_meta));
    VINEYARD_CHECK_OK(rpc_client.Persist(blob_meta.GetId()));
  }
  vineyardFile->meta_.AddMember("buffer", blob_meta);
  vineyardFile->meta_.AddKeyValue("path", path_);
  vineyardFile->meta_.AddKeyValue("size", Size());
  vineyardFile->meta_.SetTypeName(type_name<VineyardFile>());

  auto access_time = std::chrono::system_clock::now().time_since_epoch();
  vineyardFile->meta_.AddKeyValue(
      "access_time",
      std::chrono::duration_cast<std::chrono::nanoseconds>(access_time)
          .count());
  if (ipc_client.Connected()) {
    VINEYARD_CHECK_OK(
        ipc_client.CreateMetaData(vineyardFile->meta_, vineyardFile->id_));
    VINEYARD_CHECK_OK(ipc_client.Persist(vineyardFile->id_));
    Status status = ipc_client.PutName(vineyardFile->id_, path_);
  } else {
    VINEYARD_CHECK_OK(
        rpc_client.CreateMetaData(vineyardFile->meta_, vineyardFile->id_));
    VINEYARD_CHECK_OK(rpc_client.Persist(vineyardFile->id_));
    Status status = rpc_client.PutName(vineyardFile->id_, path_);
  }

  return vineyardFile;
}

std::vector<std::shared_ptr<Object>> VineyardFileBuilder::BatchedSealAndPersist(
    RPCClient& rpc_client, Client& ipc_client,
    std::vector<std::shared_ptr<VineyardFileBuilder>>& builders) {
  std::vector<std::shared_ptr<Object>> vineyard_file_objects;
  std::vector<ObjectMeta> blob_metas;
  if (ipc_client.Connected()) {
    for (auto builder : builders) {
      std::shared_ptr<Object> object;
      VINEYARD_DISCARD(
          builder->writer_->Shrink(ipc_client, builder->writer_->size()));
      VINEYARD_CHECK_OK(builder->writer_->Seal(ipc_client, object));
      blob_metas.push_back(object->meta());
    }
  } else {
    std::vector<std::shared_ptr<RemoteBlobWriter>> remote_writers;
    for (const auto& builder : builders) {
      VINEYARD_CHECK_OK(builder->Build(rpc_client, ipc_client));
      remote_writers.push_back(builder->remote_writer_);
    }
    VINEYARD_CHECK_OK(rpc_client.CreateRemoteBlobs(remote_writers, blob_metas));
  }

  for (size_t i = 0; i < blob_metas.size(); i++) {
    std::shared_ptr<VineyardFile> vineyard_file =
        std::make_shared<VineyardFile>();
    vineyard_file->meta_.AddMember("buffer", blob_metas[i]);
    vineyard_file->meta_.AddKeyValue("path", builders[i]->path_);
    vineyard_file->meta_.AddKeyValue("size", builders[i]->Size());
    vineyard_file->meta_.SetTypeName(type_name<VineyardFile>());

    auto access_time = std::chrono::system_clock::now().time_since_epoch();
    vineyard_file->meta_.AddKeyValue(
        "access_time",
        std::chrono::duration_cast<std::chrono::nanoseconds>(access_time)
            .count());
    if (ipc_client.Connected()) {
      VINEYARD_CHECK_OK(
          ipc_client.CreateMetaData(vineyard_file->meta_, vineyard_file->id_));
      VINEYARD_CHECK_OK(ipc_client.Persist(vineyard_file->id_));
      Status status =
          ipc_client.PutName(vineyard_file->id_, builders[i]->path_);
    } else {
      VINEYARD_CHECK_OK(
          rpc_client.CreateMetaData(vineyard_file->meta_, vineyard_file->id_));
      VINEYARD_CHECK_OK(rpc_client.Persist(vineyard_file->id_));
      Status status =
          rpc_client.PutName(vineyard_file->id_, builders[i]->path_);
    }
  }

  return vineyard_file_objects;
}

Status VineyardFileBuilder::Write(const void* buffer, size_t size,
                                  size_t offset) {
  if (writer_ == nullptr && remote_writer_ == nullptr) {
    return Status::Invalid("VineyardFileBuilder has not been initialized");
  }
  if (writer_ != nullptr) {
    if (offset + size > writer_->size()) {
      return Status::Invalid("Write out of range");
    }
    memcpy(writer_->data() + offset, buffer, size);
  } else {
    if (offset + size > remote_writer_->size()) {
      return Status::Invalid("Write out of range");
    }
    memcpy(remote_writer_->data() + offset, buffer, size);
  }
  return Status::OK();
}

}  // namespace vineyard
