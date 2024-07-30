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

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common/util/logging.h"
#include "gulrak/filesystem.hpp"
#include "llm-cache/ds/vineyard_file.h"
#include "llm-cache/storage/vineyard_file_storage.h"
#include "llm-cache/thread_group.h"

namespace vineyard {
std::shared_ptr<FileDescriptor> VineyardFileStorage::CreateFileDescriptor() {
  return std::make_shared<VineyardFileDescriptor>();
}

Status VineyardFileStorage::Open(std::string path,
                                 std::shared_ptr<FileDescriptor>& fd,
                                 FileOperationType fileOperationType) {
  std::shared_ptr<VineyardFileDescriptor> lfd =
      std::static_pointer_cast<VineyardFileDescriptor>(fd);
  lfd->path = path;
  lfd->cur_pos = 0;

  if (fileOperationType & FileOperationType::READ) {
    RETURN_ON_ERROR(VineyardFile::Make(lfd->vineyard_file, rpc_client_, path));
  } else {
    RETURN_ON_ERROR(VineyardFileBuilder::Make(lfd->builder, rpc_client_, path,
                                              max_file_size_));
  }
  lfd->opt_type = fileOperationType;
  return Status::OK();
}

Status VineyardFileStorage::Seek(std::shared_ptr<FileDescriptor>& fd,
                                 size_t offset) {
  std::shared_ptr<VineyardFileDescriptor> lfd =
      std::static_pointer_cast<VineyardFileDescriptor>(fd);
  size_t size;
  RETURN_ON_ERROR(GetFileSize(fd, size));
  if (offset > size) {
    return Status::Invalid("Seek out of range");
  }
  lfd->cur_pos = offset;
  return Status::OK();
}

Status VineyardFileStorage::Read(std::shared_ptr<FileDescriptor>& fd,
                                 void* data, size_t size) {
  std::shared_ptr<VineyardFileDescriptor> lfd =
      std::static_pointer_cast<VineyardFileDescriptor>(fd);
  // TBD
  if (lfd->opt_type == FileOperationType::READ) {
    RETURN_ON_ERROR(lfd->vineyard_file->Read(data, size, lfd->cur_pos));
    lfd->cur_pos += size;
  } else {
    return Status::Invalid("File is not opened for read");
  }

  return Status::OK();
}

Status VineyardFileStorage::Write(std::shared_ptr<FileDescriptor>& fd,
                                  const void* data, size_t size) {
  std::shared_ptr<VineyardFileDescriptor> lfd =
      std::static_pointer_cast<VineyardFileDescriptor>(fd);
  // TBD
  if (lfd->opt_type == FileOperationType::WRITE) {
    RETURN_ON_ERROR(lfd->builder->Write(data, size, lfd->cur_pos));
    lfd->cur_pos += size;
  } else {
    return Status::Invalid("File is not opened for write");
  }
  return Status::OK();
}

Status VineyardFileStorage::Mkdir(std::string path) { return Status::OK(); }

Status VineyardFileStorage::Flush(std::shared_ptr<FileDescriptor>& fd) {
  return Status::OK();
}

Status VineyardFileStorage::GetCurrentPos(std::shared_ptr<FileDescriptor>& fd,
                                          size_t& pos) {
  std::shared_ptr<VineyardFileDescriptor> lfd =
      std::static_pointer_cast<VineyardFileDescriptor>(fd);
  pos = lfd->cur_pos;
  return Status::OK();
}

Status VineyardFileStorage::Close(std::shared_ptr<FileDescriptor>& fd) {
  std::shared_ptr<VineyardFileDescriptor> lfd =
      std::static_pointer_cast<VineyardFileDescriptor>(fd);
  if (lfd->opt_type == FileOperationType::WRITE) {
    lfd->builder->SealAndPersist(rpc_client_);
  }
  return Status::OK();
}

Status VineyardFileStorage::GetFileSize(std::shared_ptr<FileDescriptor>& fd,
                                        size_t& size) {
  std::shared_ptr<VineyardFileDescriptor> lfd =
      std::static_pointer_cast<VineyardFileDescriptor>(fd);
  if (lfd->opt_type == FileOperationType::READ) {
    size = lfd->vineyard_file->Size();
  } else {
    size = lfd->builder->Size();
  }
  return Status::OK();
}

bool VineyardFileStorage::IsFileExist(const std::string& path) {
  ObjectID file_id;
  std::string origin_path = std::regex_replace(path, std::regex("/+"), "\\/");
  if (rpc_client_.GetName(origin_path, file_id, false).ok()) {
    return true;
  }
  return false;
}

Status VineyardFileStorage::Delete(std::string path) {
  std::string origin_path = std::regex_replace(path, std::regex("/+"), "\\/");
  std::string lock_path;
  bool result = false;
  VineyardFileLock lock(rpc_client_, origin_path);
  RETURN_ON_ERROR(lock.TryLock());
  ObjectID file_id;
  Status status = Status::OK();
  if (rpc_client_.GetName(origin_path, file_id, false).ok()) {
    status = rpc_client_.DelData(std::vector<ObjectID>{file_id}, true, true);
    status = rpc_client_.DropName(origin_path);
  }
  do {
    rpc_client_.TryReleaseLock(lock_path, result);
  } while (!result);
  return status;
}

std::string VineyardFileStorage::GetTmpFileDir(std::string suffix) {
  return this->rootPath;
}

Status VineyardFileStorage::MoveFileAtomic(std::string src, std::string dst) {
  if (src == dst) {
    return Status::OK();
  }
  return Status::Invalid("Vineyard file storage does not support atomic move");
}

Status VineyardFileStorage::GetFileAccessTime(
    const std::string& path,
    std::chrono::duration<int64_t, std::nano>& accessTime) {
  ObjectID file_id;
  ObjectMeta meta;
  std::string origin_path = std::regex_replace(path, std::regex("/+"), "\\/");
  RETURN_ON_ERROR(rpc_client_.GetName(origin_path, file_id, false));
  RETURN_ON_ERROR(rpc_client_.GetMetaData(file_id, meta, false));
  uint64_t time = meta.GetKeyValue<uint64_t>("access_time");
  accessTime = std::chrono::nanoseconds(time);
  return Status::OK();
}

Status VineyardFileStorage::TouchFile(const std::string& path) {
  ObjectID file_id;
  ObjectMeta meta;
  std::string lock_path;
  std::string origin_path = std::regex_replace(path, std::regex("/+"), "\\/");
  VineyardFileLock lock(rpc_client_, origin_path);
  RETURN_ON_ERROR(lock.TryLock());
  RETURN_ON_ERROR(rpc_client_.GetName(origin_path, file_id, false));
  RETURN_ON_ERROR(rpc_client_.GetMetaData(file_id, meta, false));
  meta.AddKeyValue(
      "access_time",
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count());
  ObjectID new_object_id;
  RETURN_ON_ERROR(rpc_client_.CreateMetaData(meta, new_object_id));
  RETURN_ON_ERROR(rpc_client_.Persist(new_object_id));
  RETURN_ON_ERROR(
      rpc_client_.DelData(std::vector<ObjectID>{file_id}, false, false));
  RETURN_ON_ERROR(rpc_client_.DropName(origin_path));
  RETURN_ON_ERROR(rpc_client_.PutName(new_object_id, origin_path));
  return Status::OK();
}

Status VineyardFileStorage::GetFileList(std::string dirPath,
                                        std::vector<std::string>& fileList) {
  std::string origin_path =
      std::regex_replace(dirPath, std::regex("/+"), "\\/");
  std::map<std::string, ObjectID> file_name_to_ids;
  RETURN_ON_ERROR(
      rpc_client_.ListNames(origin_path, false, UINT64_MAX, file_name_to_ids));
  fileList.resize(file_name_to_ids.size());
  size_t i = 0;
  for (auto& kv : file_name_to_ids) {
    fileList[i++] = kv.first;
  }
  return Status::OK();
}

}  // namespace vineyard
