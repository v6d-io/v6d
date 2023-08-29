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

#include "server/util/spill_file.h"

#include <limits>
#include <memory>
#include <string>

#include "common/memory/payload.h"
#include "common/util/status.h"
#include "common/util/uuid.h"
#include "server/memory/memory.h"
#include "server/util/file_io_adaptor.h"

namespace vineyard {
namespace io {

Status SpillFileWriter::Init(const ObjectID object_id) {
  if (io_adaptor_) {
    return Status::Invalid(
        "Warning: may not flushed before io_adaptor_ is overwritten");
  }
  io_adaptor_ =
      std::make_unique<FileIOAdaptor>(spill_path_ + std::to_string(object_id));
  return Status::OK();
}

Status SpillFileWriter::Write(const std::shared_ptr<Payload>& payload) {
  RETURN_ON_ERROR(Init(payload->object_id));
  if (io_adaptor_ == nullptr) {
    return Status::IOError("Can't open io_adaptor");
  }
  RETURN_ON_ERROR(io_adaptor_->Open("w"));
  RETURN_ON_ERROR(
      io_adaptor_->Write(reinterpret_cast<char*>(&(payload->object_id)),
                         sizeof(payload->object_id)));
  RETURN_ON_ERROR(
      io_adaptor_->Write(reinterpret_cast<char*>(&(payload->data_size)),
                         sizeof(payload->data_size)));
  return io_adaptor_->Write(reinterpret_cast<const char*>(payload->pointer),
                            payload->data_size);
}

Status SpillFileWriter::Sync() {
  RETURN_ON_ERROR(io_adaptor_->Flush());
  io_adaptor_ = nullptr;
  return Status::OK();
}

Status SpillFileReader::Init(const ObjectID object_id) {
  if (io_adaptor_) {
    return Status::Invalid(
        "Warning: may not flushed before io_adaptor_ is overwritten");
  }
  io_adaptor_ =
      std::make_unique<FileIOAdaptor>(spill_path_ + std::to_string(object_id));
  return Status::OK();
}

Status SpillFileReader::Read(const std::shared_ptr<Payload>& payload,
                             const std::shared_ptr<BulkStore>& bulk_store) {
  RETURN_ON_ERROR(Init(payload->object_id));
  // reload 1. object_id 2. data_size 3. content
  if (io_adaptor_ == nullptr) {
    return Status::IOError("Can't open io_adaptor");
  }
  RETURN_ON_ERROR(io_adaptor_->Open());
  {
    ObjectID object_id = InvalidObjectID();
    RETURN_ON_ERROR(io_adaptor_->Read(&object_id, sizeof(object_id)));
    if (payload->object_id != object_id) {
      return Status::IOError(
          "Incorrect 'object_id': opening wrong file: " + spill_path_ +
          ObjectIDToString(payload->object_id));
    }
  }
  {
    int64_t data_size = std::numeric_limits<int64_t>::min();
    RETURN_ON_ERROR(io_adaptor_->Read(&data_size, sizeof(data_size)));
    if (payload->data_size != data_size) {
      return Status::IOError(
          "Incorrect 'data_size': opening wrong file: " + spill_path_ +
          ObjectIDToString(payload->object_id));
    }
  }
  payload->pointer = bulk_store->AllocateMemoryWithSpill(
      payload->data_size, &(payload->store_fd), &(payload->map_size),
      &(payload->data_offset));
  if (payload->pointer == nullptr) {
    return Status::NotEnoughMemory("Failed to allocate memory of size " +
                                   std::to_string(payload->data_size) +
                                   " while reload spilling file");
  }
  RETURN_ON_ERROR(io_adaptor_->Read(payload->pointer, payload->data_size));
  RETURN_ON_ERROR(Delete_(payload->object_id));
  io_adaptor_ = nullptr;
  return Status::OK();
}

Status SpillFileReader::Delete_(const ObjectID id) {
  if (!io_adaptor_) {
    return Status::Invalid("I/O adaptor is not initialized");
  }
  RETURN_ON_ERROR(io_adaptor_->RemoveFile(spill_path_ + std::to_string(id)));
  return Status::OK();
}

}  // namespace io
}  // namespace vineyard
