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

#include "server/util/spill_file.h"
#include <glog/logging.h>

#include <memory>
#include <string>
#include "common/memory/payload.h"
#include "common/util/base64.h"
#include "common/util/status.h"
#include "common/util/uuid.h"
#include "io/io/local_io_adaptor.h"
#include "server/memory/memory.h"
#include "server/util/file_io_adaptor.h"

namespace util {
using vineyard::BulkStore;
using vineyard::Payload;
using vineyard::Status;

void PutFixed64(std::string* dst, uint64_t value) {
  char buf[sizeof(value)];
  EncodeFixed64(buf, value);
  dst->append(buf, sizeof(buf));
}

Status SpillWriteFile::Init(uint64_t object_id) {
  if (io_adaptor_) {
    return Status::Invalid(
        "Warning: may not flushed before io_adaptor_ is overwriten");
  }
  io_adaptor_ =
      std::make_unique<FileIOAdaptor>(spill_path_ + std::to_string(object_id));
  return Status::OK();
}

Status SpillWriteFile::Write(const std::shared_ptr<Payload>& payload) {
  Init(payload->object_id);
  if (io_adaptor_ == nullptr) {
    return Status::IOError("Can't open io_adaptor");
  }
  RETURN_ON_ERROR(io_adaptor_->Open("w"));
  std::string to_spill;
  PutFixed64(&to_spill, payload->object_id);
  PutFixed64(&to_spill, static_cast<uint64_t>(payload->data_size));
  to_spill.append(reinterpret_cast<const char*>(payload->pointer),
                  payload->data_size);
  return io_adaptor_->Write(to_spill.data(), to_spill.size());
}

Status SpillWriteFile::Sync() {
  RETURN_ON_ERROR(io_adaptor_->Flush());
  io_adaptor_ = nullptr;
  return Status::OK();
}

Status SpillReadFile::Init(uint64_t object_id) {
  if (io_adaptor_) {
    return Status::Invalid(
        "Warning: may not flushed before io_adaptor_ is overwriten");
  }
  io_adaptor_ =
      std::make_unique<FileIOAdaptor>(spill_path_ + std::to_string(object_id));
  return Status::OK();
}

Status SpillReadFile::Read(std::shared_ptr<Payload>& payload,
                           std::shared_ptr<BulkStore> bulk_store_ptr) {
  RETURN_ON_ERROR(Init(payload->object_id));
  // reload 1. object_id 2. data_size 3. content
  if (io_adaptor_ == nullptr) {
    return Status::IOError("Can't open io_adaptor");
  }
  RETURN_ON_ERROR(io_adaptor_->Open());
  {
    char buf[sizeof(payload->object_id)];
    RETURN_ON_ERROR(io_adaptor_->Read(buf, sizeof(payload->object_id)));
    if (payload->object_id != util::DecodeFixed64(buf)) {
      return Status::IOError("Open wrong file: " + spill_path_ +
                             std::to_string(payload->object_id));
    }
  }
  {
    char buf[sizeof(payload->data_size)];
    RETURN_ON_ERROR(io_adaptor_->Read(buf, sizeof(payload->data_size)));
    if (static_cast<uint64_t>(payload->data_size) != util::DecodeFixed64(buf)) {
      return Status::IOError("Open wrong file: " + spill_path_ +
                             std::to_string(payload->object_id));
    }
  }
  payload->pointer = bulk_store_ptr->AllocateMemoryWithSpill(
      payload->data_size, &payload->store_fd, &payload->map_size,
      &payload->data_offset);
  if (payload->pointer == nullptr) {
    return Status::NotEnoughMemory("Failed to allocate memory of size " +
                                   std::to_string(payload->data_size) +
                                   " while reload spilling file");
  }
  RETURN_ON_ERROR(io_adaptor_->Read(payload->pointer, payload->data_size));
  std::cout << "Gonna Delete File" << std::endl;
  RETURN_ON_ERROR(Delete_(payload->object_id));
  std::cout << "Finish Delete" << std::endl;
  io_adaptor_ = nullptr;
  return Status::OK();
}

Status SpillReadFile::Delete_(const vineyard::ObjectID& id) {
  if (!io_adaptor_) {
    return Status::Invalid("io_adaptor_ is not initialized");
  }
  // RETURN_ON_ERROR(io_adaptor_->RemoveFile(spill_path_ + std::to_string(id)));
  return Status::OK();
}

}  // namespace util
