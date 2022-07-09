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

#include "server/util/file_io.h"

#include <memory>
#include <string>
#include "common/util/status.h"
#include "common/util/uuid.h"
#include "io/io/local_io_adaptor.h"

namespace util {
using vineyard::Status;
void PutFixed32(std::string* dst, uint32_t value) {
  char buf[sizeof(value)];
  EncodeFixed32(buf, value);
  dst->append(buf, sizeof(buf));
}

void PutFixed64(std::string* dst, uint64_t value) {
  char buf[sizeof(value)];
  EncodeFixed64(buf, value);
  dst->append(buf, sizeof(buf));
}

Status SpillWriteFile::Open() {
  io_adaptor_ = vineyard::IOFactory::CreateIOAdaptor(spill_path_);
  if (io_adaptor_ == nullptr) {
    return Status::IOError("Failed to open" + spill_path_);
  }
  return io_adaptor_->Open("w");
}

Status SpillWriteFile::Write(const char* data, const size_t& size,
                             const vineyard::ObjectID& object_id) {
  std::string to_spill;

  PutFixed64(&to_spill, object_id);
  PutFixed64(&to_spill, static_cast<uint64_t>(size));
  to_spill.append(data, size);
  return io_adaptor_->Write(reinterpret_cast<const void*>(to_spill.data()),
                            to_spill.size());
}

Status SpillWriteFile::Sync() { return io_adaptor_->Flush(); }

Status SpillReadFile::Open() {
  io_adaptor_ = vineyard::IOFactory::CreateIOAdaptor(spill_path_);
  if (io_adaptor_ == nullptr) {
    return Status::IOError("Failed to open" + spill_path_);
  }
  io_adaptor_->SetPartialRead(0, 1);
  return io_adaptor_->Open();
}

Status SpillReadFile::Read(size_t n, char* result) {
  return io_adaptor_->Read(result, n);
}
}  // namespace util