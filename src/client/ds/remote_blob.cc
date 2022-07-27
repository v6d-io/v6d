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

#include "client/ds/remote_blob.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "client/rpc_client.h"
#include "common/memory/payload.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

ObjectID RemoteBlob::id() const { return id_; }

ObjectID RemoteBlob::instance_id() const { return instance_id_; }

size_t RemoteBlob::size() const { return allocated_size(); }

size_t RemoteBlob::allocated_size() const { return size_; }

const char* RemoteBlob::data() const {
  if (size_ == 0) {
    return nullptr;
  }
  if (size_ > 0 && (buffer_ == nullptr || buffer_->size() == 0)) {
    throw std::invalid_argument(
        "The object might be a (partially) remote object and the payload data "
        "is not locally available: " +
        ObjectIDToString(id_));
  }
  return reinterpret_cast<const char*>(buffer_->data());
}

const std::shared_ptr<arrow::Buffer>& RemoteBlob::Buffer() const {
  if (size_ > 0 && (buffer_ == nullptr || buffer_->size() == 0)) {
    throw std::invalid_argument(
        "The object might be a (partially) remote object and the payload data "
        "is not locally available: " +
        ObjectIDToString(id_));
  }
  return buffer_;
}

void RemoteBlob::Dump() const {
#ifndef NDEBUG
  std::stringstream ss;
  ss << "size = " << size_ << ", buffer = ";
  {
    std::ios::fmtflags os_flags(std::cout.flags());
    auto ptr = reinterpret_cast<const uint8_t*>(this->data());
    for (size_t idx = 0; idx < size_; ++idx) {
      ss << std::setfill('0') << std::setw(2) << "\\x" << std::hex
         << static_cast<const uint32_t>(ptr[idx]);
    }
    std::cout.flags(os_flags);
  }
  std::clog << "[debug] buffer is " << ss.str() << std::endl;
#endif
}

RemoteBlob::RemoteBlob(const ObjectID id, const InstanceID instance_id,
                       const size_t size)
    : id_(id), instance_id_(instance_id), size_(size) {
  if (size > 0) {
    auto r = arrow::AllocateBuffer(size_, arrow::default_memory_pool());
    VINEYARD_ASSERT(r.ok(), "Failed to create an arrow buffer");
    this->buffer_ = r.MoveValueUnsafe();
  }
}

char* RemoteBlob::mutable_data() const {
  if (size_ == 0) {
    return nullptr;
  }
  if (size_ > 0 && (buffer_ == nullptr || buffer_->size() == 0)) {
    throw std::invalid_argument(
        "The object might be a (partially) remote object and the payload data "
        "is not locally available: " +
        ObjectIDToString(id_));
  }
  return reinterpret_cast<char*>(buffer_->mutable_data());
}

RemoteBlobWriter::RemoteBlobWriter(const size_t size) {
  if (size > 0) {
    auto r = arrow::AllocateBuffer(size, arrow::default_memory_pool());
    VINEYARD_ASSERT(r.ok(), "Failed to create an arrow buffer");
    this->buffer_ =
        std::make_shared<arrow::MutableBuffer>(r.MoveValueUnsafe(), 0, size);
  }
}
RemoteBlobWriter::RemoteBlobWriter(
    std::shared_ptr<arrow::MutableBuffer> const& buffer)
    : buffer_(buffer) {}

RemoteBlobWriter::~RemoteBlobWriter() {}

size_t RemoteBlobWriter::size() const { return buffer_ ? buffer_->size() : 0; }

char* RemoteBlobWriter::data() {
  return reinterpret_cast<char*>(buffer_->mutable_data());
}
const char* RemoteBlobWriter::data() const {
  return reinterpret_cast<const char*>(buffer_->data());
}

const std::shared_ptr<arrow::MutableBuffer>& RemoteBlobWriter::Buffer() const {
  return buffer_;
}

Status RemoteBlobWriter::Abort() { return Status::OK(); }

void RemoteBlobWriter::Dump() const {
#ifndef NDEBUG
  std::stringstream ss;
  ss << "size = " << size() << ", buffer = ";
  {
    std::ios::fmtflags os_flags(std::cout.flags());
    auto ptr = reinterpret_cast<const uint8_t*>(this->data());
    for (size_t idx = 0; idx < size(); ++idx) {
      ss << std::setfill('0') << std::setw(2) << "\\x" << std::hex
         << static_cast<const uint32_t>(ptr[idx]);
    }
    std::cout.flags(os_flags);
  }
  std::clog << "[debug] buffer is " << ss.str() << std::endl;
#endif
}

}  // namespace vineyard
