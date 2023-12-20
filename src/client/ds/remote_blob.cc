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

#include "client/ds/remote_blob.h"

#include <cstring>
#include <iomanip>
#include <memory>
#include <string>
#include <utility>

#include "client/client_base.h"
#include "client/ds/blob.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

ObjectID RemoteBlob::id() const { return id_; }

ObjectID RemoteBlob::instance_id() const { return instance_id_; }

void RemoteBlob::Construct(ObjectMeta const& meta) {
  std::string __type_name = type_name<RemoteBlob>();
  VINEYARD_ASSERT(meta.GetTypeName() == __type_name,
                  "Expect typename '" + __type_name + "', but got '" +
                      meta.GetTypeName() + "'");
  this->meta_ = meta;
  this->id_ = meta.GetId();

  if (this->buffer_ != nullptr) {
    return;
  }
  if (this->id_ == EmptyBlobID() || meta.GetNBytes() == 0) {
    this->size_ = 0;
    return;
  }

  if (meta.GetClient()->IsRPC() &&
      meta.GetClient()->remote_instance_id() != meta.GetInstanceId()) {
    throw std::runtime_error(
        "RemoteBlob::Construct(): Invalid internal state: remote blob found "
        "but it is not located with the instance connected by rpc client");
  }

  if (meta.GetBuffer(meta.GetId(), this->buffer_).ok()) {
    if (this->buffer_ == nullptr) {
      throw std::runtime_error(
          "RemoteBlob::Construct(): Invalid internal state: remote blob found "
          "but it is nullptr: " +
          ObjectIDToString(meta.GetId()));
    }
    this->size_ = this->buffer_->size();
  } else {
    throw std::runtime_error(
        "RemoteBlob::Construct(): Invalid internal state: failed to construct "
        "remote blob since payload is missing: " +
        ObjectIDToString(meta.GetId()));
  }
}

size_t RemoteBlob::size() const { return allocated_size(); }

size_t RemoteBlob::allocated_size() const { return size_; }

const char* RemoteBlob::data() const {
  if (size_ == 0) {
    return nullptr;
  }
  if (size_ > 0 && (buffer_ == nullptr || buffer_->size() == 0)) {
    throw std::invalid_argument(
        "RemoteBlob::data(): the object might be a (partially) remote object "
        "and the payload data is not locally available: " +
        ObjectIDToString(id_));
  }
  return reinterpret_cast<const char*>(buffer_->data());
}

const std::shared_ptr<Buffer>& RemoteBlob::Buffer() const {
  if (size_ > 0 && (buffer_ == nullptr || buffer_->size() == 0)) {
    throw std::invalid_argument(
        "RemoteBlob::Buffer(): the object might be a (partially) remote object "
        "and the payload data is not locally available: " +
        ObjectIDToString(id_));
  }
  return buffer_;
}

const std::shared_ptr<Buffer> RemoteBlob::BufferOrEmpty() const {
  auto buffer = this->Buffer();
  if (size_ == 0 && buffer == nullptr) {
    buffer = std::make_shared<vineyard::Buffer>(nullptr, 0);
  }
  return buffer;
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
    auto buffer = MallocBuffer::AllocateBuffer(size);
    VINEYARD_ASSERT(
        buffer != nullptr,
        "Failed to malloc the internal buffer of size " + std::to_string(size));
    this->buffer_ = std::dynamic_pointer_cast<MutableBuffer>(std::move(buffer));
  }
}

char* RemoteBlob::mutable_data() const {
  if (size_ == 0) {
    return nullptr;
  }
  if (size_ > 0 && (buffer_ == nullptr || buffer_->size() == 0)) {
    throw std::invalid_argument(
        "RemoteBlob::mutable_data(): The object might be a (partially) remote "
        "object and the payload data is not locally available: " +
        ObjectIDToString(id_));
  }
  return reinterpret_cast<char*>(buffer_->mutable_data());
}

RemoteBlobWriter::RemoteBlobWriter(const size_t size) {
  if (size > 0) {
    auto buffer = MallocBuffer::AllocateBuffer(size);
    VINEYARD_ASSERT(
        buffer != nullptr,
        "Failed to malloc the internal buffer of size " + std::to_string(size));
    this->buffer_ = std::move(buffer);
  }
}

RemoteBlobWriter::RemoteBlobWriter(std::shared_ptr<MutableBuffer> const& buffer)
    : buffer_(buffer) {}

RemoteBlobWriter::~RemoteBlobWriter() {}

std::shared_ptr<RemoteBlobWriter> RemoteBlobWriter::Make(const size_t size) {
  return std::shared_ptr<RemoteBlobWriter>(new RemoteBlobWriter(size));
}

std::shared_ptr<RemoteBlobWriter> RemoteBlobWriter::Wrap(const uint8_t* data,
                                                         const size_t size) {
  auto buffer = std::dynamic_pointer_cast<MutableBuffer>(
      MutableBuffer::Wrap(const_cast<uint8_t*>(data), size));
  return std::shared_ptr<RemoteBlobWriter>(new RemoteBlobWriter(buffer));
}

size_t RemoteBlobWriter::size() const { return buffer_ ? buffer_->size() : 0; }

char* RemoteBlobWriter::data() {
  return reinterpret_cast<char*>(buffer_->mutable_data());
}
const char* RemoteBlobWriter::data() const {
  return reinterpret_cast<const char*>(buffer_->data());
}

const std::shared_ptr<MutableBuffer>& RemoteBlobWriter::Buffer() const {
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
