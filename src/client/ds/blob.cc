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

#include "client/ds/blob.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

#if defined(ENABLE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "client/client.h"
#include "common/memory/memcpy.h"
#include "common/memory/payload.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

CUDABufferMirror::CUDABufferMirror(Buffer& buffer, const bool initializing) {
#if defined(ENABLE_CUDA)
  if (!buffer.is_cpu() && (buffer.size() > 0)) {
    VINEYARD_ASSERT(buffer.is_mutable() || initializing,
                    "For immutable buffer, initializing must be true.");
    size_ = buffer.size();
    if (buffer.is_mutable()) {
      // for immutable buffer, no need to copy it back.
      cuda_pointer_ = buffer.mutable_data();
    }
    VINEYARD_ASSERT(cudaHostAlloc(&(host_pointer_), buffer.size(),
                                  cudaHostAllocDefault) == 0,
                    "Failed to allocate unified memory");
    if (initializing) {
      VINEYARD_ASSERT(cudaMemcpy(host_pointer_, buffer.data(), buffer.size(),
                                 cudaMemcpyDeviceToHost) == 0,
                      "Failed to copy memory from device to host");
    }
  } else {
    size_ = 0 /* no allocation */;
    cuda_pointer_ = nullptr;
    host_pointer_ = buffer.mutable_data();
  }
#else
  size_ = 0;
  cuda_pointer_ = nullptr;
  host_pointer_ = buffer.mutable_data();
#endif
}

CUDABufferMirror::~CUDABufferMirror() {
#if defined(ENABLE_CUDA)
  if (cuda_pointer_ != nullptr && host_pointer_ != nullptr) {
    VINEYARD_ASSERT(cudaMemcpy(cuda_pointer_, host_pointer_, size_,
                               cudaMemcpyHostToDevice) == 0,
                    "Failed to copy memory from host to device");
  }
  if (host_pointer_ != nullptr && size_ != 0) {
    cudaFreeHost(host_pointer_);
  }
#endif
}

size_t Blob::size() const { return allocated_size(); }

size_t Blob::allocated_size() const { return size_; }

const char* Blob::data() const {
  if (size_ == 0) {
    return nullptr;
  }
  if (size_ > 0 && (buffer_ == nullptr || buffer_->size() == 0)) {
    throw std::invalid_argument(
        "Blob::data(): the object might be a (partially) remote object and the "
        "payload data is not locally available: " +
        ObjectIDToString(id_));
  }
  return reinterpret_cast<const char*>(buffer_->data());
}

const std::shared_ptr<Buffer>& Blob::Buffer() const {
  if (size_ > 0 && (buffer_ == nullptr || buffer_->size() == 0)) {
    throw std::invalid_argument(
        "Blob::Buffer(): the object might be a (partially) remote object and "
        "the payload data is not locally available: " +
        ObjectIDToString(id_));
  }
  return buffer_;
}

const std::shared_ptr<Buffer> Blob::BufferOrEmpty() const {
  auto buffer = this->Buffer();
  if (size_ == 0 && buffer == nullptr) {
    buffer = std::make_shared<vineyard::Buffer>(nullptr, 0);
  }
  return buffer;
}

void Blob::Construct(ObjectMeta const& meta) {
  std::string __type_name = type_name<Blob>();
  VINEYARD_ASSERT(meta.GetTypeName() == __type_name,
                  "Expect typename '" + __type_name + "', but got '" +
                      meta.GetTypeName() + "'");
  this->meta_ = meta;
  this->id_ = meta.GetId();
  if (this->buffer_ != nullptr) {
    return;
  }
  if (this->id_ == EmptyBlobID()) {
    this->size_ = 0;
    return;
  }
  if (!meta.IsLocal()) {
    return;
  }
  if (meta.GetBuffer(meta.GetId(), this->buffer_).ok()) {
    if (this->buffer_ == nullptr) {
      throw std::runtime_error(
          "Blob::Construct(): Invalid internal state: local blob found but it "
          "is nullptr: " +
          ObjectIDToString(meta.GetId()));
    }
    this->size_ = this->buffer_->size();
  } else {
    throw std::runtime_error(
        "Blob::Construct(): Invalid internal state: failed to construct local "
        "blob since payload is missing: " +
        ObjectIDToString(meta.GetId()));
  }
}

void Blob::Dump() const {
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

std::shared_ptr<Blob> Blob::MakeEmpty(Client& client) {
  std::shared_ptr<Blob> empty_blob(new Blob());
  empty_blob->id_ = EmptyBlobID();
  empty_blob->size_ = 0;
  empty_blob->meta_.SetId(EmptyBlobID());
  empty_blob->meta_.SetSignature(static_cast<Signature>(EmptyBlobID()));
  empty_blob->meta_.SetTypeName(type_name<Blob>());
  empty_blob->meta_.AddKeyValue("length", 0);
  empty_blob->meta_.SetNBytes(0);

  empty_blob->meta_.SetClient(&client);
  empty_blob->meta_.AddKeyValue("instance_id", client.instance_id());
  empty_blob->meta_.AddKeyValue("transient", true);
  return empty_blob;
}

std::shared_ptr<Blob> Blob::FromAllocator(Client& client,
                                          const ObjectID object_id,
                                          const uintptr_t pointer,
                                          const size_t size) {
  std::shared_ptr<Blob> blob = std::shared_ptr<Blob>(new Blob());
  blob->id_ = object_id;
  blob->size_ = size;
  blob->meta_.SetId(object_id);
  blob->meta_.SetSignature(static_cast<Signature>(object_id));
  blob->meta_.SetTypeName(type_name<Blob>());
  blob->meta_.AddKeyValue("length", size);
  blob->meta_.SetNBytes(size);

  blob->buffer_ = Buffer::Wrap(reinterpret_cast<const uint8_t*>(pointer), size);

  // n.b.: the later emplacement requires object id exists
  VINEYARD_CHECK_OK(blob->meta_.buffer_set_->EmplaceBuffer(object_id));
  VINEYARD_CHECK_OK(
      blob->meta_.buffer_set_->EmplaceBuffer(object_id, blob->buffer_));

  blob->meta_.SetClient(&client);
  blob->meta_.AddKeyValue("instance_id", client.instance_id());
  blob->meta_.AddKeyValue("transient", true);
  return blob;
}

std::shared_ptr<Blob> Blob::FromPointer(Client& client, const uintptr_t pointer,
                                        const size_t size) {
  ObjectID object_id = InvalidObjectID();
  if (size == 0 || reinterpret_cast<uint8_t*>(pointer) == nullptr) {
    return Blob::MakeEmpty(client);
  }
  if (client.IsSharedMemory(pointer, object_id)) {
    std::shared_ptr<Blob> blob = std::shared_ptr<Blob>(new Blob());
    blob->id_ = object_id;
    blob->size_ = size;
    blob->meta_.SetId(object_id);
    blob->meta_.SetSignature(static_cast<Signature>(object_id));
    blob->meta_.SetTypeName(type_name<Blob>());
    blob->meta_.AddKeyValue("length", size);
    blob->meta_.SetNBytes(size);

    blob->buffer_ =
        Buffer::Wrap(reinterpret_cast<const uint8_t*>(pointer), size);

    // n.b.: the later emplacement requires object id exists
    VINEYARD_CHECK_OK(blob->meta_.buffer_set_->EmplaceBuffer(object_id));
    VINEYARD_CHECK_OK(
        blob->meta_.buffer_set_->EmplaceBuffer(object_id, blob->buffer_));

    blob->meta_.SetClient(&client);
    blob->meta_.AddKeyValue("instance_id", client.instance_id());
    blob->meta_.AddKeyValue("transient", true);
    return blob;
  } else {
    std::unique_ptr<BlobWriter> writer;
    VINEYARD_CHECK_OK(client.CreateBlob(size, writer));
    memory::inline_memcpy(writer->data(), reinterpret_cast<uint8_t*>(pointer),
                          size);
    return std::dynamic_pointer_cast<Blob>(writer->Seal(client));
  }
}

const std::shared_ptr<Buffer>& Blob::BufferUnsafe() const { return buffer_; }

ObjectID BlobWriter::id() const { return object_id_; }

size_t BlobWriter::size() const { return buffer_ ? buffer_->size() : 0; }

char* BlobWriter::data() {
  return reinterpret_cast<char*>(buffer_->mutable_data());
}
const char* BlobWriter::data() const {
  return reinterpret_cast<const char*>(buffer_->data());
}

const std::shared_ptr<MutableBuffer>& BlobWriter::Buffer() const {
  return buffer_;
}

Status BlobWriter::Build(Client& client) { return Status::OK(); }

Status BlobWriter::Abort(Client& client) {
  if (this->sealed()) {
    return Status::ObjectSealed("Cannot abort a sealed buffer");
  }
  return client.DropBuffer(this->object_id_, this->payload_.store_fd);
}

Status BlobWriter::Shrink(Client& client, const size_t size) {
  if (this->sealed()) {
    return Status::ObjectSealed("Cannot shrink a sealed buffer.");
  }
  RETURN_ON_ERROR(client.ShrinkBuffer(this->object_id_, size));
  this->payload_.data_size = size;
  if (buffer_) {
    buffer_.reset(new MutableBuffer(buffer_->mutable_data(), size));
  }
  return Status::OK();
}

void BlobWriter::AddKeyValue(std::string const& key, std::string const& value) {
  this->metadata_.emplace(key, value);
}

void BlobWriter::AddKeyValue(std::string const& key, std::string&& value) {
  this->metadata_.emplace(key, std::move(value));
}

void BlobWriter::Dump() const {
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

Status BlobWriter::_Seal(Client& client, std::shared_ptr<Object>& object) {
  RETURN_ON_ASSERT(!this->sealed(), "The blob writer has been already sealed.");
  // get blob and re-map
  uint8_t *mmapped_ptr = nullptr, *dist = nullptr;
  if (payload_.data_size > 0) {
    std::lock_guard<std::recursive_mutex> __guard(client.client_mutex_);
    RETURN_ON_ERROR(client.shm_->Mmap(
        payload_.store_fd, payload_.object_id, payload_.map_size,
        payload_.data_size, payload_.data_offset,
        payload_.pointer - payload_.data_offset, false, true, &mmapped_ptr));
    dist = mmapped_ptr + payload_.data_offset;
  }
  auto buffer = Buffer::Wrap(dist, payload_.data_size);

  std::shared_ptr<Blob> blob(new Blob());
  object = blob;

  blob->id_ = object_id_;
  blob->size_ = size();
  blob->meta_.SetId(object_id_);  // blob's id is the address

  // create meta in vineyardd
  blob->meta_.SetTypeName(type_name<Blob>());
  blob->meta_.AddKeyValue("length", size());
  blob->meta_.SetNBytes(size());
  blob->meta_.AddKeyValue("instance_id", client.instance_id());
  blob->meta_.AddKeyValue("transient", true);

  blob->buffer_ = buffer;  // assign the readonly buffer.
  RETURN_ON_ERROR(blob->meta_.buffer_set_->EmplaceBuffer(object_id_));
  RETURN_ON_ERROR(blob->meta_.buffer_set_->EmplaceBuffer(object_id_, buffer));

  RETURN_ON_ERROR(client.Seal(object_id_));
  // associate extra key-value metadata
  for (auto const& kv : metadata_) {
    blob->meta_.AddKeyValue(kv.first, kv.second);
  }
  this->set_sealed(true);
  return Status::OK();
}

Status BufferSet::EmplaceBuffer(ObjectID const id) {
  auto p = buffers_.find(id);
  if (p != buffers_.end() && p->second != nullptr) {
    return Status::Invalid(
        "Invalid internal state: the buffer shouldn't has been filled, id "
        "= " +
        ObjectIDToString(id));
  }
  buffer_ids_.emplace(id);
  buffers_.emplace(id, nullptr);
  return Status::OK();
}

Status BufferSet::EmplaceBuffer(ObjectID const id,
                                std::shared_ptr<Buffer> const& buffer) {
  auto p = buffers_.find(id);
  if (p == buffers_.end()) {
    return Status::Invalid(
        "Invalid internal state: no such buffer defined, id = " +
        ObjectIDToString(id));
  } else {
    if (p->second != nullptr) {
      return Status::Invalid(
          "Invalid internal state: duplicated buffer, id = " +
          ObjectIDToString(id));
    } else {
      p->second = buffer;
      return Status::OK();
    }
  }
}

void BufferSet::Extend(BufferSet const& others) {
  for (auto const& kv : others.buffers_) {
    buffers_.emplace(kv.first, kv.second);
  }
}

void BufferSet::Extend(std::shared_ptr<BufferSet> const& others) {
  this->Extend(*others);
}

bool BufferSet::Contains(ObjectID const id) const {
  return buffers_.find(id) != buffers_.end();
}

bool BufferSet::Get(ObjectID const id, std::shared_ptr<Buffer>& buffer) const {
  auto iter = buffers_.find(id);
  if (iter == buffers_.end()) {
    return false;
  } else {
    buffer = iter->second;
    return true;
  }
}

}  // namespace vineyard
