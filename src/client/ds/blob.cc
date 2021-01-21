/** Copyright 2020 Alibaba Group Holding Limited.

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

#include <limits>

#include "client/client.h"

namespace vineyard {

Blob::Blob() {
  this->id_ = InvalidObjectID();
  this->size_ = std::numeric_limits<size_t>::max();
  this->buffer_ = nullptr;
}

Blob::Blob(const ObjectID id, const size_t size) {
  this->id_ = id;
  this->size_ = size;
  this->buffer_ = nullptr;
}

Blob::Blob(const ObjectID id, const size_t size,
           std::shared_ptr<arrow::Buffer> const& buffer) {
  this->id_ = id;
  this->size_ = size;
  this->buffer_ = buffer;
}

size_t Blob::size() const { return size_; }

const char* Blob::data() const {
  if (size_ > 0 && buffer_ == nullptr) {
    throw std::invalid_argument(
        "The object might be a (partially) remote object and the payload data "
        "is not locally available");
  }
  return reinterpret_cast<const char*>(buffer_->data());
}

const std::shared_ptr<arrow::Buffer>& Blob::Buffer() const {
  if (size_ > 0 && buffer_ == nullptr) {
    throw std::invalid_argument(
        "The object might be a (partially) remote object and the payload data "
        "is not locally available");
  }
  return buffer_;
}

void Blob::Construct(ObjectMeta const& meta) {
  std::string __type_name = type_name<Blob>();
  CHECK(meta.GetTypeName() == __type_name);
  this->meta_ = meta;
  this->id_ = meta.GetId();
  meta.GetKeyValue("length", this->size_);
  if (auto client = dynamic_cast<Client*>(meta.GetClient())) {
    Payload object;
    if (this->size_ == 0) {
      // dummy blob
      buffer_ = nullptr;  // indicates empty blob
    } else {
      auto status = client->GetBuffer(meta.GetId(), object);
      if (status.ok()) {
        uint8_t* mmapped_ptr = nullptr;
        if (object.data_size > 0) {
          VINEYARD_CHECK_OK(client->mmapToClient(
              object.store_fd, object.map_size, true, &mmapped_ptr));
        }
        buffer_ = arrow::Buffer::Wrap(mmapped_ptr + object.data_offset,
                                      object.data_size);
      } else {
        throw std::runtime_error("Failed to construct blob: " +
                                 VYObjectIDToString(meta.GetId()));
      }
    }
  }
}

std::shared_ptr<Blob> Blob::MakeEmpty(Client& client) {
  std::shared_ptr<Blob> empty_blob(new Blob(EmptyBlobID(), 0, nullptr));
  empty_blob->meta_.SetId(EmptyBlobID());
  empty_blob->meta_.SetTypeName(type_name<Blob>());
  empty_blob->meta_.AddKeyValue("length", 0);
  empty_blob->meta_.SetNBytes(0);

  empty_blob->meta_.AddKeyValue("instance_id", client.instance_id());
  empty_blob->meta_.AddKeyValue("transient", true);

  // NB: no need to create metadata in vineyardd at once
  return empty_blob;
}

const std::shared_ptr<arrow::Buffer>& Blob::BufferUnsafe() const {
  return buffer_;
}

size_t BlobWriter::size() const { return buffer_ ? buffer_->size() : 0; }

char* BlobWriter::data() {
  return reinterpret_cast<char*>(buffer_->mutable_data());
}
const char* BlobWriter::data() const {
  return reinterpret_cast<const char*>(buffer_->data());
}

const std::shared_ptr<arrow::MutableBuffer>& BlobWriter::Buffer() const {
  return buffer_;
}

Status BlobWriter::Build(Client& client) { return Status::OK(); }

void BlobWriter::AddKeyValue(std::string const& key, std::string const& value) {
  this->metadata_.emplace(key, value);
}

void BlobWriter::AddKeyValue(std::string const& key, std::string&& value) {
  this->metadata_.emplace(key, std::move(value));
}

std::shared_ptr<Object> BlobWriter::_Seal(Client& client) {
  // get blob and re-map
  Payload object;
  VINEYARD_CHECK_OK(client.GetBuffer(object_id_, object));
  uint8_t* mmapped_ptr = nullptr;
  if (object.data_size > 0) {
    VINEYARD_CHECK_OK(client.mmapToClient(object.store_fd, object.map_size,
                                          false, &mmapped_ptr));
  }
  auto ro_buffer =
      arrow::Buffer::Wrap(mmapped_ptr + object.data_offset, object.data_size);

  std::shared_ptr<Blob> blob(new Blob(object_id_, size(), ro_buffer));

  blob->meta_.SetId(object_id_);  // blob's id is the address
  // create meta in vineyardd
  blob->meta_.SetTypeName(type_name<Blob>());
  blob->meta_.AddKeyValue("length", size());
  blob->meta_.SetNBytes(size());

  // assoicate extra key-value metadata
  for (auto const& kv : metadata_) {
    blob->meta_.AddKeyValue(kv.first, kv.second);
  }

  VINEYARD_CHECK_OK(client.CreateMetaData(blob->meta_, blob->id_));
  return blob;
}

void BlobSet::EmplaceId(ObjectID const id, size_t const size, bool local) {
  if (local) {
    ids_.emplace(id);
  }
  blobs_.emplace(id, Blob(id, size));
}

void BlobSet::EmplaceBlob(ObjectID const id,
                          std::shared_ptr<arrow::Buffer> const& buffer) {
  ids_.emplace(id);
  auto p = blobs_.find(id);
  if (p == blobs_.end()) {
    blobs_.emplace(id, Blob(id, buffer->size(), buffer));
  } else {
    p->second.buffer_ = buffer;
  }
}

void BlobSet::Extend(BlobSet const& others) {
  for (auto const& id : others.ids_) {
    ids_.emplace(id);
  }
  for (auto const& kv : others.blobs_) {
    blobs_.emplace(kv.first, kv.second);
  }
}

void BlobSet::Extend(std::shared_ptr<BlobSet> const& others) {
  this->Extend(*others);
}

bool BlobSet::Contains(ObjectID const id) const {
  return ids_.find(id) != ids_.end();
}

}  // namespace vineyard
