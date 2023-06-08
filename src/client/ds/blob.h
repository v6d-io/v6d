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

#ifndef SRC_CLIENT_DS_BLOB_H_
#define SRC_CLIENT_DS_BLOB_H_

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "client/ds/i_object.h"
#include "common/memory/payload.h"
#include "common/util/uuid.h"

namespace arrow {
class Buffer;
}  // namespace arrow

namespace vineyard {

class Buffer {
 public:
  Buffer(const uint8_t* data, int64_t size)
      : is_mutable_(false),
        is_cpu_(true),
        data_(data),
        size_(size),
        capacity_(size) {}

  virtual ~Buffer() = default;

  uint8_t operator[](std::size_t i) const { return data_[i]; }

  template <typename T, typename SizeType = int64_t>
  static std::shared_ptr<Buffer> Wrap(const T* data, SizeType length) {
    return std::make_shared<Buffer>(reinterpret_cast<const uint8_t*>(data),
                                    static_cast<int64_t>(sizeof(T) * length));
  }

  const uint8_t* data() const { return likely(is_cpu_) ? data_ : nullptr; }

  uint8_t* mutable_data() {
    return likely(is_cpu_ && is_mutable_) ? const_cast<uint8_t*>(data_)
                                          : nullptr;
  }

  uintptr_t address() const { return reinterpret_cast<uintptr_t>(data_); }

  uintptr_t mutable_address() const {
    return likely(is_mutable_) ? reinterpret_cast<uintptr_t>(data_) : 0;
  }

  int64_t size() const { return size_; }

  int64_t capacity() const { return capacity_; }

  bool is_cpu() const { return is_cpu_; }

  bool is_mutable() const { return is_mutable_; }

 protected:
  bool is_mutable_;
  bool is_cpu_;
  const uint8_t* data_;
  int64_t size_;
  int64_t capacity_;

 private:
  Buffer() = delete;

  Buffer(const Buffer&) = delete;
  Buffer(Buffer&&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer& operator=(Buffer&&) = delete;
};

class MutableBuffer : public Buffer {
 public:
  MutableBuffer(uint8_t* data, const int64_t size) : Buffer(data, size) {
    is_mutable_ = true;
  }

  template <typename T, typename SizeType = int64_t>
  static std::shared_ptr<Buffer> Wrap(T* data, SizeType length) {
    return std::make_shared<MutableBuffer>(
        reinterpret_cast<uint8_t*>(data),
        static_cast<int64_t>(sizeof(T) * length));
  }

 protected:
  MutableBuffer() : Buffer(nullptr, 0) {}
};

class MallocBuffer : public MutableBuffer {
 public:
  ~MallocBuffer() {
    if (buffer_ != nullptr) {
      free(buffer_);
    }
  }

  static std::unique_ptr<MallocBuffer> AllocateBuffer(const size_t size) {
    void* buffer = malloc(size);
    if (buffer) {
      return std::unique_ptr<MallocBuffer>(new MallocBuffer(buffer, size));
    } else {
      return nullptr;
    }
  }

 private:
  MallocBuffer(void* buffer, const size_t size)
      : MutableBuffer(reinterpret_cast<uint8_t*>(buffer), size),
        buffer_(buffer) {}

  void* buffer_ = nullptr;
};

class BlobWriter;
class BufferSet;
class Client;
class PlasmaClient;
class ObjectMeta;

/**
 * @brief The unit to store data payload in vineyard.
 *
 * When the client gets a blob from vineyard, the vineyard server maps
 * a chunk of memory from its memory space to the client space in a
 * zero-copy fashion.
 */
class Blob : public Registered<Blob> {
 public:
  /**
   * @brief Get the size of the blob, i.e., the number of bytes of the data
   * payload in the blob.
   *
   * Note that the size of blob is the "allocated size" of the blob, and may
   * (usually) not be the same value of the requested size.
   *
   * @return The (allocated) size of the blob.
   */
  size_t size() const __attribute__((
      deprecated("The method `size()` no longer report accurate requested "
                 "memory size, use allocated_size() instead.")));

  /**
   * @brief Get the allocated size of the blob, i.e., the number of bytes of the
   * data payload in the blob.
   *
   * @return The allocated size of the blob.
   */
  size_t allocated_size() const;

  /**
   * @brief Get the const data pointer of the data payload in the blob.
   *
   * @return The const data pointer.
   */
  const char* data() const;

  /**
   * @brief Get the buffer of the blob.
   *
   * @return The buffer which holds the data payload
   * of the blob.
   */
  const std::shared_ptr<vineyard::Buffer>& Buffer() const;

  /**
   * @brief Get the arrow buffer of the blob.
   *
   * @return The buffer which holds the data payload
   * of the blob.
   */
  const std::shared_ptr<arrow::Buffer> ArrowBuffer() const;

  /**
   * @brief Get the buffer of the blob, ensure a valid shared_ptr been
   * returned even the blob is empty (size == 0).
   *
   * @return The buffer which holds the data payload
   * of the blob.
   */
  const std::shared_ptr<vineyard::Buffer> BufferOrEmpty() const;

  /**
   * @brief Get the arrow buffer of the blob, ensure a valid shared_ptr been
   * returned even the blob is empty (size == 0).
   *
   * @return The buffer which holds the data payload
   * of the blob.
   */
  const std::shared_ptr<arrow::Buffer> ArrowBufferOrEmpty() const;

  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(std::unique_ptr<Blob>{new Blob()});
  }

  /**
   * @brief Construct the blob locally for the given object meta.
   *
   * @param meta The given object meta.
   */
  void Construct(ObjectMeta const& meta) override;

  /**
   * @brief Dump the buffer for debugging.
   */
  void Dump() const;

  /**
   * @brief Create an empty blob in the vineyard server.
   *
   * @param client The client connected to the vineyard server.
   */
  static std::shared_ptr<Blob> MakeEmpty(Client& client);

  /**
   * @brief Create the blob from a buffer from the client-side allocator.
   *
   * @param object_id The object ID of this blob.
   * @param pointer The address of buffer in the client-side allocator.
   * @param size The estimated size of the buffer.
   */
  static std::shared_ptr<Blob> FromAllocator(Client& client,
                                             const ObjectID object_id,
                                             const uintptr_t pointer,
                                             const size_t size);

  /**
   * @brief Create the blob from a given buffer. If the buffer already lies in
   * the vineyardd, it would return immediately without copying, otherwise a
   *        blob writer will be created and the content of the buffer will be
   *        copied into.
   *
   * @param pointer The address of the buffer.
   * @param size The estimated size of the buffer.
   */
  static std::shared_ptr<Blob> FromPointer(Client& client,
                                           const uintptr_t pointer,
                                           const size_t size);

 private:
  /**
   * The default constructor is only used in BlobWriter.
   */
  Blob() {
    this->id_ = InvalidObjectID();
    this->size_ = std::numeric_limits<size_t>::max();
    this->buffer_ = nullptr;
  }

  const std::shared_ptr<vineyard::Buffer>& BufferUnsafe() const;

  size_t size_ = 0;
  std::shared_ptr<vineyard::Buffer> buffer_ = nullptr;

  friend class Client;
  friend class PlasmaClient;
  friend class RPCClient;
  friend class BlobWriter;
  friend class BufferSet;
  friend class ObjectMeta;
};

/**
 * @brief The writer to write a blob in vineyard.
 *
 * The writer is initialized in the client with a local buffer and its size,
 * and a blob in vineyard will be created when Build is invoked.
 */
class BlobWriter : public ObjectBuilder {
 public:
  /**
   * @brief Return the object id of this blob builder. Note that before sealing
   * the blob builder the object id cannot be used to get "Blob" objects.
   *
   * @return The ObjectID of the blob writer.
   */
  ObjectID id() const;

  /**
   * @brief Get the size of the blob, i.e., the number of bytes of the data
   * payload in the blob.
   *
   * @return The size of the blob.
   */
  size_t size() const;

  /**
   * @brief Get the data pointer of the data payload in the blob.
   *
   * @return The data pointer.
   */
  char* data();

  /**
   * @brief Get the const data pointer of the data payload in the blob.
   *
   * @return The const data pointer.
   */
  const char* data() const;

  /**
   * @brief Get the mutable buffer of the blob.
   *
   * @return The mutable buffer of the blob, which can be modified
   * to update the content in the blob.
   */
  const std::shared_ptr<MutableBuffer>& Buffer() const;

  /**
   * @brief Build a blob in vineyard server.
   *
   * @param client The client connected to the vineyard server.
   */
  Status Build(Client& client) override;

  /**
   * @brief Abort the blob builder.
   *
   * @param client Release the blob builder object if it is not sealed.
   */
  Status Abort(Client& client);

  /**
   * @brief Add key-value metadata for the blob.
   *
   * @param key The key of the metadata.
   * @param value The value of the metadata.
   */
  void AddKeyValue(std::string const& key, std::string const& value);

  /**
   * @brief Add key-value metadata for the blob.
   *
   * @param key The key of the metadata.
   * @param value The value of the metadata.
   */
  void AddKeyValue(std::string const& key, std::string&& value);

  /**
   * @brief Dump the buffer for debugging.
   */
  void Dump() const;

 protected:
  Status _Seal(Client& client, std::shared_ptr<Object>& object) override;

 private:
  BlobWriter(ObjectID const object_id, const Payload& payload,
             std::shared_ptr<MutableBuffer> const& buffer)
      : object_id_(object_id), payload_(payload), buffer_(buffer) {}

  BlobWriter(ObjectID const object_id, Payload&& payload,
             std::shared_ptr<MutableBuffer> const& buffer)
      : object_id_(object_id), payload_(payload), buffer_(buffer) {}

  ObjectID object_id_;
  Payload payload_;
  std::shared_ptr<MutableBuffer> buffer_;
  // Allowing blobs have extra key-value metadata
  std::unordered_map<std::string, std::string> metadata_;

  friend class Client;
  friend class PlasmaClient;
  friend class RPCClient;
};

/**
 * @brief A set of (readonly) buffers that been associated with an object and
 * its members (recursively).
 */
class BufferSet {
 public:
  const std::set<ObjectID>& AllBufferIds() const { return buffer_ids_; }

  const std::map<ObjectID, std::shared_ptr<Buffer>>& AllBuffers() const {
    return buffers_;
  }

  Status EmplaceBuffer(ObjectID const id);

  Status EmplaceBuffer(ObjectID const id,
                       std::shared_ptr<Buffer> const& buffer);

  void Extend(BufferSet const& others);

  void Extend(std::shared_ptr<BufferSet> const& others);

  bool Contains(ObjectID const id) const;

  bool Get(ObjectID const id, std::shared_ptr<Buffer>& buffer) const;

 private:
  // blob ids to buffer mapping: local blobs (not null) + remote blobs (null).
  std::set<ObjectID> buffer_ids_;
  std::map<ObjectID, std::shared_ptr<Buffer>> buffers_;
};

}  // namespace vineyard

#endif  // SRC_CLIENT_DS_BLOB_H_
