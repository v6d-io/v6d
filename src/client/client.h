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

#ifndef SRC_CLIENT_CLIENT_H_
#define SRC_CLIENT_CLIENT_H_

#include <sys/mman.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "arrow/buffer.h"

#include "client/client_base.h"
#include "client/ds/i_object.h"
#include "client/ds/object_meta.h"
#include "common/memory/payload.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

class Blob;
class BlobWriter;

/**
 * @brief MmapEntry represents a memory-mapped fd on the client side. The fd
 * can be mmapped as readonly or readwrite memory.
 */
class MmapEntry {
 public:
  MmapEntry(int fd, int64_t map_size, bool readonly)
      : fd_(fd), ro_pointer_(nullptr), rw_pointer_(nullptr), length_(0) {
    // fake_mmap in malloc.h leaves a gap between memory segments, to make
    // map_size page-aligned again.
    length_ = map_size - sizeof(size_t);
  }

  ~MmapEntry() {
    if (ro_pointer_) {
      int r = munmap(ro_pointer_, length_);
      if (r != 0) {
        LOG(ERROR) << "munmap returned " << r << ", errno = " << errno << ": "
                   << strerror(errno);
      }
    }
    if (rw_pointer_) {
      int r = munmap(rw_pointer_, length_);
      if (r != 0) {
        LOG(ERROR) << "munmap returned " << r << ", errno = " << errno << ": "
                   << strerror(errno);
      }
    }
    close(fd_);
  }

  /**
   * @brief Map the shared memory represents by `fd_` as readonly memory.
   *
   * @returns A untyped pointer that points to the shared readonly memory.
   */
  uint8_t* map_readonly() {
    if (!ro_pointer_) {
      ro_pointer_ = reinterpret_cast<uint8_t*>(
          mmap(NULL, length_, PROT_READ, MAP_SHARED, fd_, 0));
      if (ro_pointer_ == MAP_FAILED) {
        LOG(ERROR) << "mmap failed: errno = " << errno << ": "
                   << strerror(errno);
        ro_pointer_ = nullptr;
      }
    }
    return ro_pointer_;
  }

  /**
   * @brief Map the shared memory represents by `fd_` as writeable memory.
   *
   * @returns A untyped pointer that points to the shared writeable memory.
   */
  uint8_t* map_readwrite() {
    if (!rw_pointer_) {
      rw_pointer_ = reinterpret_cast<uint8_t*>(
          mmap(NULL, length_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
      if (rw_pointer_ == MAP_FAILED) {
        LOG(ERROR) << "mmap failed: errno = " << errno << ": "
                   << strerror(errno);
        rw_pointer_ = nullptr;
      }
    }
    return rw_pointer_;
  }

  int fd() { return fd_; }

 private:
  /// The associated file descriptor on the client.
  int fd_;
  /// The result of mmap for this file descriptor.
  uint8_t *ro_pointer_, *rw_pointer_;
  /// The length of the memory-mapped file.
  size_t length_;
};

/**
 * @brief Vineyard's IPC Client connects to to UNIX domain socket of the
 *        vineyard server. Vineyard's IPC Client talks to vineyard server
 *        and manipulate objects in vineyard.
 */
class Client : public ClientBase {
 public:
  ~Client() override;

  /**
   * @brief Connect to vineyard using the UNIX domain socket file specified by
   *        the environment variable `VINEYARD_IPC_SOCKET`.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect();

  /**
   * @brief Connect to vineyardd using the given UNIX domain socket
   * `ipc_socket`.
   *
   * @param ipc_socket Location of the UNIX domain socket.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(const std::string& ipc_socket);

  /**
   * @brief Get a default client reference, using the UNIX domain socket file
   *        specified by the environment variable `VINEYARD_IPC_SOCKET`.
   *
   * @return A reference of the default Client instance.
   */
  static Client& Default();

  /**
   * @brief Obtain metadata from vineyard server.
   *
   * @param id The object id to get.
   * @param meta_data The result metadata will be store in `meta_data` as return
   * value.
   * @param sync_remote Whether to trigger an immediate remote metadata
   *        synchronization before get specific metadata. Default is false.
   *
   * @return Status that indicates whether the get action has succeeded.
   */
  Status GetMetaData(const ObjectID id, ObjectMeta& meta_data,
                     const bool sync_remote = false) override;

  /**
   * @brief Obtain multiple metadatas from vineyard server.
   *
   * @param ids The object ids to get.
   * @param meta_data The result metadata will be store in `meta_data` as return
   * value.
   * @param sync_remote Whether to trigger an immediate remote metadata
   *        synchronization before get specific metadata. Default is false.
   *
   * @return Status that indicates whether the get action has succeeded.
   */
  Status GetMetaData(const std::vector<ObjectID>& id, std::vector<ObjectMeta>&,
                     const bool sync_remote = false);

  /**
   * @brief Create a blob in vineyard server. When creating a blob, vineyard
   * server's bulk allocator will prepare a block of memory of the requested
   * size, the map the memory to client's process to share the allocated memory.
   *
   * @param size The size of requested blob.
   * @param blob The result mutable blob will be set in `blob`.
   *
   * @return Status that indicates whether the create action has succeeded.
   */
  Status CreateBlob(size_t size, std::unique_ptr<BlobWriter>& blob);

  /**
   * @brief Allocate a stream on vineyard. The metadata of parameter `id` must
   * has already been created on vineyard.
   *
   * @param id The id of metadata that will be used to create stream.
   *
   * @return Status that indicates whether the create action has succeeded.
   */
  Status CreateStream(const ObjectID& id);

  /**
   * @brief Allocate a chunk of given size in vineyard for a stream. When the
   * request cannot be statisfied immediately, e.g., vineyard doesn't have
   * enough memory or the specified has accumulated too many chunks, the request
   * will be blocked until the request been processed.
   *
   * @param id The id of the stream.
   * @param size The size of the chunk to allocate.
   * @param blob The allocated mutable buffer will be set in `blob`.
   *
   * @return Status that indicates whether the allocation has succeeded.
   */
  Status GetNextStreamChunk(ObjectID const id, size_t const size,
                            std::unique_ptr<arrow::MutableBuffer>& blob);

  /**
   * @brief Poll a chunk from a stream. When there's no more chunk available in
   * the stream, i.e., the stream has been stoped, a status code
   * `kStreamDrained` or `kStreamFinish` will be returned, otherwise the reader
   * will be blocked until writer creates a new chunk in the stream.
   *
   * @param id The id of the stream.
   * @param blob The immutable chunk generated by the writer of the stream.
   *
   * @return Status that indicates whether the polling has succeeded.
   */
  Status PullNextStreamChunk(ObjectID const id,
                             std::unique_ptr<arrow::Buffer>& blob);

  /**
   * @brief Stop a stream, mark it as finished or aborted.
   *
   * @param id The id of the stream.
   * @param failed Whether the stream is stoped at a successful state. True
   * means the stream has been exited normally, otherwise false.
   *
   * @return Status that indicates whether the request has succeeded.
   */
  Status StopStream(ObjectID const id, bool failed);

  /**
   * @brief Get an object from vineyard. The ObjectFactory will be used to
   * resolve the constructor of the object.
   *
   * @param id The object id to get.
   *
   * @return A std::shared_ptr<Object> that can be safely cast to the underlying
   * concrete object type. When the object doesn't exists an std::runtime_error
   * exception will be raised.
   */
  std::shared_ptr<Object> GetObject(const ObjectID id);

  /**
   * @brief Get an object from vineyard. The ObjectFactory will be used to
   * resolve the constructor of the object.
   *
   * @param id The object id to get.
   * @param object The result object will be set in parameter `object`.
   *
   * @return When errors occur during the request, this method won't throw
   * exceptions, rather, it results a status to represents the error.
   */
  Status GetObject(const ObjectID id, std::shared_ptr<Object>& object);

  /**
   * @brief Get an object from vineyard. The type parameter `T` will be used to
   * resolve the constructor of the object.
   *
   * @param id The object id to get.
   *
   * @return A std::shared_ptr<Object> that can be safely cast to the underlying
   * concrete object type. When the object doesn't exists an std::runtime_error
   * exception will be raised.
   */
  template <typename T>
  std::shared_ptr<T> GetObject(const ObjectID id) {
    return std::dynamic_pointer_cast<T>(GetObject(id));
  }

  /**
   * @brief Get an object from vineyard. The type parameter `T` will be used to
   * resolve the constructor of the object.
   *
   * This method can be used to get concrete object from vineyard without
   * explicitly `dynamic_cast`, and the template type parameter can be deduced
   * in many situations:
   *
   * \code{.cpp}
   *    std::shared_ptr<Array<int>> int_array;
   *    client.GetObject(id, int_array);
   * \endcode
   *
   * @param id The object id to get.
   * @param object The result object will be set in parameter `object`.
   *
   * @return When errors occur during the request, this method won't throw
   * exceptions, rather, it results a status to represents the error.
   */
  template <typename T>
  Status GetObject(const ObjectID id, std::shared_ptr<T>& object) {
    object = std::dynamic_pointer_cast<T>(GetObject(id));
    if (object == nullptr) {
      return Status::ObjectNotExists();
    } else {
      return Status::OK();
    }
  }

  /**
   * @brief Get multiple objects from vineayrd.
   *
   * @param ids The object IDs to get.
   *
   * @return A list of objects.
   */
  std::vector<std::shared_ptr<Object>> GetObjects(
      const std::vector<ObjectID>& ids);

  /**
   * @brief List objects in vineyard, using the given typename patterns.
   *
   * @param pattern The pattern string that will be used to matched against
   * objects' `typename`.
   * @param regex Whether the pattern is a regular expression pattern. Default
   * is false. When `regex` is false, the pattern will be treated as a glob
   * pattern.
   * @param limit The number limit for how many objects will be returned at
   * most.
   *
   * @return A vector of objects that listed from vineyard server.
   */
  std::vector<std::shared_ptr<Object>> ListObjects(std::string const& pattern,
                                                   const bool regex = false,
                                                   size_t const limit = 5);

 private:
  Status CreateBuffer(const size_t size, ObjectID& id, Payload& object);

  Status GetBuffer(const ObjectID id, Payload& object);

  Status GetBuffers(const std::unordered_set<ObjectID>& ids,
                    std::unordered_map<ObjectID, Payload>& objects);

  Status mmapToClient(int fd, int64_t map_size, bool readonly, uint8_t** ptr);

  std::unordered_map<int, std::unique_ptr<MmapEntry>> mmap_table_;

  friend class Blob;
  friend class BlobWriter;
};

}  // namespace vineyard

#endif  // SRC_CLIENT_CLIENT_H_
