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

#ifndef SRC_CLIENT_CLIENT_H_
#define SRC_CLIENT_CLIENT_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "client/client_base.h"
#include "client/ds/i_object.h"
#include "client/ds/object_meta.h"
#include "common/memory/payload.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

class Blob;
class BlobWriter;

namespace detail {

/**
 * @brief MmapEntry represents a memory-mapped fd on the client side. The fd
 * can be mmapped as readonly or readwrite memory.
 */
class MmapEntry {
 public:
  MmapEntry(int fd, int64_t map_size, bool readonly, bool realign = false);

  ~MmapEntry();

  /**
   * @brief Map the shared memory represents by `fd_` as readonly memory.
   *
   * @returns A untyped pointer that points to the shared readonly memory.
   */
  uint8_t* map_readonly();

  /**
   * @brief Map the shared memory represents by `fd_` as writeable memory.
   *
   * @returns A untyped pointer that points to the shared writeable memory.
   */
  uint8_t* map_readwrite();

  int fd() { return fd_; }

 private:
  /// The associated file descriptor on the client.
  int fd_;
  /// The result of mmap for this file descriptor.
  uint8_t *ro_pointer_, *rw_pointer_;
  /// The length of the memory-mapped file.
  size_t length_;
};

class SharedMemoryManager {
 public:
  explicit SharedMemoryManager(int vineyard_conn);

  Status Mmap(int fd, int64_t map_size, bool readonly, bool realign,
              uint8_t** ptr);

  bool Exists(const uintptr_t target);

  bool Exists(const void* target);

 private:
  // UNIX-domain socket
  int vineyard_conn_ = -1;

  // mmap table
  std::unordered_map<int, std::unique_ptr<MmapEntry>> mmap_table_;

  // sorted shm segments for fast "if exists" query
  std::set<std::pair<uintptr_t, size_t>> segments_;
};

}  // namespace detail

/**
 * @brief Vineyard's IPC Client connects to to UNIX domain socket of the
 *        vineyard server. Vineyard's IPC Client talks to vineyard server
 *        and manipulate objects in vineyard.
 */
class Client : public ClientBase {
 public:
  Client();

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
   * @brief Create a new anonymous session in vineyardd and connect to it .
   *
   * @param ipc_socket Location of the UNIX domain socket.
   *
   * @return Status that indicates whether the connection of has succeeded.
   */
  Status Open(std::string const& ipc_socket);

  /**
   * @brief Create a new client using self UNIX domain socket.
   */
  Status Fork(Client& client);

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

  // bring the overloadings in parent class to current scope.
  using ClientBase::PullNextStreamChunk;

  /**
   * @brief Pull a chunk from a stream. When there's no more chunk available in
   * the stream, i.e., the stream has been stoped, a status code
   * `kStreamDrained` or `kStreamFinish` will be returned, otherwise the reader
   * will be blocked until writer creates a new chunk in the stream.
   *
   * @param id The id of the stream.
   * @param chunk The immutable chunk generated by the writer of the stream.
   *
   * @return Status that indicates whether the polling has succeeded.
   */
  Status PullNextStreamChunk(ObjectID const id,
                             std::unique_ptr<arrow::Buffer>& chunk);

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
    std::shared_ptr<Object> _object;
    RETURN_ON_ERROR(GetObject(id, _object));
    object = std::dynamic_pointer_cast<T>(_object);
    if (object == nullptr) {
      return Status::ObjectNotExists("object not exists: " +
                                     ObjectIDToString(id));
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
   * @brief List object metadatas in vineyard, using the given typename
   * patterns.
   *
   * @param pattern The pattern string that will be used to matched against
   * objects' `typename`.
   * @param regex Whether the pattern is a regular expression pattern. Default
   * is false. When `regex` is false, the pattern will be treated as a glob
   * pattern.
   * @param limit The number limit for how many objects will be returned at
   * most.
   *
   * @return A vector of object metadatas that listed from vineyard server.
   */
  std::vector<ObjectMeta> ListObjectMeta(std::string const& pattern,
                                         const bool regex = false,
                                         size_t const limit = 5,
                                         bool nobuffer = false);

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

  /**
   * Check if the given address belongs to the shared memory region.
   *
   * Return true if the address (client-side address) comes from the vineyard
   * server.
   */
  bool IsSharedMemory(const void* target) const;

  /**
   * Check if the given address belongs to the shared memory region.
   *
   * Return true if the address (client-side address) comes from the vineyard
   * server.
   */
  bool IsSharedMemory(const uintptr_t target) const;

  /**
   * Get the allocated size for the given object.
   */
  Status AllocatedSize(const ObjectID id, size_t& size);

  Status CreateArena(const size_t size, int& fd, size_t& available_size,
                     uintptr_t& base, uintptr_t& space);

  Status ReleaseArena(const int fd, std::vector<size_t> const& offsets,
                      std::vector<size_t> const& sizes);

 protected:
  Status CreateBuffer(const size_t size, ObjectID& id, Payload& payload,
                      std::shared_ptr<arrow::MutableBuffer>& buffer);

  /**
   * @brief Get a blob from vineyard server. When obtaining blobs from vineyard
   * server, the memory address in the server process will be mmapped to the
   * client's process to share the memory.
   *
   * @param id Object id for the blob to get.
   * @param buffer: The result immutable blob will be set in `blob`. Note that
   * blob is special, since it can be get as immutable object before sealing.
   *
   * @return Status that indicates whether the create action has succeeded.
   */
  Status GetBuffer(const ObjectID id, std::shared_ptr<arrow::Buffer>& buffer);

  /**
   * @brief Get a set of blobs from vineyard server. See also `GetBuffer`.
   *
   * @param ids Object ids for the blobs to get.
   * @param buffers: The result immutable blobs will be added to `buffers`.
   *
   * @return Status that indicates whether the create action has succeeded.
   */
  Status GetBuffers(
      const std::set<ObjectID>& ids,
      std::map<ObjectID, std::shared_ptr<arrow::Buffer>>& buffers);

  Status GetBufferSizes(const std::set<ObjectID>& ids,
                        std::map<ObjectID, size_t>& sizes);

  /**
   * @brief An (unsafe) internal-usage method that drop the buffer, without
   * checking the dependency. To achieve the "DeleteObject" semantic for blobs,
   * use `DelData` instead.
   *
   * Note that the target buffer could be both sealed and unsealed.
   */
  Status DropBuffer(const ObjectID id, const int fd);

 private:
  std::shared_ptr<detail::SharedMemoryManager> shm_;

 private:
  friend class Blob;
  friend class BlobWriter;
};

}  // namespace vineyard

#endif  // SRC_CLIENT_CLIENT_H_
