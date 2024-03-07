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

#ifndef SRC_CLIENT_RPC_CLIENT_H_
#define SRC_CLIENT_RPC_CLIENT_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "client/client_base.h"
#include "client/ds/i_object.h"
#include "client/ds/object_meta.h"
#include "client/ds/remote_blob.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

namespace vineyard {

class Blob;
class BlobWriter;

class RPCClient final : public ClientBase {
 public:
  ~RPCClient() override;

  /**
   * @brief Connect to vineyard using the TCP endpoint specified by
   *        the environment variable `VINEYARD_RPC_ENDPOINT`.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect();

  /**
   * @brief Connect to vineyard using the TCP endpoint specified by
   *        the environment variable `VINEYARD_RPC_ENDPOINT`.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(std::string const& username, std::string const& password);

  /**
   * @brief Connect to vineyardd using the given TCP endpoint `rpc_endpoint`.
   *
   * @param rpc_endpoint The TPC endpoint of vineyard server, in the format of
   * `host:port`.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(const std::string& rpc_endpoint);

  /**
   * @brief Connect to vineyardd using the given TCP endpoint `rpc_endpoint`.
   *
   * @param rpc_endpoint The TPC endpoint of vineyard server, in the format of
   * `host:port`.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(const std::string& rpc_endpoint, std::string const& username,
                 std::string const& password);

  /**
   * @brief Connect to vineyardd using the given TCP endpoint `rpc_endpoint`.
   *
   * @param rpc_endpoint The TPC endpoint of vineyard server, in the format of
   * `host:port`.
   * @param session_id Connect to specified session.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(const std::string& rpc_endpoint, const SessionID session_id,
                 std::string const& username = "",
                 std::string const& password = "");

  /**
   * @brief Connect to vineyardd using the given TCP `host` and `port`.
   *
   * @param host The host of vineyard server.
   * @param port The TCP port of vineyard server's RPC service.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(const std::string& host, uint32_t port);

  /**
   * @brief Connect to vineyardd using the given TCP `host` and `port`.
   *
   * @param host The host of vineyard server.
   * @param port The TCP port of vineyard server's RPC service.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(const std::string& host, uint32_t port,
                 std::string const& username, std::string const& password);

  /**
   * @brief Connect to vineyardd using the given TCP `host` and `port`.
   *
   * @param host The host of vineyard server.
   * @param port The TCP port of vineyard server's RPC service.
   * @param session_id Connect to specified session.
   *
   * @return Status that indicates whether the connect has succeeded.
   */
  Status Connect(const std::string& host, uint32_t port,
                 const SessionID session_id, std::string const& username = "",
                 std::string const& password = "");

  /**
   * @brief Create a new client using self endpoint.
   */
  Status Fork(RPCClient& client);

  /**
   * @brief Obtain metadata from vineyard server. Note that unlike IPC client,
   * RPC client doesn't map shared memorys to the client's process.
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
  Status GetMetaData(const std::vector<ObjectID>& id,
                     std::vector<ObjectMeta>& meta_data,
                     const bool sync_remote = false);

  /**
   * @brief Get an object from vineyard. The ObjectFactory will be used to
   * resolve the constructor of the object.
   *
   * In RPCClient, all blob fields in the result object are unaccessible, access
   * those fields will trigger an `std::runtime_error`.
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
   * @brief Get multiple objects from vineyard.
   *
   * @param ids The object IDs to get.
   *
   * @return A list of objects.
   */
  std::vector<std::shared_ptr<Object>> GetObjects(
      const std::vector<ObjectID>& ids);

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
      return Status::ObjectTypeError(type_name<T>(),
                                     _object->meta().GetTypeName());
    } else {
      return Status::OK();
    }
  }

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
   * @brief Get the remote instance id of the connected vineyard server.
   *
   * Note that for RPC client the instance id is not available, thus we have
   * the "remote instance id" to indicate which server we are connecting to.
   *
   * @return The vineyard server's instance id.
   */
  const InstanceID remote_instance_id() const override {
    return remote_instance_id_;
  }

  /**
   * @brief Check if the client is a RPC client.
   *
   * @return True means the client is a RPC client.
   */
  bool IsRPC() const override { return true; }

  /**
   * @brief Whether the instance connected by rpc client is the same as object
   * metadata's instance.
   *
   * @return True means the instance is the same as object metadata's instance.
   */
  bool IsFetchable(const ObjectMeta& meta);

  /**
   * @brief Create a remote blob on the vineyard server.
   */
  Status CreateRemoteBlob(std::shared_ptr<RemoteBlobWriter> const& buffer,
                          ObjectMeta& meta);

  /**
   * @brief Create remote blobs on the vineyard server.
   */
  Status CreateRemoteBlobs(
      std::vector<std::shared_ptr<RemoteBlobWriter>> const& buffers,
      std::vector<ObjectMeta>& metas);

  /**
   * @brief Get the remote blob of the connected vineyard server, using the RPC
   * socket.
   *
   * Note that getting remote blobs requires an expensive copy over network.
   */
  Status GetRemoteBlob(const ObjectID& id, std::shared_ptr<RemoteBlob>& buffer);

  /**
   * @brief Get the remote blob of the connected vineyard server, using the RPC
   * socket, and optionally bypass the "seal" check.
   *
   * Note that getting remote blobs requires an expensive copy over network.
   */
  Status GetRemoteBlob(const ObjectID& id, const bool unsafe,
                       std::shared_ptr<RemoteBlob>& buffer);

  /**
   * @brief Get the remote blobs of the connected vineyard server, using the RPC
   * socket.
   *
   * Note that getting remote blobs requires an expensive copy over network.
   */
  Status GetRemoteBlobs(std::vector<ObjectID> const& ids,
                        std::vector<std::shared_ptr<RemoteBlob>>& remote_blobs);

  Status GetRemoteBlobs(
      std::set<ObjectID> const& ids,
      std::map<ObjectID, std::shared_ptr<RemoteBlob>>& remote_blobs);

  /**
   * @brief Get the remote blobs of the connected vineyard server, using the RPC
   * socket. and optionally bypass the "seal" check.
   *
   * Note that getting remote blobs requires an expensive copy over network.
   */
  Status GetRemoteBlobs(std::vector<ObjectID> const& ids, const bool unsafe,
                        std::vector<std::shared_ptr<RemoteBlob>>& remote_blobs);

  Status GetRemoteBlobs(
      std::set<ObjectID> const& ids, const bool unsafe,
      std::map<ObjectID, std::shared_ptr<RemoteBlob>>& remote_blobs);
  /**
   * @brief Try to acquire a distributed lock.
   *
   * @param key The key of the lock.
   *
   * @return Status that indicates whether the lock process succeeds.
   */
  Status TryAcquireLock(std::string key, bool& result,
                        std::string& actual_key) override {
    // TBD
    return Status::NotImplemented("TryAcquireLock is not implemented yet.");
  }

  /**
   * @brief Try to release a distributed lock.
   *
   * @param key The key of the lock.
   *
   * @return Status that indicates whether the unlock process succeeds.
   */
  Status TryReleaseLock(std::string key, bool& result) override {
    // TBD
    return Status::NotImplemented("TryAcquireLock is not implemented yet.");
  }

 private:
  InstanceID remote_instance_id_;

  friend class Client;
};

}  // namespace vineyard

#endif  // SRC_CLIENT_RPC_CLIENT_H_
