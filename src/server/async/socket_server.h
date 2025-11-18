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

#ifndef SRC_SERVER_ASYNC_SOCKET_SERVER_H_
#define SRC_SERVER_ASYNC_SOCKET_SERVER_H_

#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/memory/payload.h"
#include "common/util/asio.h"  // IWYU pragma: keep
#include "common/util/callback.h"
#include "common/util/sidecar.h"
#include "common/util/uuid.h"

#include "common/rdma/rdma_server.h"

#include "thread-pool/thread_pool.h"

namespace vineyard {

using boost::asio::generic::stream_protocol;

class SocketServer;
class IPCServer;
class RPCServer;
class BulkStore;
class PlasmaBulkStore;
class VineyardServer;

using socket_message_queue_t = std::deque<std::string>;

/**
 * @brief SocketConnection handles the socket connection in vineyard
 *
 */
class SocketConnection : public std::enable_shared_from_this<SocketConnection> {
 public:
  SocketConnection(stream_protocol::socket socket,
                   std::shared_ptr<VineyardServer> server_ptr,
                   std::shared_ptr<SocketServer> socket_server_ptr, int conn_id,
                   std::string peer_host = "");

  bool Start();

  /**
   * @brief Invoke internal doStop, and set the running status to false.
   */
  bool Stop();

 protected:
  bool doRegister(json const& root);
  bool doRequireExtraRequestMemory(json const& root);

  bool doCreateBuffer(json const& root);
  bool doCreateBuffers(json const& root);
  bool doCreateDiskBuffer(json const& root);
  bool doCreateGPUBuffer(json const& root);
  bool doSealBlob(json const& root);
  bool doGetBuffers(json const& root);
  bool doGetGPUBuffers(json const& root);
  bool doDropBuffer(json const& root);
  bool doShrinkBuffer(json const& root);

  /**
   * @brief doCreateBuffer differs from doCreateRemoteBuffer, that the content
   * of blob is in the request body, rather than via memory sharing.
   */
  bool doCreateRemoteBuffer(json const& root);
  bool doCreateRemoteBuffers(json const& root);

  /**
   * @brief doGetRemoteBuffers differs from doGetRemoteBuffers, that the
   * content of blob is in the response body, rather than via memory sharing.
   */
  bool doGetRemoteBuffers(json const& root);

  bool doIncreaseReferenceCount(json const& root);
  bool doRelease(json const& root);
  bool doDelDataWithFeedbacks(json const& root);
  bool doDelHugeData(json const& root);

  bool doCreateBufferByPlasma(json const& root);
  bool doGetBuffersByPlasma(json const& root);
  bool doSealPlasmaBlob(json const& root);
  bool doPlasmaRelease(json const& root);
  bool doPlasmaDelData(json const& root);
  bool doCreateUserBuffers(json const& root);
  bool doDeleteUserBuffers(json const& root);
  bool doGetUserBuffers(json const& root);

  bool doCreateData(json const& root);
  bool doCreateDatas(json const& root);
  bool doGetData(json const& root);
  bool doListData(json const& root);
  bool doDelData(json const& root);
  bool doExists(json const& root);
  bool doPersist(json const& root);
  bool doIfPersist(json const& root);
  bool doLabelObject(json const& root);
  bool doClear(json const& root);
  bool doMemoryTrim(json const& root);
  bool doBatchPersist(json const& root);
  bool doCreatehugeDatas(json const& root);
  bool doGetHugeData(json const& root);

  bool doCreateStream(json const& root);
  bool doOpenStream(json const& root);
  bool doGetNextStreamChunk(json const& root);
  bool doPushNextStreamChunk(json const& root);
  bool doPushNextStreamChunkByOffset(json const& root);
  bool doPullNextStreamChunk(json const& root);
  bool doCheckFixedStreamReceived(json const& root);
  bool doStopStream(json const& root);
  bool doDropStream(json const& root);
  bool doAbortStream(json const& root);
  bool doPutStreamName(json const& root);
  bool doGetStreamIDByName(json const& root);
  bool doActivateRemoteFixedStream(const json& root);
  bool doCreateFixedStream(json const& root);
  bool doOpenFixedStream(json const& root);
  bool doCloseStream(json const& root);
  bool doDeleteStream(json const& root);

  bool doVineyardOpenRemoteFixedStream(const json& root);
  bool doVineyardActivateRemoteFixedStreamWithOffset(const json& root);
  bool doVineyardStopStream(const json& root);
  bool doVineyardDropStream(const json& root);
  bool doVineyardAbortRemoteStream(const json& root);
  bool doVineyardCloseRemoteFixedStream(const json& root);
  bool doVineyardGetMetasByNames(const json& root);
  bool doVineyardGetRemoteBlobs(const json& root);
  bool doVineyardGetRemoteBlobsWithOffset(const json& root);

  bool doPutName(json const& root);
  bool doGetName(json const& root);
  bool doListName(json const& root);
  bool doDropName(json const& root);
  bool doGetObjectLocation(const json& root);
  bool doPutObjectLocation(const json& root);
  bool doGetNames(json const& root);
  bool doDropNames(json const& root);
  bool doPutNames(json const& root);

  bool doMakeArena(json const& root);
  bool doFinalizeArena(json const& root);

  bool doNewSession(json const& root);
  bool doDeleteSession(json const& root);
  bool doMoveBuffersOwnership(json const& root);

  bool doEvictObjects(json const& root);
  bool doLoadObjects(json const& root);
  bool doUnpinObjects(json const& root);
  bool doIsSpilled(json const& root);
  bool doIsInUse(json const& root);

  bool doGetVineyardMmapFd(json const& root);
  bool doClusterMeta(json const& root);
  bool doInstanceStatus(json const& root);
  bool doMigrateObject(json const& root);
  bool doShallowCopy(json const& root);

  bool doDebug(json const& root);

  bool doAcquireLock(json const& root);
  bool doReleaseLock(json const& root);

  bool doReleaseBlobsWithRDMA(json const& root);

 protected:
  template <typename FROM, typename TO>
  Status MoveBuffers(std::map<FROM, TO> mapping,
                     std::shared_ptr<VineyardServer>& source_session);

  int nativeHandle() { return socket_.native_handle(); }

  int getConnId() { return conn_id_; }

  virtual bool sendFd(int fd);

  /**
   * @brief Return should be exit after this message.
   *
   * @return Returns true if stop the client and close the connection (i.e.,
   * handling a ExitRequest), otherwise false.
   */
  bool processMessage(const std::string& message_in);

  void doReadHeader();

  void doReadBody();

  virtual void doWrite(const std::string& buf);

  virtual void doWriteWithoutRead(std::string& buf);

  virtual void doWrite(std::string&& buf);

  virtual void doWrite(const std::string& buf, callback_t<> callback,
                       const bool partial = false);

  /**
   * Being called when the encounter a socket error (in read/write), or by
   * external "conn->Stop()" (from the `SocketServer`.).
   *
   * Just do some clean up and won't remove connection from parent's pool.
   */
  void doStop();

  void doAsyncWrite(std::string&& buf);

  void doAsyncWrite(std::string&& buf, callback_t<> callback,
                    const bool partial = false);

  void doAsyncWriteWithoutRead(std::string&& buf);

  void switchSession(std::shared_ptr<VineyardServer>& session) {
    this->server_ptr_ = session;
  }

  void LockTransmissionObjects(const std::vector<ObjectID>& ids);

  void UnlockTransmissionObjects(const std::vector<ObjectID>& ids);

  void ClearLockedObjects();

  // TODO: remove this
  void ThrowException();

  // whether the connection has been correctly "registered"
  std::atomic_bool registered_;

  stream_protocol::socket socket_;
  std::shared_ptr<VineyardServer> server_ptr_;
  std::shared_ptr<SocketServer> socket_server_ptr_;

  // hold a reference of the bulkstore to avoid dtor conflict.
  std::shared_ptr<BulkStore> bulk_store_;
  std::shared_ptr<PlasmaBulkStore> plasma_bulk_store_;

  int conn_id_;
  std::atomic_bool running_;
  std::string peer_host;

  asio::streambuf buf_;

  std::unordered_set<int> used_fds_;
  // the associated reader of the stream
  std::unordered_set<ObjectID> associated_streams_;

  size_t read_msg_header_;
  std::string read_msg_body_;

  std::unordered_map<ObjectID, int> locked_objects_;
  std::mutex locked_objects_mutex_;

  int trace_log_level_ = 0;

  friend class IPCServer;
  friend class RPCServer;
};

/**
 * @brief SocketServer is the base class of IPCServer and RPCServer.
 *
 */
class SocketServer {
 public:
  explicit SocketServer(std::shared_ptr<VineyardServer> vs_ptr);

  /**
   * Stop the acceptor.
   */
  virtual ~SocketServer() {}

  virtual void Start();

  /**
   * Call "Stop" on all connections, then clear the connection pool.
   */
  virtual void Stop();

  /**
   * Cancel the "async_accept" action on the acceptor to stop accepting
   * further connections.
   */
  virtual void Close();

  /**
   * Check if @conn_id@ exists in the connection pool.
   */
  virtual bool ExistsConnection(int conn_id) const;

  /**
   * Remove @conn_id@ from connection pool, before removing, the "Stop"
   * on the connection has already been called.
   */
  virtual void RemoveConnection(int conn_id);

  /**
   * Invoke the "Stop" on the connection, and then remove it from the
   * connection pool.
   */
  virtual void CloseConnection(int conn_id);

  /**
   * Inspect the size of current alive connections.
   */
  size_t AliveConnections() const;

  /**
   * Register the new connection. RPC server will set the expected server ptr
   * to the connection object.
   */
  virtual Status Register(std::shared_ptr<SocketConnection> conn,
                          const SessionID session_id) = 0;

  virtual Status SendDataWithRDMA(int tcp_conn, uint64_t addr,
                                  uint64_t local_addr, size_t size,
                                  uint64_t rkey) {
    return Status::NotImplemented("SendDataWithRDMA is not implemented");
  }

  Status SendDataWithRDMA(std::vector<uint64_t>& addr_list, uint64_t local_addr,
                          size_t offset, std::vector<size_t>& size_list,
                          size_t& size, std::vector<uint64_t>& rkey_list,
                          const std::string& peer_host, const int port,
                          const std::string& advice_device) {
    return Status::NotImplemented("RDMA is not supported yet.");
  }

  Status RequireExtraRequestMemory(int conn_id, size_t size, int& fd);

  Status ReleaseExtraRequestMemory(int conn_id);

  Status ReadExtraMessage(void* data, size_t size, int conn_id);

  Status WriteExtraMessage(const void* data, size_t size, int conn_id);

  Status LseekExtraMsgWritePos(uint64_t offset, int conn_id);

  Status LseekExtraMsgReadPos(uint64_t offset, int conn_id);

  Status GetClientAttributeMsg(uint64_t conn_id, ClientAttributes& attr);

 protected:
  struct ExtraRequestMem {
    void* addr_ = nullptr;
    size_t size_ = 0;
    int fd_ = -1;
    uint64_t write_pos_ = 0;
    uint64_t read_pos_ = 0;
  };

  std::atomic_bool stopped_;  // if the socket server being stopped.

  std::atomic_bool closable_;  // if client want to close the session,
  std::shared_ptr<VineyardServer> vs_ptr_;
  int next_conn_id_;
  std::unordered_map<int, std::shared_ptr<SocketConnection>> connections_;
  mutable std::recursive_mutex connections_mutex_;  // protect `connections_`
  std::map<int, ExtraRequestMem> conn_id_to_extra_request_mem_;
  std::recursive_mutex conn_id_to_extra_request_mem_mutex_;

 private:
  virtual void doAccept() = 0;
};

}  // namespace vineyard

#endif  // SRC_SERVER_ASYNC_SOCKET_SERVER_H_
