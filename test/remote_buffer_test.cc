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

#include <memory>
#include <string>
#include <thread>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/array.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "client/rpc_client.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

constexpr uint64_t element_num =
    (uint64_t) 1024 * 1024 * 1024 * 16 / sizeof(uint64_t);

void RemoteCreateTest(Client& ipc_client, RPCClient& rpc_client) {
  std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};
  char* array_data = reinterpret_cast<char*>(double_array.data());

  // create remote buffer
  auto remote_blob_writer =
      std::make_shared<RemoteBlobWriter>(double_array.size() * sizeof(double));
  std::memcpy(remote_blob_writer->data(), array_data,
              double_array.size() * sizeof(double));
  ObjectMeta blob_meta;
  VINEYARD_CHECK_OK(rpc_client.CreateRemoteBlob(remote_blob_writer, blob_meta));
  ObjectID blob_id = blob_meta.GetId();
  CHECK_NE(blob_id, InvalidObjectID());

  LOG(INFO) << "created blob: " << ObjectIDToString(blob_id);

  // get local buffer
  std::shared_ptr<Blob> local_buffer;
  VINEYARD_CHECK_OK(ipc_client.GetBlob(blob_id, local_buffer));

  // check the local buffer
  CHECK_EQ(local_buffer->id(), blob_id);
  CHECK_EQ(local_buffer->allocated_size(),
           double_array.size() * sizeof(double));

  // compare the buffer contents
  for (size_t index = 0; index < local_buffer->allocated_size(); ++index) {
    CHECK_EQ(local_buffer->data()[index], array_data[index]);
  }

  LOG(INFO) << "Passed remote buffer (remote create & local get) tests...";
}

void RemoteGetTest(Client& ipc_client, RPCClient& rpc_client) {
  std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};

  ArrayBuilder<double> builder(ipc_client, double_array);
  auto sealed_double_array =
      std::dynamic_pointer_cast<Array<double>>(builder.Seal(ipc_client));
  VINEYARD_CHECK_OK(ipc_client.Persist(sealed_double_array->id()));

  ObjectID id = sealed_double_array->id();
  ObjectMeta array_meta;
  VINEYARD_CHECK_OK(ipc_client.GetMetaData(id, array_meta));
  ObjectID blob_id = array_meta.GetMemberMeta("buffer_").GetId();

  LOG(INFO) << "created blob: " << ObjectIDToString(blob_id);

  // get local buffer
  std::shared_ptr<Blob> local_buffer;
  VINEYARD_CHECK_OK(ipc_client.GetBlob(blob_id, local_buffer));

  // get remote buffer
  std::shared_ptr<RemoteBlob> remote_buffer;
  VINEYARD_CHECK_OK(rpc_client.GetRemoteBlob(blob_id, remote_buffer));

  // check the remote buffer
  CHECK_EQ(remote_buffer->id(), blob_id);
  CHECK_EQ(remote_buffer->instance_id(), ipc_client.instance_id());
  CHECK_EQ(remote_buffer->instance_id(), rpc_client.remote_instance_id());
  CHECK_EQ(remote_buffer->allocated_size(), local_buffer->allocated_size());

  // compare the buffer contents
  for (size_t index = 0; index < local_buffer->allocated_size(); ++index) {
    CHECK_EQ(local_buffer->data()[index], remote_buffer->data()[index]);
  }

  LOG(INFO) << "Passed remote buffer (local create & remote get) tests...";
}

void RemoteCreateAndGetTest(Client& ipc_client, RPCClient& rpc_client) {
  std::vector<double> double_array = {1.0, 7.0, 3.0, 4.0, 2.0};
  char* array_data = reinterpret_cast<char*>(double_array.data());

  // create remote buffer
  auto remote_blob_writer =
      std::make_shared<RemoteBlobWriter>(double_array.size() * sizeof(double));
  std::memcpy(remote_blob_writer->data(), array_data,
              double_array.size() * sizeof(double));
  ObjectMeta blob_meta;
  VINEYARD_CHECK_OK(rpc_client.CreateRemoteBlob(remote_blob_writer, blob_meta));
  ObjectID blob_id = blob_meta.GetId();
  CHECK_NE(blob_id, InvalidObjectID());

  LOG(INFO) << "created blob: " << ObjectIDToString(blob_id);

  // get remote buffer
  std::shared_ptr<RemoteBlob> remote_buffer;
  VINEYARD_CHECK_OK(rpc_client.GetRemoteBlob(blob_id, remote_buffer));

  // check the remote buffer
  CHECK_EQ(remote_buffer->id(), blob_id);
  CHECK_EQ(remote_buffer->instance_id(), ipc_client.instance_id());
  CHECK_EQ(remote_buffer->instance_id(), rpc_client.remote_instance_id());
  CHECK_EQ(remote_buffer->allocated_size(),
           double_array.size() * sizeof(double));

  // compare the buffer contents
  for (size_t index = 0; index < remote_buffer->allocated_size(); ++index) {
    CHECK_EQ(remote_buffer->data()[index], array_data[index]);
  }

  LOG(INFO) << "Passed remote buffer (remote create & remote get) tests...";
}

void RemoteBigArrayCreateAndGetTest(Client& ipc_client, RPCClient& rpc_client) {
  std::vector<uint64_t> big_array_vec;
  for (size_t i = 0; i < element_num; ++i) {
    big_array_vec.push_back(i + 1000);
  }
  uint64_t const* array_data = big_array_vec.data();

  // create remote buffer
  auto remote_blob_writer = std::make_shared<RemoteBlobWriter>(
      big_array_vec.size() * sizeof(uint64_t));
  std::memcpy(remote_blob_writer->data(), array_data,
              big_array_vec.size() * sizeof(uint64_t));
  ObjectMeta blob_meta;
  VINEYARD_CHECK_OK(rpc_client.CreateRemoteBlob(remote_blob_writer, blob_meta));
  ObjectID blob_id = blob_meta.GetId();
  CHECK_NE(blob_id, InvalidObjectID());

  LOG(INFO) << "created blob: " << ObjectIDToString(blob_id);

  // get remote buffer
  std::shared_ptr<RemoteBlob> remote_buffer;
  VINEYARD_CHECK_OK(rpc_client.GetRemoteBlob(blob_id, remote_buffer));

  // check the remote buffer
  CHECK_EQ(remote_buffer->id(), blob_id);
  CHECK_EQ(remote_buffer->instance_id(), ipc_client.instance_id());
  CHECK_EQ(remote_buffer->instance_id(), rpc_client.remote_instance_id());
  CHECK_EQ(remote_buffer->allocated_size(),
           big_array_vec.size() * sizeof(uint64_t));

  // compare the buffer contents
  for (size_t index = 0; index < element_num; ++index) {
    CHECK_EQ((reinterpret_cast<const uint64_t*>(remote_buffer->data()))[index],
             array_data[index]);
  }

  LOG(INFO)
      << "Passed remote big size buffer (remote create & remote get) tests...";
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf(
        "usage ./remote_buffer_test <ipc_socket> <rpc_endpoint> "
        "<rdma_endpoint>");
    return 1;
  }
  std::string ipc_socket(argv[1]);
  std::string rpc_endpoint(argv[2]);
  std::string rdma_endpoint = "";
  if (argc > 3) {
    rdma_endpoint = std::string(argv[3]);
  }

  Client ipc_client;
  VINEYARD_CHECK_OK(ipc_client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  RPCClient rpc_client;
  VINEYARD_CHECK_OK(rpc_client.Connect(rpc_endpoint, "", "", rdma_endpoint));
  LOG(INFO) << "Connected to RPCServer: " << rpc_endpoint;

  RemoteCreateTest(ipc_client, rpc_client);
  RemoteGetTest(ipc_client, rpc_client);
  RemoteCreateAndGetTest(ipc_client, rpc_client);
  RemoteBigArrayCreateAndGetTest(ipc_client, rpc_client);

  LOG(INFO) << "Passed remote buffer tests...";

  ipc_client.Disconnect();
  rpc_client.Disconnect();

  return 0;
}
