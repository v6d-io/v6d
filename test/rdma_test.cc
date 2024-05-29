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
#include <vector>

#include "common/rdma/rdma_client.h"
#include "common/rdma/rdma_server.h"
#include "common/rdma/util.h"
#include "client/ds/blob.h"

using namespace vineyard; // NOLINT(build/namespaces)

constexpr int port = 9228;
std::shared_ptr<RDMAServer> server;
std::shared_ptr<RDMAClient> client;

RegisterMemInfo serverMemInfo;
RegisterMemInfo clientMemInfo;
#define MEM_SIZE 1024

void PrepareData(std::vector<int> &vec) {
  for (int i = 0; i < 100; i++) {
    vec.push_back(i);
  }
}

void HelloToServer() {
  void *msg = nullptr;
  VINEYARD_CHECK_OK(client->GetRXFreeMsgBuffer(msg));
  VINEYARD_CHECK_OK(client->Recv(msg, sizeof(VineyardMsg), nullptr));

  msg = nullptr;
  VINEYARD_CHECK_OK(client->GetTXFreeMsgBuffer(msg));
  char buf[] = "Hello, server!";
  VINEYARD_CHECK_OK(client->Write(buf, 15, (uint64_t)serverMemInfo.address, serverMemInfo.rkey, NULL, nullptr));
  VINEYARD_CHECK_OK(client->GetTXCompletion(-1, nullptr));
  LOG(INFO) << "Write to:" << (void *)serverMemInfo.address;

  msg = nullptr;
  VINEYARD_CHECK_OK(client->GetTXFreeMsgBuffer(msg));
  VINEYARD_CHECK_OK(client->Send(msg, sizeof(VineyardMsg), nullptr));

  VINEYARD_CHECK_OK(client->GetRXCompletion(-1, nullptr));
  LOG(INFO) << "Receive from server:" << (char *)clientMemInfo.address;
}

void HelloToClient() {
  void *msg = nullptr;
  server->GetRXFreeMsgBuffer(msg);
  server->Recv(TEST_CLIENT_ID, msg, sizeof(VineyardMsg), nullptr);

  char buf[] = "Hello, client!";
  server->Write(TEST_CLIENT_ID, buf, 15, (uint64_t)clientMemInfo.address, (uint64_t)clientMemInfo.rkey, NULL, nullptr);
  server->GetTXCompletion(-1, nullptr);

  msg = nullptr;
  server->GetTXFreeMsgBuffer(msg);
  server->Send(TEST_CLIENT_ID, msg, sizeof(VineyardMsg), nullptr);

  server->GetRXCompletion(-1, nullptr);
  LOG(INFO) << "Address: " << (void *)serverMemInfo.address;
  LOG(INFO) << "Receive from client:" << (char *)serverMemInfo.address;
}

void ClientExchangeKeys() {
  VineyardMSGBufferContext context;
  void *buffer = nullptr;
  VINEYARD_CHECK_OK(client->GetTXFreeMsgBuffer(buffer));
  VineyardMsg *msg = (VineyardMsg *)buffer;
  memset(msg, 0, sizeof(VineyardMsg));
  msg->type = VINEYARD_MSG_TEST;
  msg->test.remote_address = (uint64_t)clientMemInfo.address;
  msg->test.key = clientMemInfo.rkey;
  msg->test.len = MEM_SIZE;
  context.buffer = buffer;
  LOG(INFO) << "client address: " << clientMemInfo.address;
  LOG(INFO) << "client key: " << clientMemInfo.rkey;

  LOG(INFO) << "Send";
  client->Send(buffer, sizeof(VineyardMsg), &context);

  void *recv_context = nullptr;
  LOG(INFO) << "wait complete";
  client->GetRXCompletion(-1, &recv_context);
  LOG(INFO) << "complete";
  printf("recv context: %p\n", recv_context);
  VineyardMsg *recv_msg = (VineyardMsg *)((VineyardMSGBufferContext *)recv_context)->buffer;
  printf("msg %p\n", recv_msg);

  LOG(INFO) << "receive remote address: " << (void *)recv_msg->test.remote_address;
  LOG(INFO) << "receive key: " << recv_msg->test.key;
  LOG(INFO) << "receive length: " << recv_msg->test.len;

  serverMemInfo.address = recv_msg->test.remote_address;
  serverMemInfo.rkey = recv_msg->test.key;
  serverMemInfo.size = recv_msg->test.len;
}

void ServerExchangeKeys() {
  VineyardMSGBufferContext context;
  void *buffer = nullptr;
  VINEYARD_CHECK_OK(server->GetTXFreeMsgBuffer(buffer));
  VineyardMsg *msg = (VineyardMsg *)buffer;
  memset(msg, 0, sizeof(VineyardMsg));
  msg->type = VINEYARD_MSG_TEST;
  msg->test.remote_address = (uint64_t)serverMemInfo.address;
  msg->test.key = serverMemInfo.rkey;
  msg->test.len = MEM_SIZE;
  context.buffer = buffer;
  LOG(INFO) << "server address: " << serverMemInfo.address;
  LOG(INFO) << "server key: " << serverMemInfo.rkey;

  LOG(INFO) << "Send";
  server->Send(TEST_CLIENT_ID, buffer, sizeof(VineyardMsg), &context);

  void *recv_context = nullptr;
  LOG(INFO) << "wait complete";
  server->GetRXCompletion(-1, &recv_context);
  LOG(INFO) << "complete";
  printf("recv context: %p\n", recv_context);
  VineyardMsg *recv_msg = (VineyardMsg *)((VineyardMSGBufferContext *)recv_context)->buffer;
  printf("msg %p\n", recv_msg);

  LOG(INFO) << "receive remote address: " << (void *)recv_msg->test.remote_address;
  LOG(INFO) << "receive key: " << recv_msg->test.key;
  LOG(INFO) << "receive length: " << recv_msg->test.len;
  clientMemInfo.address = recv_msg->test.remote_address;
  clientMemInfo.rkey = recv_msg->test.key;
  clientMemInfo.size = recv_msg->test.len;
}

void StartServer() {
  VINEYARD_CHECK_OK(RDMAServer::Make(server, port));
  void *handle;
  VINEYARD_CHECK_OK(server->WaitConnect(handle));
  server->AddClient(TEST_CLIENT_ID, handle);

  void *buffer = nullptr;
  VINEYARD_CHECK_OK(server->GetRXFreeMsgBuffer(buffer));
  VineyardMSGBufferContext *bufferContext = (VineyardMSGBufferContext *)malloc(sizeof(VineyardMSGBufferContext));
  printf("recv buffer context: %p\n", bufferContext);
  bufferContext->buffer = buffer;
  server->Recv(TEST_CLIENT_ID, buffer, sizeof(VineyardMsg), bufferContext);

  void *serverMemAddr = malloc(MEM_SIZE);
  memset(serverMemAddr, 0, MEM_SIZE);
  serverMemInfo.address = (uint64_t)serverMemAddr;
  serverMemInfo.size = MEM_SIZE;
  
  VINEYARD_CHECK_OK(server->RegisterMemory(serverMemInfo));

  ServerExchangeKeys();

  HelloToClient();
}

void StartClient(std::string server_address) {
  VINEYARD_CHECK_OK(RDMAClient::Make(client, server_address, port));

  void *buffer = nullptr;
  VINEYARD_CHECK_OK(client->GetRXFreeMsgBuffer(buffer));
  VineyardMSGBufferContext *bufferContext = (VineyardMSGBufferContext *)malloc(sizeof(VineyardMSGBufferContext));
  printf("recv buffer context: %p\n", bufferContext);
  bufferContext->buffer = buffer;
  client->Recv(buffer, sizeof(VineyardMsg), bufferContext);

  VINEYARD_CHECK_OK(client->Connect());
  LOG(INFO) << "connect to server";

  void *clientMemAddr = malloc(MEM_SIZE);
  memset(clientMemAddr, 0, MEM_SIZE);
  clientMemInfo.address = (uint64_t)clientMemAddr;
  clientMemInfo.size = MEM_SIZE;
  
  VINEYARD_CHECK_OK(client->RegisterMemory(clientMemInfo));

  ClientExchangeKeys();
  HelloToServer();
}

void WriteDataToServer(std::vector<int> dataVec) {
  //TBD
}

void WriteDataToClient(std::vector<int> dataVec) {
  //TBD
}

void ReadDataFromServer(std::vector<int> &dataVec) {
  //TBD
}

void ReadDataFromClient(std::vector<int> &dataVec) {
  //TBD
}

int main(int argc, char** argv) {
  std::vector<int> clientDataVec;
  std::vector<int> serverDataVec;
  if (argc >= 2) {
    PrepareData(clientDataVec);

    std::string server_address = std::string(argv[1]);
    StartClient(server_address);

    WriteDataToServer(clientDataVec);
    ReadDataFromServer(serverDataVec);
  } else {
    PrepareData(serverDataVec);
    StartServer();
    WriteDataToClient(serverDataVec);
    ReadDataFromClient(clientDataVec);
  }

  LOG(INFO) << "Pass rdma test.";

  return 0;
}


