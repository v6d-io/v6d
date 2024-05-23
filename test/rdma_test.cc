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

using namespace vineyard; // NOLINT(build/namespaces)

constexpr int port = 9228;
std::shared_ptr<RDMAServer> server;
std::shared_ptr<RDMAClient> client;

void PrepareData(std::vector<int> &vec) {
  for (int i = 0; i < 100; i++) {
    vec.push_back(i);
  }
}

void StartServer() {
  VINEYARD_CHECK_OK(RDMAServer::Make(server, port));
  VINEYARD_CHECK_OK(server->WaitConnect());
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

  VineyardMSGBufferContext context;
  buffer = nullptr;
  VINEYARD_CHECK_OK(client->GetTXFreeMsgBuffer(buffer));
  VineyardMsg *msg = (VineyardMsg *)buffer;
  msg->type = VINEYARD_MSG_TEST;
  msg->test.remote_address = 0x123456789;
  msg->test.key = 0x987654321;
  context.buffer = buffer;

  LOG(INFO) << "Send";
  client->Send(buffer, sizeof(VineyardMsg), &context);

  void *recv_context = nullptr;
  LOG(INFO) << "wait complete";
  client->GetRXCompletion(-1, &recv_context);
  LOG(INFO) << "complete";
  printf("recv context: %p\n", recv_context);
  VineyardMsg *recv_msg = (VineyardMsg *)((VineyardMSGBufferContext *)recv_context)->buffer;
  printf("msg %p\n", recv_msg);

  LOG(INFO) << "receive remote address: " << recv_msg->test.remote_address;
  LOG(INFO) << "receive key: " << recv_msg->test.key;
  LOG(INFO) << "receive length: " << recv_msg->test.len;

  client->GetRXFreeMsgBuffer(buffer);
  printf("recv buffer context: %p\n", bufferContext);
}

void WriteDataToServer(std::vector<int> dataVec) {

}

void WriteDataToClient(std::vector<int> dataVec) {

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


