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

#include "basic/ds/dataframe.h"
#include "basic/ds/tensor.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"
#include "common/memory/gpu/unified_memory.h"

using namespace vineyard;  

__global__ void printKernel(void *data) {
  printf("%s GPU\n", data);
}

auto create_gpu_objects = [](Client& client, std::vector<std::string> const& data,
                            bool do_seal, std::vector<ObjectID> &oids) {
  for(size_t i = 0; i < data.size(); i++){
    Payload object;
    std::shared_ptr<GPUUnifiedAddress> gua = nullptr;
    RETURN_ON_ERROR(client.CreateGPUBuffer(data[i].size(), oids[i], object, gua));
    void *ptr = nullptr;
    GUAError_t res;
    res = gua->ManagedMalloc(data[i].size(), &ptr);
    LOG(INFO) << "ManagedMalloc Error: " << guaErrorToString(res) << std::endl;
    memcpy(gua->getCPUMemPtr(), data[i].c_str(), data[i].size());
    res = gua->syncFromCPU();
    LOG(INFO) << "syncFromCPU Error: " << guaErrorToString(res) << std::endl;
    LOG(INFO) << "create buffer:" << std::endl;
    printKernel<<<1, 1>>>(gua->getGPUMemPtr());
    cudaDeviceSynchronize();
  }
  return Status::OK();
};

auto get_gpu_objects = [](Client& client,std::vector<std::string> const& data,
                          std::vector<ObjectID>& _oids, bool check_seal) {
    std::map<ObjectID, GPUUnifiedAddress> GUAs;
    std::set<ObjectID> oids;
    for(auto oid: _oids) {
      oids.emplace(oid);
    }
    auto status = client.GetGPUBuffers(oids, true, GUAs);
    for(auto oid : oids){
      LOG(INFO) << "get buffer, oid: " << oid  << std::endl;
      void *ptr = nullptr;
      GUAs[oid].GPUData(&ptr);
      printKernel<<<1, 1>>>(ptr);
      cudaDeviceSynchronize();
    } 
    return Status::OK();
  };

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./global_object_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);
  //connection test
  Client client;
  VINEYARD_CHECK_OK(client.BasicIPCClient::Open(ipc_socket, StoreType::kDefault));
  LOG(INFO) << "Connected to IPCServer: type kGPU" << ipc_socket;
  std::vector<std::string> data;
  data.emplace_back("Create Success.");
  data.emplace_back("hello world!");
  std::vector<ObjectID> oids(data.size());
  LOG(INFO) << "Create GPU Object tests\n"; 
  create_gpu_objects(client, data, false, oids);
  LOG(INFO) << "Passed GPU create test...";
  LOG(INFO) << "Get GPU Object tests\n"; 
  get_gpu_objects(client, data, oids, false);
  LOG(INFO) << "Passed GPU get test...";
  client.Disconnect();

  //create/get test
  LOG(INFO) << "Passed GPU  test...";

  return 0;
}
