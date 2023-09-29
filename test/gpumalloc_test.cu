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

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;

__global__ void printKernel(const uint8_t* data) {
  printf("GPU data is: %s\n", reinterpret_cast<const char*>(data));
}

auto create_gpu_objects = [](Client& client,
                             std::vector<std::string> const& data, bool do_seal,
                             std::set<ObjectID>& object_ids) {
  for (size_t i = 0; i < data.size(); i++) {
    ObjectID object_id;
    Payload object;
    std::shared_ptr<MutableBuffer> buffer = nullptr;
    VINEYARD_CHECK_OK(
        client.CreateGPUBuffer(data[i].size(), object_id, object, buffer));
    object_ids.emplace(object_id);
    CHECK(buffer != nullptr);
    CHECK(!buffer->is_cpu());
    CHECK(buffer->is_mutable());

    {
      CUDABufferMirror mirror(*buffer, false);
      memcpy(mirror.mutable_data(), data[i].c_str(), data[i].size());
    }
    printKernel<<<1, 1>>>(buffer->data());
    cudaDeviceSynchronize();
  }
  return Status::OK();
};

auto get_gpu_objects = [](Client& client, std::vector<std::string> const& data,
                          std::set<ObjectID>& object_ids, bool check_seal) {
  std::map<ObjectID, std::shared_ptr<Buffer>> buffers;
  VINEYARD_CHECK_OK(client.GetGPUBuffers(object_ids, true, buffers));
  for (auto& item : buffers) {
    auto &buffer = item.second;
    CHECK(!buffer->is_cpu());
    CHECK(!buffer->is_mutable());

    printKernel<<<1, 1>>>(buffer->data());
    cudaDeviceSynchronize();

    {
      CUDABufferMirror mirror(*buffer, true);
      printf("CPU data from GPU is: %s\n",
             reinterpret_cast<const char*>(mirror.data()));
    }
  }
  return Status::OK();
};

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./gpumalloc_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  std::set<ObjectID> object_ids;
  std::vector<std::string> data;
  data.emplace_back("gpu buffer create success.");
  data.emplace_back("hello world!");


  LOG(INFO) << "Create GPU Object tests\n";
  VINEYARD_CHECK_OK(create_gpu_objects(client, data, false, object_ids));
  LOG(INFO) << "Passed GPU create test...";

  LOG(INFO) << "Get GPU Object tests\n";
  VINEYARD_CHECK_OK(get_gpu_objects(client, data, object_ids, false));
  LOG(INFO) << "Passed GPU get test...";

  client.Disconnect();

  LOG(INFO) << "Passed GPU test...";

  return 0;
}
