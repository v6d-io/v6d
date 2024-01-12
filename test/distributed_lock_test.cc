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

#include <cstdlib>

#include "client/client.h"
#include "common/util/logging.h"

using namespace vineyard;

int main() {
  std::string socket = std::string(getenv("VINEYARD_IPC_SOCKET"));
  Client client;
  client.Connect(socket);

  bool result;
  client.TryAcquireLock("/test", result);
  LOG(INFO) << "Acquire Lock: " << result;
  client.TryReleaseLock("/test", result);
  LOG(INFO) << "Release Lock: " << result;
  client.TryAcquireLock("/test", result);
  LOG(INFO) << "Acquire Lock: " << result;
  client.TryReleaseLock("/test", result);
  LOG(INFO) << "Release Lock: " << result;
  client.TryAcquireLock("/test", result);
  LOG(INFO) << "Acquire Lock: " << result;
  client.TryReleaseLock("/test", result);
  LOG(INFO) << "Release Lock: " << result;
  client.TryAcquireLock("/test", result);
  LOG(INFO) << "Acquire Lock: " << result;
  client.TryReleaseLock("/test", result);
  LOG(INFO) << "Release Lock: " << result;

  return 0;
}