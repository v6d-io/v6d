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
#include <thread>

#include "client/client.h"
#include "common/util/logging.h"

using namespace vineyard;

int numThreads = 5;

static int count = 0;

void test(int i) {
  std::string socket = std::string(getenv("VINEYARD_IPC_SOCKET"));
  Client client;
  client.Connect(socket);

  bool result;
  std::string actural_key_of_lock;

  LOG(INFO) << "Thread: " << i << " try to acquire lock: test";
  client.TryAcquireLock("test", result, actural_key_of_lock);
  LOG(INFO) << "Thread: " << i
            << " acquire Lock: " << (result == true ? "success" : "fail")
            << ", key is :" + actural_key_of_lock;

  if (result) {
    count++;
    LOG(INFO) << "count: " << count;

    sleep(3);

    LOG(INFO) << "Thread: " << i << " try to release lock: test";
    client.TryReleaseLock(actural_key_of_lock, result);
    LOG(INFO) << "Thread: " << i
              << " release Lock: " << (result == true ? "success" : "fail");
  }
}

int main() {
  std::thread threads[numThreads];
  for (int i = 0; i < numThreads; i++) {
    threads[i] = std::thread(test, i);
  }

  for (int i = 0; i < numThreads; i++) {
    threads[i].join();
  }

  return 0;
}