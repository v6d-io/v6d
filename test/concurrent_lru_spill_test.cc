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
#include <vector>

#include "basic/ds/array.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "client/rpc_client.h"
#include "common/util/logging.h"
#include "common/util/status.h"
#include "common/util/uuid.h"

using namespace vineyard;  // NOLINT
using namespace std;       // NOLINT

constexpr double delta = 1E-10;

template <typename T>
ObjectID GetObjectID(const std::shared_ptr<Array<T>>& sealed_array) {
  return ObjectIDFromString(sealed_array->meta()
                                .MetaData()["buffer_"]["id"]
                                .template get_ref<std::string const&>());
}

template <typename T>
vector<T> InitArray(int size, std::function<T(int n)> init_func) {
  std::vector<T> array;
  array.resize(size);
  for (int i = 0; i < size; i++) {
    array[i] = init_func(i);
  }
  return array;
}

void ConcurrentPutWithClient(std::string ipc_socket) {
  const int array_size = 250;
  const int num_objects = 500;
  const int num_threads = 10;
  auto create_and_seal_array = [&](Client& c) {
    auto double_array = InitArray<double>(array_size, [](int i) { return i; });
    ArrayBuilder<double> builder(c, double_array);
    auto sealed_array =
        std::dynamic_pointer_cast<Array<double>>(builder.Seal(c));
  };

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  auto worker = [&]() {
    Client client;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));
    for (int i = 0; i < num_objects; ++i) {
      create_and_seal_array(client);
    }
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  VINEYARD_CHECK_OK(client.Clear());
}

void ConcurrentGetWithClient(std::string ipc_socket) {
  const int array_size = 250;
  const int num_objects = 500;
  const int num_threads = 10;
  auto create_and_seal_array = [&](Client& c) {
    auto double_array = InitArray<double>(array_size, [](int i) { return i; });
    ArrayBuilder<double> builder(c, double_array);
    auto sealed_array =
        std::dynamic_pointer_cast<Array<double>>(builder.Seal(c));
    return GetObjectID(sealed_array);
  };
  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  std::vector<ObjectID> objects;
  for (int i = 0; i < num_objects * num_threads; i++) {
    objects.push_back(create_and_seal_array(client));
  }

  auto worker = [&](std::vector<ObjectID> ids) {
    Client client;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));
    for (int i = 0; i < num_objects * num_threads; ++i) {
      std::shared_ptr<Object> object;
      ObjectID id = ids[i];
      VINEYARD_CHECK_OK(client.GetObject(id, object));
    }
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker, objects);
  }

  for (auto& thread : threads) {
    thread.join();
  }
  VINEYARD_CHECK_OK(client.Clear());
}

void ConcurrentPutWithRPCClient(std::string rpc_endpoint) {
  const int array_size = 250;
  const int num_objects = 500;
  const int num_threads = 10;

  auto double_array = InitArray<double>(array_size, [](int i) { return i; });
  auto create_remote_blob = [&](RPCClient& c) {
    auto remote_blob_writer = std::make_shared<RemoteBlobWriter>(
        double_array.size() * sizeof(double));
    std::memcpy(remote_blob_writer->data(), double_array.data(),
                double_array.size() * sizeof(double));
    ObjectMeta meta;
    VINEYARD_CHECK_OK(c.CreateRemoteBlob(remote_blob_writer, meta));
  };

  RPCClient rpc_client;
  VINEYARD_CHECK_OK(rpc_client.Connect(rpc_endpoint));

  auto worker = [&]() {
    RPCClient rpc_client;
    VINEYARD_CHECK_OK(rpc_client.Connect(rpc_endpoint));
    for (int i = 0; i < num_objects; ++i) {
      create_remote_blob(rpc_client);
    }
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  VINEYARD_CHECK_OK(rpc_client.Clear());
}

void ConcurrentGetWithRPCClient(std::string rpc_endpoint) {
  const int array_size = 250;
  const int num_objects = 500;
  const int num_threads = 10;

  auto double_array = InitArray<double>(array_size, [](int i) { return i; });
  auto create_remote_blob = [&](RPCClient& c) {
    auto remote_blob_writer = std::make_shared<RemoteBlobWriter>(
        double_array.size() * sizeof(double));
    std::memcpy(remote_blob_writer->data(), double_array.data(),
                double_array.size() * sizeof(double));
    ObjectMeta meta;
    VINEYARD_CHECK_OK(c.CreateRemoteBlob(remote_blob_writer, meta));
    return meta.GetId();
  };

  RPCClient rpc_client;
  VINEYARD_CHECK_OK(rpc_client.Connect(rpc_endpoint));

  std::vector<ObjectID> objects;
  for (int i = 0; i < num_objects * num_threads; i++) {
    objects.push_back(create_remote_blob(rpc_client));
  }
  auto worker = [&](std::vector<ObjectID> ids) {
    RPCClient rpc_client;
    VINEYARD_CHECK_OK(rpc_client.Connect(rpc_endpoint));
    for (int i = 0; i < num_objects * num_threads; ++i) {
      ObjectID id = ids[i];
      rpc_client.GetObject(id);
    }
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker, objects);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  VINEYARD_CHECK_OK(rpc_client.Clear());
}

void ConcurrentGetAndPut(std::string ipc_socket, std::string rpc_endpoint) {
  const int num_threads = 20;
  const int num_objects = 500;
  const int array_size = 250;
  const int initial_objects = 100;

  std::vector<ObjectID> object_ids;
  std::mutex object_ids_mutex;

  auto create_and_seal_array = [&](Client& c) {
    try {
      auto double_array = InitArray<double>(
          array_size, [](int i) { return static_cast<double>(i); });
      ArrayBuilder<double> builder(c, double_array);
      auto sealed_array =
          std::dynamic_pointer_cast<Array<double>>(builder.Seal(c));
      return GetObjectID(sealed_array);
    } catch (std::exception& e) {
      LOG(ERROR) << e.what();
      return InvalidObjectID();
    }
  };

  auto create_remote_blob = [&](RPCClient& c) {
    try {
      auto double_array = InitArray<double>(
          array_size, [](int i) { return static_cast<double>(i); });
      auto remote_blob_writer = std::make_shared<RemoteBlobWriter>(
          double_array.size() * sizeof(double));
      std::memcpy(remote_blob_writer->data(), double_array.data(),
                  double_array.size() * sizeof(double));
      ObjectMeta blob_meta;
      VINEYARD_CHECK_OK(c.CreateRemoteBlob(remote_blob_writer, blob_meta));
      return blob_meta.GetId();
    } catch (std::exception& e) {
      LOG(ERROR) << e.what();
      return InvalidObjectID();
    }
  };

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  for (int i = 0; i < initial_objects; i++) {
    object_ids.push_back(create_and_seal_array(client));
  }

  auto worker = [&](int id, std::vector<ObjectID> object_ids) {
    Client client;
    VINEYARD_CHECK_OK(client.Connect(ipc_socket));
    RPCClient rpc_client;
    VINEYARD_CHECK_OK(rpc_client.Connect(rpc_endpoint));
    std::vector<ObjectID> ids;

    for (int i = 0; i < num_objects; ++i) {
      if (id % 2 == 0) {
        if (i % 3 == 0) {
          ObjectID new_id = create_and_seal_array(client);
          if (new_id != InvalidObjectID()) {
            VINEYARD_DISCARD(client.Release(new_id));
          }

          {
            std::lock_guard<std::mutex> lock(object_ids_mutex);
            object_ids.push_back(new_id);
          }
        } else {
          ObjectID id_to_get;
          {
            std::lock_guard<std::mutex> lock(object_ids_mutex);
            id_to_get = object_ids[rand() % object_ids.size()];
          }
          if (id_to_get != InvalidObjectID()) {
            std::shared_ptr<Object> object = client.GetObject(id_to_get);
            VINEYARD_DISCARD(client.Release(object->id()));
          }
        }
      } else {
        if (i % 3 == 0) {
          ObjectID new_id = create_remote_blob(rpc_client);
          {
            std::lock_guard<std::mutex> lock(object_ids_mutex);
            object_ids.push_back(new_id);
          }
        } else {
          ObjectID id_to_get;
          {
            std::lock_guard<std::mutex> lock(object_ids_mutex);
            id_to_get = object_ids[rand() % object_ids.size()];
          }
          if (id_to_get != InvalidObjectID()) {
            std::shared_ptr<Object> object = rpc_client.GetObject(id_to_get);
          }
        }
      }
    }
    client.Disconnect();
    rpc_client.Disconnect();
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(worker, i, object_ids);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  VINEYARD_CHECK_OK(client.Clear());
  client.Disconnect();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage ./concurrent_lru_spill_test <ipc_socket> <rpc_endpoint>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);
  std::string rpc_endpoint = std::string(argv[2]);

  LOG(INFO) << "Start concurrent put test with IPCClient ...";
  ConcurrentPutWithClient(ipc_socket);
  LOG(INFO) << "Passed concurrent put test with IPCClient";

  LOG(INFO) << "Start concurrent get test with IPCClient ...";
  ConcurrentGetWithClient(ipc_socket);
  LOG(INFO) << "Passed concurrent get test with IPCClient";

  LOG(INFO) << "Start concurrent put test with RPCClient ...";
  ConcurrentPutWithRPCClient(rpc_endpoint);
  LOG(INFO) << "Passed concurrent put test with RPCClient";

  LOG(INFO) << "Start concurrent get test with RPCClient ...";
  ConcurrentGetWithRPCClient(rpc_endpoint);
  LOG(INFO) << "Passed concurrent get test with RPCClient";

  LOG(INFO) << "Start concurrent get and put test ...";
  ConcurrentGetAndPut(ipc_socket, rpc_endpoint);
  LOG(INFO) << "Passed concurrent get and put test";

  LOG(INFO) << "Passed concurrent lru spill tests ...";
}
