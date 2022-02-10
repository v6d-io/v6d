/** Copyright 2020-2021 Alibaba Group Holding Limited.

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

#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

template <typename T>
class VectorBuilder;

template <typename T>
class Vector : public vineyard::Registered<Vector<T>> {
 private:
  size_t size;
  const T* data = nullptr;

 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<Vector<T>>{new Vector<T>()});
  }

  void Construct(const ObjectMeta& meta) override {
    this->size = meta.GetKeyValue<size_t>("size");

    auto buffer = std::dynamic_pointer_cast<Blob>(meta.GetMember("buffer"));
    this->data = reinterpret_cast<const T*>(buffer->data());
  }

  Vector() : size(0), data(nullptr) {}

  Vector(const int size, const T* data) : size(size), data(data) {}

  size_t length() const { return size; }

  const T& operator[](size_t index) {
    assert(index < size);
    return data[index];
  }

  friend class VectorBuilder<T>;
};

template <typename T>
class VectorBuilder : public vineyard::ObjectBuilder {
 private:
  std::unique_ptr<BlobWriter> buffer_builder;
  std::size_t size;
  T* data;

 public:
  explicit VectorBuilder(size_t size) : size(size) {
    data = static_cast<T*>(malloc(sizeof(T) * size));
  }

  Status Build(Client& client) override {
    VINEYARD_CHECK_OK(client.CreateBlob(size * sizeof(T), buffer_builder));
    memcpy(buffer_builder->data(), data, size * sizeof(T));
    return Status::OK();
  }

  std::shared_ptr<Object> _Seal(Client& client) override {
    VINEYARD_CHECK_OK(this->Build(client));

    auto vec = std::make_shared<Vector<int>>();
    auto buffer = std::dynamic_pointer_cast<vineyard::Blob>(
        this->buffer_builder->Seal(client));
    vec->size = size;
    vec->data = reinterpret_cast<const T*>(buffer->data());

    vec->meta_.SetTypeName(vineyard::type_name<Vector<T>>());
    vec->meta_.SetNBytes(size * sizeof(T));
    vec->meta_.AddKeyValue("size", size);
    vec->meta_.AddMember("buffer", buffer);
    VINEYARD_CHECK_OK(client.CreateMetaData(vec->meta_, vec->id_));

    return vec;
  }

  T& operator[](size_t index) {
    assert(index < size);
    return data[index];
  }
};

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./custom_vector_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  auto builder = VectorBuilder<int>(3);
  builder[0] = 1;
  builder[1] = 2;
  builder[2] = 3;
  auto result = builder.Seal(client);

  auto vec =
      std::dynamic_pointer_cast<Vector<int>>(client.GetObject(result->id()));
  for (size_t index = 0; index < vec->length(); ++index) {
    std::cout << "element at " << index << " is: " << (*vec)[index]
              << std::endl;
  }

  LOG(INFO) << "Passed double array tests...";

  client.Disconnect();

  return 0;
}
