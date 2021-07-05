/** Copyright 2020 Alibaba Group Holding Limited.

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
#include <unordered_map>

#include "arrow/status.h"
#include "arrow/util/io_util.h"
#include "arrow/util/logging.h"
#include "glog/logging.h"

#include "basic/ds/dataframe.h"
#include "basic/stream/object_stream.h"
#include "client/client.h"
#include "client/ds/object_meta.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./object_stream_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  ObjectID stream_id = InvalidObjectID();
  {
    ObjectStreamBuilder builder(client);
    builder.SetParams(std::unordered_map<std::string, std::string>{
        {"kind", "test"}, {"test_name", "stream_test"}});
    auto object_stream =
        std::dynamic_pointer_cast<ObjectStream>(builder.Seal(client));
    stream_id = object_stream->id();
    CHECK(stream_id != InvalidObjectID());
  }

  std::thread recv_thrd([&]() {
    Client reader_client;
    VINEYARD_CHECK_OK(reader_client.Connect(ipc_socket));

    auto object_stream = reader_client.GetObject<ObjectStream>(stream_id);
    CHECK(object_stream != nullptr);

    std::unique_ptr<ObjectStreamReader> reader;
    VINEYARD_CHECK_OK(object_stream->OpenReader(reader_client, reader));

    std::unique_ptr<ObjectStreamReader> failed_reader;
    auto status1 = object_stream->OpenReader(reader_client, failed_reader);
    CHECK(status1.IsStreamOpened());

    for (size_t i = 1; true; ++i) {
      ObjectID next_object;
      auto status = reader->GetNext(next_object);
      if (status.ok()) {
        auto df = reader_client.GetObject<DataFrame>(next_object);
        auto column_b =
            std::dynamic_pointer_cast<Tensor<int64_t>>(df->Column("b"));
        CHECK_EQ(column_b->shape()[0], 100 * i);
      } else {
        CHECK(status.IsStreamDrained());
        break;
      }
    }
  });

  std::thread send_thrd([&]() {
    Client writer_client;
    VINEYARD_CHECK_OK(writer_client.Connect(ipc_socket));

    auto object_stream = writer_client.GetObject<ObjectStream>(stream_id);
    CHECK(object_stream != nullptr);

    std::unique_ptr<ObjectStreamWriter> writer;
    VINEYARD_CHECK_OK(object_stream->OpenWriter(writer_client, writer));

    std::unique_ptr<ObjectStreamWriter> failed_writer;
    auto status1 = object_stream->OpenWriter(writer_client, failed_writer);
    CHECK(status1.IsStreamOpened());

    CHECK(writer != nullptr);
    for (int64_t idx = 1; idx <= 11; ++idx) {
      DataFrameBuilder builder(client);

      auto tb = std::make_shared<TensorBuilder<int64_t>>(
          client, std::vector<int64_t>{idx * 100});
      builder.AddColumn("b", tb);

      auto column_b = std::dynamic_pointer_cast<TensorBuilder<int64_t>>(
          builder.Column("b"));
      auto data = column_b->data();
      for (int64_t i = 0; i < idx * 100; ++i) {
        data[i] = i * i * i;
      }
      auto seal_df = builder.Seal(client);
      VINEYARD_CHECK_OK(client.Persist(seal_df->id()));

      writer->PutNext(seal_df->id());
      sleep(1);
    }
    VINEYARD_CHECK_OK(writer->Finish());

    ObjectID po;
    VINEYARD_CHECK_OK(writer->PersistToObject(po));
    CHECK(po != stream_id);
  });

  send_thrd.join();
  recv_thrd.join();

  // when stream fail
  {
    ObjectStreamBuilder builder(client);
    builder.SetParams(std::unordered_map<std::string, std::string>{
        {"kind", "test"}, {"test_name", "stream_test"}});
    auto object_stream =
        std::dynamic_pointer_cast<ObjectStream>(builder.Seal(client));
    stream_id = object_stream->id();
    CHECK(stream_id != InvalidObjectID());
  }

  auto failed_object_stream = client.GetObject<ObjectStream>(stream_id);

  std::unique_ptr<ObjectStreamReader> reader = nullptr;
  std::unique_ptr<ObjectStreamWriter> writer = nullptr;
  VINEYARD_CHECK_OK(failed_object_stream->OpenReader(client, reader));
  VINEYARD_CHECK_OK(failed_object_stream->OpenWriter(client, writer));
  CHECK(reader != nullptr);
  CHECK(writer != nullptr);
  VINEYARD_CHECK_OK(writer->Abort());

  ObjectID next_object;
  auto status = reader->GetNext(next_object);
  CHECK(status.IsStreamFailed());

  LOG(INFO) << "Passed object stream tests...";

  client.Disconnect();

  return 0;
}
