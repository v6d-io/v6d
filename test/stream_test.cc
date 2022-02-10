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
#include <unordered_map>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/stream/byte_stream.h"
#include "basic/stream/dataframe_stream.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

void testByteStream(Client& client, std::string const& ipc_socket) {
  ObjectID stream_id = InvalidObjectID();
  {
    ByteStreamBuilder builder(client);
    builder.SetParams(std::unordered_map<std::string, std::string>{
        {"kind", "test"}, {"test_name", "stream_test"}});
    auto bstream = std::dynamic_pointer_cast<ByteStream>(builder.Seal(client));
    stream_id = bstream->id();
    CHECK(stream_id != InvalidObjectID());
  }

  size_t send_chunks = 0, recv_chunks = 0;
  std::vector<size_t> send_chunks_size, recv_chunks_size;

  std::thread recv_thrd([&]() {
    Client reader_client;
    VINEYARD_CHECK_OK(reader_client.Connect(ipc_socket));

    auto byte_stream = reader_client.GetObject<ByteStream>(stream_id);
    CHECK(byte_stream != nullptr);

    std::unique_ptr<ByteStreamReader> reader;
    VINEYARD_CHECK_OK(byte_stream->OpenReader(reader_client, reader));

    std::unique_ptr<ByteStreamReader> failed_reader;
    auto status1 = byte_stream->OpenReader(reader_client, failed_reader);
    CHECK(status1.IsStreamOpened());

    while (true) {
      std::unique_ptr<arrow::Buffer> buffer = nullptr;
      auto status = reader->GetNext(buffer);
      if (status.ok()) {
        CHECK(buffer != nullptr);
        recv_chunks += 1;
        recv_chunks_size.emplace_back(buffer->size());
      } else {
        CHECK(status.IsStreamDrained());
        break;
      }
    }
  });

  std::thread send_thrd([&]() {
    Client writer_client;
    VINEYARD_CHECK_OK(writer_client.Connect(ipc_socket));

    auto byte_stream = writer_client.GetObject<ByteStream>(stream_id);
    CHECK(byte_stream != nullptr);

    std::unique_ptr<ByteStreamWriter> writer;
    VINEYARD_CHECK_OK(byte_stream->OpenWriter(writer_client, writer));

    std::unique_ptr<ByteStreamWriter> failed_writer;
    auto status1 = byte_stream->OpenWriter(writer_client, failed_writer);
    CHECK(status1.IsStreamOpened());

    CHECK(writer != nullptr);
    for (size_t idx = 1; idx <= 11; ++idx) {
      std::unique_ptr<arrow::MutableBuffer> buffer = nullptr;
      VINEYARD_CHECK_OK(writer->GetNext(1 << idx, buffer));
      CHECK(buffer != nullptr);
      send_chunks += 1;
      send_chunks_size.emplace_back(1 << idx);
      sleep(1);
    }
    VINEYARD_CHECK_OK(writer->Finish());
  });

  send_thrd.join();
  recv_thrd.join();

  CHECK_EQ(send_chunks, recv_chunks);
  CHECK_EQ(send_chunks_size.size(), recv_chunks_size.size());
  for (size_t idx = 0; idx < send_chunks_size.size(); ++idx) {
    CHECK_EQ(send_chunks_size[idx], recv_chunks_size[idx]);
  }
}

void testByteStreamFailed(Client& client, std::string const& ipc_socket) {
  ObjectID stream_id = InvalidObjectID();
  {
    ByteStreamBuilder builder(client);
    builder.SetParams(std::unordered_map<std::string, std::string>{
        {"kind", "test"}, {"test_name", "stream_test"}});
    auto bstream = std::dynamic_pointer_cast<ByteStream>(builder.Seal(client));
    stream_id = bstream->id();
    CHECK(stream_id != InvalidObjectID());
  }

  auto failed_byte_stream = client.GetObject<ByteStream>(stream_id);

  std::unique_ptr<ByteStreamReader> reader = nullptr;
  std::unique_ptr<ByteStreamWriter> writer = nullptr;
  VINEYARD_CHECK_OK(failed_byte_stream->OpenReader(client, reader));
  VINEYARD_CHECK_OK(failed_byte_stream->OpenWriter(client, writer));
  CHECK(reader != nullptr);
  CHECK(writer != nullptr);
  VINEYARD_CHECK_OK(writer->Abort());

  std::unique_ptr<arrow::Buffer> buffer = nullptr;
  auto status = reader->GetNext(buffer);
  CHECK(status.IsStreamFailed());
}

void testEmptyStream(Client& client, std::string const& ipc_socket) {
  ObjectID stream_id = InvalidObjectID();
  {
    ByteStreamBuilder builder(client);
    builder.SetParams(std::unordered_map<std::string, std::string>{
        {"kind", "test"}, {"test_name", "stream_test"}});
    auto bstream = std::dynamic_pointer_cast<ByteStream>(builder.Seal(client));
    stream_id = bstream->id();
    CHECK(stream_id != InvalidObjectID());
  }

  auto empty_byte_stream = client.GetObject<ByteStream>(stream_id);

  std::unique_ptr<ByteStreamReader> empty_reader = nullptr;
  std::unique_ptr<ByteStreamWriter> empty_writer = nullptr;
  VINEYARD_CHECK_OK(empty_byte_stream->OpenReader(client, empty_reader));
  VINEYARD_CHECK_OK(empty_byte_stream->OpenWriter(client, empty_writer));
  CHECK(empty_reader != nullptr);
  CHECK(empty_writer != nullptr);

  {
    // write empty chunk
    std::unique_ptr<arrow::MutableBuffer> buffer = nullptr;
    VINEYARD_CHECK_OK(empty_writer->GetNext(0, buffer));

    CHECK(buffer != nullptr);
    CHECK_EQ(buffer->size(), 0);
    VINEYARD_CHECK_OK(empty_writer->Finish());
  }

  {
    // read the empty chunk
    std::unique_ptr<arrow::Buffer> buffer = nullptr;
    VINEYARD_CHECK_OK(empty_reader->GetNext(buffer));
    CHECK(buffer != nullptr);
    CHECK_EQ(buffer->size(), 0);

    CHECK(empty_reader->GetNext(buffer).IsStreamDrained());
  }
}

void testDataframeStream(Client& client, std::string const& ipc_socket) {
  ObjectID stream_id = InvalidObjectID();
  {
    DataframeStreamBuilder builder(client);
    builder.SetParams(std::unordered_map<std::string, std::string>{
        {"kind", "test"}, {"test_name", "stream_test"}});
    auto bstream =
        std::dynamic_pointer_cast<DataframeStream>(builder.Seal(client));
    stream_id = bstream->id();
    CHECK(stream_id != InvalidObjectID());
  }

  // make a batch
  std::shared_ptr<arrow::RecordBatch> batch;
  {
    arrow::LargeStringBuilder key_builder;
    arrow::Int64Builder value_builder;
    arrow::StringBuilder string_builder;

    auto sub_builder = std::make_shared<arrow::Int64Builder>();
    arrow::LargeListBuilder list_builder(arrow::default_memory_pool(),
                                         sub_builder);

    std::shared_ptr<arrow::Array> array1;
    std::shared_ptr<arrow::Array> array2;
    std::shared_ptr<arrow::Array> array3;
    std::shared_ptr<arrow::Array> array4;

    for (int64_t j = 0; j < 100; j++) {
      CHECK_ARROW_ERROR(key_builder.AppendValues({std::to_string(j)}));
      CHECK_ARROW_ERROR(value_builder.AppendValues({j}));
      CHECK_ARROW_ERROR(string_builder.AppendValues({std::to_string(j * j)}));
      CHECK_ARROW_ERROR(sub_builder->AppendValues({j, j + 1, j + 2}));
      CHECK_ARROW_ERROR(list_builder.Append(true));
    }
    CHECK_ARROW_ERROR(key_builder.Finish(&array1));
    CHECK_ARROW_ERROR(value_builder.Finish(&array2));
    CHECK_ARROW_ERROR(string_builder.Finish(&array3));
    CHECK_ARROW_ERROR(list_builder.Finish(&array4));

    auto arrowSchema = arrow::schema(
        {std::make_shared<arrow::Field>("f1", arrow::large_utf8()),
         std::make_shared<arrow::Field>("f2", arrow::int64()),
         std::make_shared<arrow::Field>("f3", arrow::utf8()),
         std::make_shared<arrow::Field>("f4",
                                        arrow::large_list(arrow::int64()))});
    batch = arrow::RecordBatch::Make(arrowSchema, array1->length(),
                                     {array1, array2, array3, array4});
  }

  size_t send_chunks = 0, recv_chunks = 0;

  std::thread recv_thrd([&]() {
    Client reader_client;
    VINEYARD_CHECK_OK(reader_client.Connect(ipc_socket));

    auto dataframe_stream = reader_client.GetObject<DataframeStream>(stream_id);
    CHECK(dataframe_stream != nullptr);

    std::unique_ptr<DataframeStreamReader> reader;
    VINEYARD_CHECK_OK(dataframe_stream->OpenReader(reader_client, reader));

    std::unique_ptr<DataframeStreamReader> failed_reader;
    auto status1 = dataframe_stream->OpenReader(reader_client, failed_reader);
    CHECK(status1.IsStreamOpened());

    while (true) {
      std::shared_ptr<arrow::RecordBatch> load_batch;
      auto status = reader->ReadBatch(load_batch);
      if (status.ok()) {
        CHECK(load_batch != nullptr);
        recv_chunks += 1;
      } else {
        CHECK(status.IsStreamDrained());
        break;
      }
    }
  });

  std::thread send_thrd([&]() {
    Client writer_client;
    VINEYARD_CHECK_OK(writer_client.Connect(ipc_socket));

    auto dataframe_stream = writer_client.GetObject<DataframeStream>(stream_id);
    CHECK(dataframe_stream != nullptr);

    std::unique_ptr<DataframeStreamWriter> writer;
    VINEYARD_CHECK_OK(dataframe_stream->OpenWriter(writer_client, writer));

    std::unique_ptr<DataframeStreamWriter> failed_writer;
    auto status1 = dataframe_stream->OpenWriter(writer_client, failed_writer);
    CHECK(status1.IsStreamOpened());

    CHECK(writer != nullptr);
    for (size_t idx = 1; idx <= 11; ++idx) {
      VINEYARD_CHECK_OK(writer->WriteBatch(batch));
      send_chunks += 1;
      sleep(1);
    }
    VINEYARD_CHECK_OK(writer->Finish());
  });

  send_thrd.join();
  recv_thrd.join();

  CHECK_EQ(send_chunks, recv_chunks);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./stream_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  testByteStream(client, ipc_socket);
  LOG(INFO) << "Passed bytestream test...";

  testByteStreamFailed(client, ipc_socket);
  LOG(INFO) << "Passed failed bytestream test...";

  testEmptyStream(client, ipc_socket);
  LOG(INFO) << "Passed empty bytestream test...";

  testDataframeStream(client, ipc_socket);
  LOG(INFO) << "Passed empty dataframe test...";

  LOG(INFO) << "Passed stream tests...";

  client.Disconnect();

  return 0;
}
