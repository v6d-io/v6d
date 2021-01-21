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

#include "basic/stream/byte_stream.h"
#include "client/client.h"
#include "client/ds/object_meta.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./vector_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

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

  // when stream fail
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

  // when stream contains empty chunks

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

  LOG(INFO) << "Passed stream tests...";

  client.Disconnect();

  return 0;
}
