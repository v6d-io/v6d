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

#include "fuse/adaptors/arrow_ipc/serializer_registry.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"
#include "arrow/ipc/reader.h"

#include "basic/ds/arrow.h"
#include "basic/ds/arrow_utils.h"
#include "basic/ds/dataframe.h"
#include "common/util/logging.h"

namespace vineyard {
namespace fuse {

static std::shared_ptr<arrow::Buffer> view(
    const size_t estimate_size,
    std::vector<std::shared_ptr<arrow::RecordBatch>> const& batches) {
  std::shared_ptr<arrow::io::BufferOutputStream> out_stream;
  CHECK_ARROW_ERROR_AND_ASSIGN(
      out_stream,
      arrow::io::BufferOutputStream::Create(estimate_size + (4 << 20)));

  arrow::ipc::IpcWriteOptions options = arrow::ipc::IpcWriteOptions::Defaults();
  options.allow_64bit = true;

  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
#if defined(ARROW_VERSION) && ARROW_VERSION < 2000000
  CHECK_ARROW_ERROR_AND_ASSIGN(
      writer, arrow::ipc::NewFileWriter(out_stream.get(), batches[0]->schema(),
                                        options));
#else
  CHECK_ARROW_ERROR_AND_ASSIGN(
      writer, arrow::ipc::MakeFileWriter(out_stream.get(), batches[0]->schema(),
                                         options));
#endif
  for (auto const& batch : batches) {
    CHECK_ARROW_ERROR(writer->WriteRecordBatch(*batch));
  }
  CHECK_ARROW_ERROR(writer->Close());

  std::shared_ptr<arrow::Buffer> buffer;
  CHECK_ARROW_ERROR_AND_ASSIGN(buffer, out_stream->Finish());
  return buffer;
}

std::shared_ptr<arrow::Buffer> arrow_view(
    std::shared_ptr<vineyard::DataFrame>& df) {
  auto estimate_size = df->meta().MemoryUsage();
  auto batch = df->AsBatch();
  return view(estimate_size, {batch});
}

std::shared_ptr<arrow::Buffer> arrow_view(
    std::shared_ptr<vineyard::RecordBatch>& rb) {
  auto estimate_size = rb->meta().MemoryUsage();
  auto batch = rb->GetRecordBatch();
  return view(estimate_size, {batch});
}

std::shared_ptr<arrow::Buffer> arrow_view(
    std::shared_ptr<vineyard::Table>& tb) {
  auto estimate_size = tb->meta().MemoryUsage();
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  for (auto const& batch : tb->batches()) {
    batches.emplace_back(batch->GetRecordBatch());
  }
  return view(estimate_size, batches);
}

static void from_arrow_view(Client* client, std::string const& path,
                            arrow::io::RandomAccessFile* fp) {
  std::shared_ptr<arrow::ipc::RecordBatchStreamReader> reader;
  CHECK_ARROW_ERROR_AND_ASSIGN(reader,
                               arrow::ipc::RecordBatchStreamReader::Open(fp));

  std::shared_ptr<arrow::Table> table;
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;

#if defined(ARROW_VERSION) && ARROW_VERSION < 9000000
  CHECK_ARROW_ERROR(reader->ReadAll(&batches));
#else
  CHECK_ARROW_ERROR_AND_ASSIGN(batches, reader->ToRecordBatches());
#endif

  VINEYARD_CHECK_OK(RecordBatchesToTable(batches, &table));

  // build it into vineyard
  TableBuilder builder(*client, table);
  auto tb = builder.Seal(*client);
  VINEYARD_CHECK_OK(client->Persist(tb->id()));
  DLOG(INFO) << tb->meta().ToString();
  VINEYARD_CHECK_OK(client->PutName(
      tb->id(), path.substr(1, path.length() - 6 /* .arrow */ - 1)));
}

void from_arrow_view(Client* client, std::string const& path,
                     std::shared_ptr<arrow::BufferBuilder> buffer) {
  // recover table from buffer
  auto fp = std::make_shared<arrow::io::BufferReader>(buffer->data(),
                                                      buffer->length());
  from_arrow_view(client, path, fp.get());
}

void from_arrow_view(Client* client, std::string const& path,
                     std::shared_ptr<arrow::Buffer> buffer) {
  // recover table from buffer
  auto fp =
      std::make_shared<arrow::io::BufferReader>(buffer->data(), buffer->size());
  from_arrow_view(client, path, fp.get());
}

}  // namespace fuse
}  // namespace vineyard
