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

#include <iostream>
#include <string>

#include "arrow/api.h"
#include "arrow/csv/api.h"
#include "arrow/io/api.h"
#include "arrow/util/config.h"

#include "basic/stream/byte_stream.h"
#include "basic/stream/dataframe_stream.h"
#include "basic/stream/parallel_stream.h"
#include "client/client.h"
#include "io/io/local_io_adaptor.h"
#include "io/io/utils.h"

using namespace vineyard;  // NOLINT(build/namespaces)

void ParseTable(std::shared_ptr<arrow::Table>* table,
                std::unique_ptr<arrow::Buffer>& buffer, char delimiter,
                bool header_row, std::vector<std::string> col_names) {
  // FIXME IF NO NEED TO COPY
  std::shared_ptr<arrow::Buffer> copied_buffer;
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
  CHECK_ARROW_ERROR(buffer->Copy(0, buffer->size(), &copied_buffer));
#else
  CHECK_ARROW_ERROR_AND_ASSIGN(copied_buffer,
                               buffer->CopySlice(0, buffer->size()));
#endif
  LOG(INFO) << "copied: buffer size = " << copied_buffer->size();
  auto buffer_reader = std::make_shared<arrow::io::BufferReader>(copied_buffer);

  std::shared_ptr<arrow::io::InputStream> input =
      arrow::io::RandomAccessFile::GetStream(buffer_reader, 0,
                                             copied_buffer->size());

  arrow::MemoryPool* pool = arrow::default_memory_pool();

  auto read_options = arrow::csv::ReadOptions::Defaults();
  auto parse_options = arrow::csv::ParseOptions::Defaults();
  auto convert_options = arrow::csv::ConvertOptions::Defaults();

  if (col_names.size() > 0) {
    read_options.column_names = col_names;
  } else {
    read_options.autogenerate_column_names = (!header_row);
  }
  parse_options.delimiter = delimiter;

  std::shared_ptr<arrow::csv::TableReader> reader;
  CHECK_ARROW_ERROR_AND_ASSIGN(
      reader, arrow::csv::TableReader::Make(pool, input, read_options,
                                            parse_options, convert_options));

  CHECK_ARROW_ERROR_AND_ASSIGN(*table, reader->Read());

  CHECK_ARROW_ERROR((*table)->Validate());

  VLOG(2) << (*table)->num_rows() << " rows, " << (*table)->num_columns()
          << " columns";
  VLOG(2) << (*table)->schema()->ToString();

  if (col_names.size() == 0) {
    col_names = (*table)->ColumnNames();
  }
}

int main(int argc, const char** argv) {
  if (argc < 3) {
    printf(
        "usage ./parallel_dataframe_parser <ipc_socket> <stream_id> "
        "<proc_index>\n");
    return 1;
  }

  std::string ipc_socket = std::string(argv[1]);
  ObjectID stream_id = VYObjectIDFromString(argv[2]);
  int proc_index = std::stoi(argv[3]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  auto s =
      std::dynamic_pointer_cast<ParallelStream>(client.GetObject(stream_id));
  LOG(INFO) << "Got parallel stream " << s->id();

  auto ls = s->GetStream<ByteStream>(proc_index);
  LOG(INFO) << "Got byte stream " << ls->id() << " at " << proc_index;

  auto params = ls->GetParams();
  std::string header_line = params["header_line"];
  std::string delimiter = params["delimiter"];
  bool header_row = false;
  std::vector<std::string> col_names;
  if (header_line != "") {
    header_row = true;
    ::boost::split(col_names, header_line, ::boost::is_any_of(delimiter));
  }

  DataframeStreamBuilder dfbuilder(client);
  auto bs = std::dynamic_pointer_cast<DataframeStream>(dfbuilder.Seal(client));
  VINEYARD_CHECK_OK(client.Persist(bs->id()));
  LOG(INFO) << "Created dataframe stream " << bs->id() << " at " << proc_index;
  ReportStatus("return", VYObjectIDToString(bs->id()));

  auto reader = ls->OpenReader(client);
  auto writer = bs->OpenWriter(client);

  while (true) {
    std::unique_ptr<arrow::Buffer> buffer;
    auto status = reader->GetNext(buffer);
    if (status.ok()) {
      LOG(INFO) << "consumer: buffer size = " << buffer->size();
      std::shared_ptr<arrow::Table> table;
      if (delimiter == "\t") {
        ParseTable(&table, buffer, '\t', header_row, col_names);
      } else {
        ParseTable(&table, buffer, delimiter[0], header_row, col_names);
      }
      auto st = writer->WriteTable(table);
      if (!st.ok()) {
        ReportStatus("error", st.ToString());
      }
      VINEYARD_CHECK_OK(st);
    } else {
      if (status.IsStreamDrained()) {
        LOG(INFO) << "Stream drained";
        break;
      }
    }
  }
  VINEYARD_CHECK_OK(writer->Finish());
  ReportStatus("exit", "");
  return 0;
}
