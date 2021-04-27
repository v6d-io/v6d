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

#include "boost/algorithm/string.hpp"

#include "basic/ds/arrow_utils.h"
#include "basic/stream/byte_stream.h"
#include "basic/stream/dataframe_stream.h"
#include "basic/stream/parallel_stream.h"
#include "client/client.h"
#include "io/io/i_io_adaptor.h"
#include "io/io/io_factory.h"

#include "io/io/utils.h"

using namespace vineyard;  // NOLINT(build/namespaces)

Status ParseTable(std::shared_ptr<arrow::Table>* table,
                  std::unique_ptr<arrow::Buffer>& buffer, char delimiter,
                  bool header_row, std::vector<std::string> columns,
                  std::vector<std::string> column_types,
                  std::vector<std::string> original_columns,
                  bool include_all_columns) {
  // FIXME IF NO NEED TO COPY
  std::shared_ptr<arrow::Buffer> copied_buffer;
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
  CHECK_ARROW_ERROR(buffer->Copy(0, buffer->size(), &copied_buffer));
#else
  CHECK_ARROW_ERROR_AND_ASSIGN(copied_buffer,
                               buffer->CopySlice(0, buffer->size()));
#endif
  auto buffer_reader = std::make_shared<arrow::io::BufferReader>(copied_buffer);

  std::shared_ptr<arrow::io::InputStream> input =
      arrow::io::RandomAccessFile::GetStream(buffer_reader, 0,
                                             copied_buffer->size());

  arrow::MemoryPool* pool = arrow::default_memory_pool();

  auto read_options = arrow::csv::ReadOptions::Defaults();
  auto parse_options = arrow::csv::ParseOptions::Defaults();

  read_options.column_names = original_columns;
  parse_options.delimiter = delimiter;

  auto convert_options = arrow::csv::ConvertOptions::Defaults();

  auto is_number = [](const std::string& s) -> bool {
    return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                           return !std::isdigit(c);
                         }) == s.end();
  };

  std::vector<int> indices;
  for (size_t i = 0; i < columns.size(); ++i) {
    if (is_number(columns[i])) {
      int col_idx = std::stoi(columns[i]);
      if (col_idx >= static_cast<int>(original_columns.size())) {
        return Status(StatusCode::kArrowError,
                      "Index out of range: " + columns[i]);
      }
      indices.push_back(col_idx);
      columns[i] = original_columns[col_idx];
    }
  }

  // If include_all_columns_ is set, push other names as well
  if (include_all_columns) {
    for (const auto& col : original_columns) {
      if (std::find(std::begin(columns), std::end(columns), col) ==
          columns.end()) {
        columns.push_back(col);
      }
    }
  }

  convert_options.include_columns = columns;

  if (column_types.size() > convert_options.include_columns.size()) {
    return Status(StatusCode::kArrowError,
                  "Format of column type schema is incorrect.");
  }
  std::unordered_map<std::string, std::shared_ptr<arrow::DataType>>
      arrow_column_types;

  for (size_t i = 0; i < column_types.size(); ++i) {
    if (!column_types[i].empty()) {
      arrow_column_types[convert_options.include_columns[i]] =
          type_name_to_arrow_type(column_types[i]);
    }
  }
  convert_options.column_types = arrow_column_types;

  std::shared_ptr<arrow::csv::TableReader> reader;
#if defined(ARROW_VERSION) && ARROW_VERSION >= 4000000
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      reader, arrow::csv::TableReader::Make(arrow::io::AsyncContext(pool),
                                            input, read_options, parse_options,
                                            convert_options));
#else
  RETURN_ON_ARROW_ERROR_AND_ASSIGN(
      reader, arrow::csv::TableReader::Make(pool, input, read_options,
                                            parse_options, convert_options));
#endif

  auto result = reader->Read();
  if (!result.status().ok()) {
    if (result.status().message() == "Empty CSV file") {
      *table = nullptr;
      return Status::OK();
    } else {
      return Status::ArrowError(result.status());
    }
  }
  *table = result.ValueOrDie();

  RETURN_ON_ARROW_ERROR((*table)->Validate());

  VLOG(2) << "Parsed: " << (*table)->num_rows() << " rows, "
          << (*table)->num_columns() << " columns";
  VLOG(2) << (*table)->schema()->ToString();
  return Status::OK();
}

int main(int argc, const char** argv) {
  if (argc < 5) {
    printf(
        "usage ./parse_bytes_to_dataframe <ipc_socket> <stream_id> "
        "<proc_num> <proc_index>\n");
    return 1;
  }

  std::string ipc_socket = std::string(argv[1]);
  ObjectID stream_id = VYObjectIDFromString(argv[2]);
  int proc_num = std::stoi(argv[3]);
  int proc_index = std::stoi(argv[4]);

  Client client;
  CHECK_AND_REPORT(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  auto s =
      std::dynamic_pointer_cast<ParallelStream>(client.GetObject(stream_id));
  LOG(INFO) << "Got parallel stream " << s->id();

  auto ls = s->GetStream<ByteStream>(proc_index);
  LOG(INFO) << "Got byte stream " << ls->id() << " at " << proc_index << " (of "
            << proc_num << ")";

  auto params = ls->GetParams();
  bool header_row = (params["header_row"] == "1");
  std::string delimiter = params["delimiter"];
  if (delimiter.empty()) {
    delimiter = ",";
  }
  std::vector<std::string> columns;
  std::vector<std::string> column_types;
  std::vector<std::string> original_columns;
  std::string header_line;

  if (header_row) {
    header_line = params["header_line"];
    if (header_line.empty()) {
      ReportStatus("error",
                   "Header line not found while header_row is set to True");
    }
    ::boost::algorithm::trim(header_line);
    ::boost::split(original_columns, header_line,
                   ::boost::is_any_of(delimiter.substr(0, 1)));
  } else {
    // Name columns as f0 ... fn
    std::string one_line = params["header_line"];
    ::boost::algorithm::trim(one_line);
    std::vector<std::string> one_column;
    ::boost::split(one_column, one_line,
                   ::boost::is_any_of(delimiter.substr(0, 1)));
    for (size_t i = 0; i < one_column.size(); ++i) {
      original_columns.push_back("f" + std::to_string(i));
    }
  }

  if (params.find("schema") != params.end()) {
    VLOG(2) << "param schema: " << params["schema"];
    ::boost::split(columns, params["schema"], ::boost::is_any_of(","));
  }
  if (params.find("column_types") != params.end()) {
    VLOG(2) << "param column_types: " << params["column_types"];
    ::boost::split(column_types, params["column_types"],
                   ::boost::is_any_of(","));
  }
  bool include_all_columns = false;
  if (params.find("include_all_columns") != params.end()) {
    VLOG(2) << "param include_all_columns: " << params["include_all_columns"];
    include_all_columns = (params["include_all_columns"] == "1");
  }

  DataframeStreamBuilder dfbuilder(client);
  dfbuilder.SetParams(params);
  auto bs = std::dynamic_pointer_cast<DataframeStream>(dfbuilder.Seal(client));
  CHECK_AND_REPORT(client.Persist(bs->id()));
  LOG(INFO) << "Created dataframe stream " << bs->id() << " at " << proc_index;
  ReportStatus("return", VYObjectIDToString(bs->id()));

  std::unique_ptr<ByteStreamReader> reader;
  std::unique_ptr<DataframeStreamWriter> writer;
  CHECK_AND_REPORT(ls->OpenReader(client, reader));
  CHECK_AND_REPORT(bs->OpenWriter(client, writer));

  while (true) {
    std::unique_ptr<arrow::Buffer> buffer;
    auto status = reader->GetNext(buffer);
    if (status.ok()) {
      VLOG(10) << "consumer: buffer size = " << buffer->size();
      std::shared_ptr<arrow::Table> table;
      Status st =
          ParseTable(&table, buffer, delimiter[0], header_row, columns,
                     column_types, original_columns, include_all_columns);
      if (!st.ok()) {
        ReportStatus("error", st.ToString());
      }
      st = writer->WriteTable(table);
      if (!st.ok()) {
        ReportStatus("error", st.ToString());
      }
    } else {
      if (status.IsStreamDrained()) {
        LOG(INFO) << "Stream drained";
        break;
      }
    }
  }
  auto status = writer->Finish();
  if (status.ok()) {
    ReportStatus("exit", "");
  } else {
    ReportStatus("error", status.ToString());
  }
  return 0;
}
