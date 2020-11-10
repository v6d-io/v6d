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

#include "basic/stream/dataframe_stream.h"
#include "client/client.h"
#include "io/io/local_io_adaptor.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, const char** argv) {
  if (argc < 3) {
    printf("usage ./dataframe_stream_reader <ipc_socket> <stream_id>");
    return 1;
  }

  std::string ipc_socket = std::string(argv[1]);
  ObjectID stream_id = VYObjectIDFromString(argv[2]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  auto s =
      std::dynamic_pointer_cast<DataframeStream>(client.GetObject(stream_id));
  LOG(INFO) << "Got dataframe stream: " << s->id();
  auto reader = s->OpenReader(client);

  std::shared_ptr<arrow::Table> table;
  VINEYARD_CHECK_OK(reader->ReadTable(table));

  LOG(INFO) << table->num_rows() << " rows, " << table->num_columns()
            << " columns";
  LOG(INFO) << table->schema()->ToString();

  return 0;
}
