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

#include <string>

#include "arrow/table.h"
#include "basic/stream/dataframe_stream.h"
#include "basic/stream/parallel_stream.h"
#include "client/client.h"
#include "io/io/oss_io_adaptor.h"
#include "io/io/utils.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, const char** argv) {
  if (argc < 5) {
    printf(
        "usage ./read_oss_dataframe <ipc_socket> <efile> <proc_num> "
        "<proc_index>");
    return 1;
  }

#ifdef OSS_ENABLED

  std::string ipc_socket = std::string(argv[1]);
  std::string efile = std::string(argv[2]);
  int pnum = std::stoi(argv[3]);
  int proc = std::stoi(argv[4]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  std::unique_ptr<OSSIOAdaptor> oss_io_adaptor(new OSSIOAdaptor(efile.c_str()));

  VINEYARD_CHECK_OK(oss_io_adaptor->SetPartialRead(proc, pnum));

  VINEYARD_CHECK_OK(oss_io_adaptor->Open());

  auto params = oss_io_adaptor->GetMeta();
  DataframeStreamBuilder builder(client);
  builder.SetParams(params);
  auto dfstream =
      std::dynamic_pointer_cast<DataframeStream>(builder.Seal(client));
  VINEYARD_CHECK_OK(client.Persist(dfstream->id()));
  LOG(INFO) << "Create dataframe stream: " << dfstream->id();

  auto lstream = std::dynamic_pointer_cast<DataframeStream>(
      client.GetObject(dfstream->id()));
  LOG(INFO) << "Local stream: " << proc << " " << lstream->id();
  ReportStatus("return", VYObjectIDToString(lstream->id()));

  auto writer = lstream->OpenWriter(client);

  std::shared_ptr<arrow::Table> table;
  VINEYARD_CHECK_OK(oss_io_adaptor->ReadTable(&table));

  auto st = writer->WriteTable(table);
  if (!st.ok()) {
    ReportStatus("error", st.ToString());
    VINEYARD_CHECK_OK(st);
  }

  oss_io_adaptor->Finalize();
  {
    auto st = oss_io_adaptor->Close();
    if (!st.ok()) {
      ReportStatus("error", st.ToString());
      VINEYARD_CHECK_OK(st);
    }
  }
  VINEYARD_CHECK_OK(writer->Finish());

#endif  // OSS_ENABLED

  return 0;
}
