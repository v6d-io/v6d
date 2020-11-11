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
#include "basic/stream/byte_stream.h"
#include "basic/stream/parallel_stream.h"
#include "client/client.h"
#include "io/io/local_io_adaptor.h"
#include "io/io/utils.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, const char** argv) {
  if (argc < 5) {
    printf(
        "usage ./read_local_bytes <ipc_socket> <efile> <proc_num> "
        "<proc_index>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);
  std::string efile = std::string(argv[2]);
  int pnum = std::stoi(argv[3]);
  int proc = std::stoi(argv[4]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  std::unique_ptr<LocalIOAdaptor> local_io_adaptor(
      new LocalIOAdaptor(efile.c_str()));

  VINEYARD_CHECK_OK(local_io_adaptor->SetPartialRead(proc, pnum));

  VINEYARD_CHECK_OK(local_io_adaptor->Open());

  auto params = local_io_adaptor->GetMeta();
  ByteStreamBuilder builder(client);
  builder.SetParams(params);
  auto bstream = std::dynamic_pointer_cast<ByteStream>(builder.Seal(client));
  VINEYARD_CHECK_OK(client.Persist(bstream->id()));
  LOG(INFO) << "Create byte stream: " << bstream->id();

  auto lstream =
      std::dynamic_pointer_cast<ByteStream>(client.GetObject(bstream->id()));
  LOG(INFO) << "Local stream: " << proc << " " << lstream->id();
  ReportStatus("return", VYObjectIDToString(lstream->id()));

  auto writer = lstream->OpenWriter(client);
  writer->SetBufferSizeLimit(2 * 1024 * 1024);

  std::string line;
  while (local_io_adaptor->ReadLine(line).ok()) {
    auto st = writer->WriteLine(line);
    if (!st.ok()) {
      ReportStatus("error", st.ToString());
      VINEYARD_CHECK_OK(st);
    }
  }

  local_io_adaptor->Finalize();
  {
    auto st = local_io_adaptor->Close();
    if (!st.ok()) {
      ReportStatus("error", st.ToString());
      VINEYARD_CHECK_OK(st);
    }
  }
  VINEYARD_CHECK_OK(writer->Finish());

  return 0;
}
