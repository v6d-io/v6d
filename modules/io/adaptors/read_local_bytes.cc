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
#include "io/io/i_io_adaptor.h"
#include "io/io/io_factory.h"

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
  CHECK_AND_REPORT(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  auto local_io_adaptor = IOFactory::CreateIOAdaptor(efile.c_str());

  CHECK_AND_REPORT(local_io_adaptor->SetPartialRead(proc, pnum));

  CHECK_AND_REPORT(local_io_adaptor->Open());

  auto params = local_io_adaptor->GetMeta();
  ByteStreamBuilder builder(client);
  builder.SetParams(params);
  auto bstream = std::dynamic_pointer_cast<ByteStream>(builder.Seal(client));
  CHECK_AND_REPORT(client.Persist(bstream->id()));
  LOG(INFO) << "Create byte stream: " << bstream->id();

  auto lstream =
      std::dynamic_pointer_cast<ByteStream>(client.GetObject(bstream->id()));
  LOG(INFO) << "Local stream: " << proc << " " << lstream->id();
  ReportStatus("return", VYObjectIDToString(lstream->id()));

  std::unique_ptr<ByteStreamWriter> writer;
  CHECK_AND_REPORT(lstream->OpenWriter(client, writer));
  writer->SetBufferSizeLimit(2 * 1024 * 1024);

  std::string line;
  while (local_io_adaptor->ReadLine(line).ok()) {
    auto st = writer->WriteLine(line);
    if (!st.ok()) {
      ReportStatus("error", st.ToString());
      CHECK_AND_REPORT(st);
    }
  }

  {
    auto st = local_io_adaptor->Close();
    if (!st.ok()) {
      ReportStatus("error", st.ToString());
      CHECK_AND_REPORT(st);
    }
  }
  CHECK_AND_REPORT(writer->Finish());

  return 0;
}
