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

#include "basic/stream/byte_stream.h"
#include "client/client.h"
#include "io/io/local_io_adaptor.h"
#include "io/io/utils.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, const char** argv) {
  if (argc < 3) {
    printf("usage ./simple_file_reader <ipc_socket> <efile>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);
  std::string efile = std::string(argv[2]);

  std::unique_ptr<LocalIOAdaptor> local_io_adaptor(
      new LocalIOAdaptor(efile.c_str()));
  VINEYARD_CHECK_OK(local_io_adaptor->Open());

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  ByteStreamBuilder builder(client);
  auto params = local_io_adaptor->GetMeta();
  builder.SetParams(params);
  auto bstream = std::dynamic_pointer_cast<ByteStream>(builder.Seal(client));
  VINEYARD_CHECK_OK(client.Persist(bstream->id()));
  ReportStatus(true, VYObjectIDToString(bstream->id()));

  auto writer = bstream->OpenWriter(client);
  writer->SetBufferSizeLimit(512 * 1024);

  std::string line;
  while (local_io_adaptor->ReadLine(line).ok()) {
    auto st = writer->WriteLine(line);
    if (!st.ok()) {
      ReportStatus(false, st.ToString());
      VINEYARD_CHECK_OK(st);
    }
  }

  {
    auto st = local_io_adaptor->Close();
    if (!st.ok()) {
      ReportStatus(false, st.ToString());
      VINEYARD_CHECK_OK(st);
    }
  }
  local_io_adaptor->Finalize();
  VINEYARD_CHECK_OK(writer->Finish());
  ReportStatus("exit", "");

  client.Disconnect();
  return 0;
}
