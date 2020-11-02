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
#include "io/io/io_factory.h"
#include "io/io/kafka_io_adaptor.h"
#include "io/io/utils.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  // kafka address format: kafka://brokers/topics/group_id/partition_num
  if (argc < 3) {
    printf("usage ./read_kafka_byte <ipc_socket> <kafka_address>");
    return 1;
  }

  std::string ipc_socket = std::string(argv[1]);
  std::string kafka_address = "kafka://" + std::string(argv[2]);
  std::unique_ptr<IIOAdaptor> kafka_io_adaptor =
      IOFactory::CreateIOAdaptor(kafka_address);
  VINEYARD_CHECK_OK(kafka_io_adaptor->Open());

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  ByteStreamBuilder builder(client);
  auto bstream = std::dynamic_pointer_cast<ByteStream>(builder.Seal(client));
  VINEYARD_CHECK_OK(client.Persist(bstream->id()));
  ReportStatus("return", VYObjectIDToString(bstream->id()));

  auto writer = bstream->OpenWriter(client);
  writer->SetBufferSizeLimit(2 * 1024 * 1024);

  std::string line;
  while (kafka_io_adaptor->ReadLine(line).ok()) {
    auto st = writer->WriteLine(line + "\n");
    if (!st.ok()) {
      ReportStatus("error", st.ToString());
      VINEYARD_CHECK_OK(st);
    }
  }

  VINEYARD_CHECK_OK(writer->Finish());
  ReportStatus("exit", "");

  return 0;
}
