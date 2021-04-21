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
#include "basic/stream/parallel_stream.h"
#include "client/client.h"
#include "io/io/io_factory.h"
#include "io/io/kafka_io_adaptor.h"
#include "io/io/utils.h"

using namespace vineyard;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  // kafka address format: kafka://brokers/topics/group_id/partition_num
  if (argc < 6) {
    printf(
        "usage ./write_kafka_dataframe <ipc_socket> "
        "<stream_id> <kafka_address> <proc_num> <proc_index>");
    return 1;
  }

  std::string ipc_socket = std::string(argv[1]);
  ObjectID stream_id = VYObjectIDFromString(argv[2]);
  std::string kafka_address = std::string(argv[3]);
  int proc_num = std::stoi(argv[4]);
  int proc_index = std::stoi(argv[5]);

  std::unique_ptr<IIOAdaptor> kafka_io_adaptor =
      IOFactory::CreateIOAdaptor(kafka_address);
  CHECK_AND_REPORT(kafka_io_adaptor->Open("w"));

  Client client;
  CHECK_AND_REPORT(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  auto s =
      std::dynamic_pointer_cast<ParallelStream>(client.GetObject(stream_id));
  LOG(INFO) << "Got parallel stream " << s->id();

  VINEYARD_ASSERT(static_cast<size_t>(proc_num) == s->GetStreamSize(),
                  "Different ProcNum(" + std::to_string(proc_num) +
                      ") from StreamSize(" +
                      std::to_string(s->GetStreamSize()) + ")");

  auto ls = s->GetStream<DataframeStream>(proc_index);
  LOG(INFO) << "Got dataframe stream " << ls->id() << " at " << proc_index;

  std::unique_ptr<DataframeStreamReader> reader;
  CHECK_AND_REPORT(ls->OpenReader(client, reader));

  std::string line;
  while (reader->ReadLine(line).ok()) {
    CHECK_AND_REPORT(kafka_io_adaptor->WriteLine(line));
  }

  return 0;
}
