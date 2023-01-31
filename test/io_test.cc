/** Copyright 2021 Alibaba Group Holding Limited.

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

#include <bitset>
#include <iostream>

#include "common/util/logging.h"
#include "common/util/uuid.h"
#include "io/io/io_factory.h"

void ReadLines(std::string const& path_to_read) {
  auto io = vineyard::IOFactory::CreateIOAdaptor(path_to_read, nullptr);
  VINEYARD_CHECK_OK(io->Open());
  size_t lineno = 1;
  std::string line;
  while (true) {
    auto s = io->ReadLine(line);
    if (s.ok()) {
      LOG(INFO) << "line " << (lineno++) << ": '" << line << "' of length "
                << line.length();
    } else {
      LOG(ERROR) << "failed (or finished): " << s.ToString();
      break;
    }
  }
}

void ReadTable(std::string const& path_to_read) {
  constexpr int total_parts = 4;
  for (int index = 0; index < total_parts; ++index) {
    auto io = vineyard::IOFactory::CreateIOAdaptor(path_to_read, nullptr);
    VINEYARD_CHECK_OK(io->SetPartialRead(index, total_parts));
    VINEYARD_CHECK_OK(io->Open());

    for (auto const& item : io->GetMeta()) {
      LOG(INFO) << "item: " << item.first << " -> " << item.second;
    }

    // read as lines first
    {
      size_t lineno = 1;
      std::string line;
      while (true) {
        auto s = io->ReadLine(line);
        if (s.ok()) {
          LOG(INFO) << "line " << (lineno++) << ": '" << line << "' of length "
                    << line.length();
        } else {
          LOG(ERROR) << "failed (or finished): " << s;
          break;
        }
      }
    }

    // read as tables
    {
      std::shared_ptr<arrow::Table> table;
      LOG(INFO) << "read table: " << index << " of part " << total_parts << ": "
                << io->ReadTable(&table);
      if (table) {
        LOG(INFO) << "table: " << table->ToString();
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("usage ./io_test <lines or table> <path to read>");
    return 1;
  }

  std::string mode = std::string(argv[1]);
  std::string path_to_read = std::string(argv[2]);

  if (mode == "lines") {
    ReadLines(path_to_read);
  }
  if (mode == "table") {
    ReadTable(path_to_read);
  }

  LOG(INFO) << "Passed double array tests...";

  return 0;
}
