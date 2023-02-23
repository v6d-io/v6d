/** Copyright 2020-2023 Alibaba Group Holding Limited.

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

#ifndef SRC_SERVER_UTIL_FILE_IO_ADAPTOR_H_
#define SRC_SERVER_UTIL_FILE_IO_ADAPTOR_H_

#include <memory>
#include <string>
#include <vector>

#include "arrow/api.h"
#include "arrow/filesystem/api.h"
#include "arrow/io/api.h"
#include "arrow/io/interfaces.h"
#include "common/util/status.h"

namespace vineyard {
namespace io {

/**
 * @brief A customized adaptor for spilling and reloading file,
 * especially implement `RemoveFile` for garbage collection
 */
class FileIOAdaptor {
 public:
  FileIOAdaptor() = delete;
  explicit FileIOAdaptor(const std::string& dir_path);
  ~FileIOAdaptor();
  Status Open();
  Status Open(const char* mode);
  Status Write(const char* buf, size_t size);
  Status Flush();
  Status Read(void* buffer, size_t size);
  Status Close();
  Status RemoveFile(const std::string& path);
  Status RemoveFiles(const std::vector<std::string>& paths);
  Status DeleteDir();

 private:
  Status CreateDir(const std::string& path);

  std::string location_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::io::RandomAccessFile> ifp_;
  std::shared_ptr<arrow::io::OutputStream> ofp_;
};

}  // namespace io
}  // namespace vineyard

#endif  // SRC_SERVER_UTIL_FILE_IO_ADAPTOR_H_
