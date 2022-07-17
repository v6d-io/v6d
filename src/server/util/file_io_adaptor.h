/** Copyright 2020-2021 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The implementation of `get_rss`, `get_shared_rss` and `get_peek_rss`
are referred from https://stackoverflow.com/a/14927379/5080177, which
has the following license header originally:

  * Author:  David Robert Nadeau
  * Site:    http://NadeauSoftware.com/
  * License: Creative Commons Attribution 3.0 Unported License
  *          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#ifndef SRC_SERVER_UTIL_FILE_IO_ADAPTOR_H_
#define SRC_SERVER_UTIL_FILE_IO_ADAPTOR_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "arrow/api.h"
#include "arrow/filesystem/api.h"
#include "arrow/io/api.h"
#include "arrow/io/interfaces.h"

#include "common/util/status.h"
#include "common/util/uuid.h"

namespace util {
/**
 * @brief A customized adaptor for spilling and reloading file,
 * especailly implement `RemoveFile` for garbage collection
 */
class FileIOAdaptor {
 public:
  FileIOAdaptor() = delete;
  explicit FileIOAdaptor(const std::string& dir_path);
  ~FileIOAdaptor();
  vineyard::Status Open();
  vineyard::Status Open(const char* mode);
  vineyard::Status Write(const char* buf, size_t size);
  vineyard::Status Flush();
  vineyard::Status Read(void* buffer, size_t size);
  vineyard::Status Close();
  vineyard::Status RemoveFile(const std::string& path);
  vineyard::Status RemoveFiles(const std::vector<std::string>& paths);
  vineyard::Status DeleteDir();

 private:
  vineyard::Status CreateDir(const std::string& path);

  std::string location_;
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::io::RandomAccessFile> ifp_;
  std::shared_ptr<arrow::io::OutputStream> ofp_;
};
}  // namespace util

#endif  // SRC_SERVER_UTIL_FILE_IO_ADAPTOR_H_
