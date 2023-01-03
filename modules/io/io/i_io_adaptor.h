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

#ifndef MODULES_IO_IO_I_IO_ADAPTOR_H_
#define MODULES_IO_IO_I_IO_ADAPTOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "common/util/status.h"

namespace vineyard {

/** I/O Interface.
 *
 * I/O Interface, handling io for varies locations.
 */
class IIOAdaptor {
 public:
  /** Default constructor. */
  IIOAdaptor() {}

  /** Default destructor. */
  virtual ~IIOAdaptor() {}

  virtual Status Open() = 0;
  virtual Status Open(const char* mode) = 0;
  virtual Status Close() = 0;

  /**
   * [SetPartialRead description]
   *
   * for local: it will read the i_th part of a file according to bytes offset.
   * table.
   *
   * @param  index
   * @param  total_parts
   * @return
   */
  virtual Status SetPartialRead(int index, int total_parts) = 0;

  /**
   * [Configure sub-class specific items]
   * e.g.,
   * local ReadLine use std::getline;
   *
   * @param  key   [description]
   * @param  value [description]
   * @return       [whether config successfully]
   */
  virtual Status Configure(const std::string& key,
                           const std::string& value) = 0;

  virtual Status ReadLine(std::string& line) = 0;
  virtual Status WriteLine(const std::string& line) = 0;

  virtual Status Read(void* buffer, size_t size) = 0;
  virtual Status Write(void* buffer, size_t size) = 0;

  virtual Status Flush() { return Status::OK(); }

  virtual Status ReadTable(std::shared_ptr<arrow::Table>* table) {
    return Status::OK();
  }

  virtual Status WriteTable(std::shared_ptr<arrow::Table> table) {
    return Status::OK();
  }

  // dir or file related:
  virtual Status ListDirectory(const std::string& path,
                               std::vector<std::string>& files) = 0;
  virtual Status MakeDirectory(const std::string& path) = 0;
  virtual bool IsExist(const std::string& path) = 0;

  virtual std::unordered_multimap<std::string, std::string> GetMeta() {
    return std::unordered_multimap<std::string, std::string>{};
  }
};

}  // namespace vineyard

#endif  // MODULES_IO_IO_I_IO_ADAPTOR_H_
