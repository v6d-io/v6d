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

#ifndef MODULES_IO_IO_LOCAL_IO_ADAPTOR_H_
#define MODULES_IO_IO_LOCAL_IO_ADAPTOR_H_

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "arrow/api.h"
#include "arrow/filesystem/api.h"
#include "arrow/io/api.h"

#include "common/util/functions.h"
#include "common/util/status.h"
#include "io/io/i_io_adaptor.h"
#include "io/io/io_factory.h"

namespace vineyard {
// FIXME: do not use fixed value, expend to double space when read to a
// threshold.
#define LINESIZE 65536

enum FileLocation {
  kFileLocationBegin = 0,
  kFileLocationCurrent = 1,
  kFileLocationEnd = 2,
};

class LocalIOAdaptor : public IIOAdaptor {
 public:
  /** Constructor.
   * @param location the location of file.
   */
  explicit LocalIOAdaptor(const std::string& location);

  /** Default destructor. */
  ~LocalIOAdaptor();

  static std::unique_ptr<IIOAdaptor> Make(const std::string& location,
                                          Client* client);

  Status Open() override;

  Status Open(const char* node) override;

  Status Close() override;

  Status ReadLine(std::string& line) override;

  /** Read the part of file given index and total_parts.
   * first cut the file into several parts with given
   * <total_part>, looking backwards for the nearest
   * '\n' with each breakpoint,then move breakpoint to
   * the next of the nearest character '\n', if successful
   * set file stream pointer to the index position.
   *
   * @param index the index in a part of file
   * @param total_parts total number of parts in file
   * @return true if set partial read successful, else false
   *
   * */
  Status SetPartialRead(const int index, const int total_parts) override;

  Status Configure(const std::string& key, const std::string& value) override;

  Status WriteLine(const std::string& line) override;

  Status Read(void* buffer, size_t size) override;

  Status Write(void* buffer, size_t size) override;

  Status Flush() override;

  Status ListDirectory(const std::string& path,
                       std::vector<std::string>& files) override;

  Status MakeDirectory(const std::string& path) override;

  bool IsExist(const std::string& path) override;

  static void Init();

  static void Finalize();

  inline const std::string& location() const { return location_; }

  Status GetPartialReadDetail(int64_t& offset, int64_t& nbytes);

  Status ReadTable(std::shared_ptr<arrow::Table>* table) override;

  Status WriteTable(std::shared_ptr<arrow::Table> table) override;

  Status ReadPartialTable(std::shared_ptr<arrow::Table>* table, int index);

  Status Seek(const int64_t offset);

  int64_t GetFullSize();

  std::unordered_multimap<std::string, std::string> GetMeta() override {
    return meta_;
  }

 private:
  int64_t tell();
  Status seek(const int64_t offset, const FileLocation seek_from);
  Status setPartialReadImpl();
  int64_t getDistanceToLineBreak(const int index);

  std::string trimBOM(const std::string& line);

  std::string location_;
  char buff[LINESIZE];
  std::shared_ptr<arrow::fs::FileSystem> fs_;
  std::shared_ptr<arrow::io::RandomAccessFile> ifp_;  // for input
  std::shared_ptr<arrow::io::OutputStream> ofp_;      // for output

  // for arrow
  std::vector<std::string> columns_;
  std::vector<std::string> column_types_;
  char delimiter_ = ',';
  bool header_row_ = false;
  std::string header_line_ = "";
  bool include_all_columns_ = false;
  // schema of header row
  std::vector<std::string> original_columns_;

  bool enable_partial_read_;
  std::vector<int64_t> partial_read_offset_;
  int total_parts_;
  int index_;
  std::unordered_multimap<std::string, std::string> meta_;

  // register
  static const bool registered_;
};
}  // namespace vineyard

#endif  // MODULES_IO_IO_LOCAL_IO_ADAPTOR_H_
