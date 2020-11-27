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

#ifndef MODULES_IO_IO_OSS_IO_ADAPTOR_H_
#define MODULES_IO_IO_OSS_IO_ADAPTOR_H_

#ifdef OSS_ENABLED

#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "alibabacloud/oss/OssClient.h"
#include "gflags/gflags.h"

#include "common/util/blocking_queue.h"
#include "io/io/i_io_adaptor.h"

DECLARE_string(oss_endpoint);
DECLARE_string(oss_access_key_id);
DECLARE_string(oss_access_key_secret);
DECLARE_string(oss_suffix);
DECLARE_int32(oss_concurrency);
DECLARE_int32(oss_retries);

namespace vineyard {
class OSSIOAdaptor : public IIOAdaptor {
 public:
  explicit OSSIOAdaptor(const std::string& location);

  ~OSSIOAdaptor() override;

  Status Open() override;

  Status Open(const char* mode) override;

  Status Close() override;

  Status SetPartialRead(int index, int total_parts) override;

  Status Configure(const std::string& key, const std::string& value) override {
    return Status::OK();
  }

  Status ReadLine(std::string& line) override;

  Status WriteLine(const std::string& line) override {
    return Status::NotImplemented();
  }

  Status Read(void* buffer, size_t size) override {
    return Status::NotImplemented();
  }

  Status Write(void* buffer, size_t size) override {
    return Status::NotImplemented();
  }

  Status ReadTable(std::shared_ptr<arrow::Table>* table) override;

  Status ListDirectory(const std::string& path,
                       std::vector<std::string>& files) override {
    return Status::NotImplemented();
  }

  Status MakeDirectory(const std::string& path) override {
    return Status::NotImplemented();
  }

  bool IsExist(const std::string& path) override;

  static void Init();

  static void Finalize();

  std::unordered_multimap<std::string, std::string> GetMeta() override {
    return meta_;
  }

 private:
  Status getTotalSize(size_t& size);

  Status readLine(std::string& line, size_t& cursor);

  Status getRange(const size_t begin, const size_t end, std::string& content);

  void parseOssCredentials(const std::string& file_name);
  void parseOssEnvironmentVariables();
  void parseGFlags();
  void parseYamlLocation(const std::string& location);

  const std::string OSS_CREDENTIALS_PATH = "$HOME/.osscredentials";
  const char* OSS_ACCESS_ID = "accessid";
  const char* OSS_ACCESS_KEY = "accesskey";
  const char* OSS_ENDPOINT = "host";

  AlibabaCloud::OSS::ClientConfiguration conf_;

  std::string current_buffer_;
  size_t buffer_next_;

  std::string location_;

  std::string oss_endpoint_;
  std::string access_id_;
  std::string access_key_;
  std::string sts_token_;

  std::string bucket_name_;
  std::string object_name_;

  std::vector<std::string> objects_;
  size_t current_object_;

  bool opened_ = false;
  bool partial_read_ = false;

  size_t producer_num_;
  std::atomic<size_t> exited_producers_;
  std::vector<std::thread> producers_;

  PCBlockingQueue<std::string> queue_;

  size_t line_cur_ = 0;
  size_t begin_ = 0;
  size_t end_;
  size_t seg_len_ = 1024;
  size_t total_len_;
  size_t part_num_;
  size_t part_id_;
  std::unordered_multimap<std::string, std::string> meta_;

  // for arrow
  std::vector<std::string> columns_;
  std::vector<std::string> column_types_;
  char delimiter_ = ',';
  bool header_row_;
  std::string header_line_ = "";
  // schema of header row
  std::vector<std::string> original_columns_;
};
}  // namespace vineyard

#endif  // OSS_ENABLED
#endif  // MODULES_IO_IO_OSS_IO_ADAPTOR_H_
