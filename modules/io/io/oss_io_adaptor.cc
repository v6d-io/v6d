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

#include "io/io/oss_io_adaptor.h"

#ifdef OSS_ENABLED

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <memory>
#include <random>
#include <thread>

#include "alibabacloud/oss/OssClient.h"
#include "alibabacloud/oss/client/RetryStrategy.h"
#include "arrow/api.h"
#include "arrow/csv/api.h"
#include "arrow/io/api.h"
#include "boost/algorithm/string.hpp"
#include "glog/logging.h"
#include "network/uri.hpp"
#include "network/uri/uri_io.hpp"

#include "basic/ds/arrow_utils.h"
#include "common/util/functions.h"

DEFINE_string(oss_endpoint, "", "OSS endpoint");
DEFINE_string(oss_access_key_id, "", "OSS Access Key ID");
DEFINE_string(oss_access_key_secret, "", "OSS Access Key Secret");
DEFINE_string(oss_suffix, "", "OSS file filtering suffix");
DEFINE_int32(oss_concurrency, 4, "concurrency of oss client");
DEFINE_int32(oss_retries, 5, "oss upload/download retry times");

using AlibabaCloud::OSS::ClientConfiguration;
using AlibabaCloud::OSS::Error;
using AlibabaCloud::OSS::ERROR_CURL_BASE;
using AlibabaCloud::OSS::GetObjectRequest;
using AlibabaCloud::OSS::InitializeSdk;
using AlibabaCloud::OSS::ListObjectOutcome;
using AlibabaCloud::OSS::ListObjectsRequest;
using AlibabaCloud::OSS::OssClient;
using AlibabaCloud::OSS::PutObjectRequest;
using AlibabaCloud::OSS::RetryStrategy;
using AlibabaCloud::OSS::ShutdownSdk;

namespace vineyard {

class UserRetryStrategy : public RetryStrategy {
 public:
  /* maxRetries表示最大重试次数，scaleFactor为重试等待时间的尺度因子。*/
  explicit UserRetryStrategy(long maxRetries = 3,     // NOLINT(runtime/int)
                             long scaleFactor = 300)  // NOLINT(runtime/int)
      : m_scaleFactor(scaleFactor), m_maxRetries(maxRetries) {}

  /* 您可以自定义shouldRetry函数，该函数用于判断是否进行重试。*/
  bool shouldRetry(const Error& error,
                   long attemptedRetries) const;  // NOLINT(runtime/int)

  /* 您可以自定义calcDelayTimeMs函数，该函数用于计算重试的延迟等待时间。*/
  long calcDelayTimeMs(const Error& error,            // NOLINT(runtime/int)
                       long attemptedRetries) const;  // NOLINT(runtime/int)

 private:
  long m_scaleFactor;  // NOLINT(runtime/int)
  long m_maxRetries;   // NOLINT(runtime/int)
};

bool UserRetryStrategy::shouldRetry(
    const Error& error,
    long attemptedRetries) const {  // NOLINT(runtime/int)
  if (attemptedRetries >= m_maxRetries)
    return false;

  long responseCode = error.Status();  // NOLINT(runtime/int)

  // http code
  if ((responseCode == 403 &&
       error.Message().find("RequestTimeTooSkewed") != std::string::npos) ||
      (responseCode > 499 && responseCode < 599)) {
    return true;
  } else {
    switch (responseCode) {
    // curl error code
    case (ERROR_CURL_BASE + 7):   // CURLE_COULDNT_CONNECT
    case (ERROR_CURL_BASE + 18):  // CURLE_PARTIAL_FILE
    case (ERROR_CURL_BASE + 23):  // CURLE_WRITE_ERROR
    case (ERROR_CURL_BASE + 28):  // CURLE_OPERATION_TIMEDOUT
    case (ERROR_CURL_BASE + 52):  // CURLE_GOT_NOTHING
    case (ERROR_CURL_BASE + 55):  // CURLE_SEND_ERROR
    case (ERROR_CURL_BASE + 56):  // CURLE_RECV_ERROR
      return true;
    default:
      break;
    }
  }

  return false;
}

long UserRetryStrategy::calcDelayTimeMs(  // NOLINT(runtime/int)
    const Error& error,
    long attemptedRetries) const {  // NOLINT(runtime/int)
  return (1 << attemptedRetries) * m_scaleFactor;
}

OSSIOAdaptor::OSSIOAdaptor(const std::string& location) : location_(location) {
  parseOssCredentials(OSS_CREDENTIALS_PATH);
  parseOssEnvironmentVariables();
  parseGFlags();
  parseYamlLocation(location);
  // There are two kinds of objects. one end with .meta, one end with .tsv,
  // We only want .tsv file
  suffix_ = FLAGS_oss_suffix;
  current_object_ = 0;

  producer_num_ = FLAGS_oss_concurrency;

  exited_producers_ = 0;
  buffer_next_ = 0;

  // default connections is 16. Use high value if specified.
  conf_.maxConnections = std::max(FLAGS_oss_concurrency, 16);

  auto defaultRetryStrategy =
      std::make_shared<UserRetryStrategy>(FLAGS_oss_retries, 300);
  conf_.retryStrategy = defaultRetryStrategy;

  if (access_id_.empty() || access_key_.empty() || oss_endpoint_.empty()) {
    LOG(FATAL) << "access id / access key / endpoint cannot be empty.";
  }
  if (bucket_name_.empty() || prefix_.empty()) {
    LOG(FATAL) << "bucket name or prefix cannot be empty.";
  }
  VLOG(2) << "id: " << access_id_;
  VLOG(2) << "key: " << access_key_;
  VLOG(2) << "sts_token: " << sts_token_;
  VLOG(2) << "endpoint: " << oss_endpoint_;
  VLOG(2) << "bucket_name: " << bucket_name_;
  VLOG(2) << "prefix: " << prefix_;
  VLOG(2) << "suffix: " << suffix_;
  VLOG(2) << "concurrency: " << producer_num_;
}

OSSIOAdaptor::~OSSIOAdaptor() {
  for (auto& thrd : producers_) {
    if (thrd.joinable()) {
      thrd.join();
    }
  }
}

void OSSIOAdaptor::parseOssCredentials(const std::string& file_name) {
  VLOG(2) << "[OSS]: loading credentials from " << file_name;
  try {
    std::ifstream credentials(ExpandEnvironmentVariables(file_name),
                              std::ifstream::in);
    std::string line;
    while (std::getline(credentials, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      std::vector<std::string> line_list;
      boost::split(line_list, line, boost::is_any_of("="));
      if (line_list.size() == 2) {
        boost::trim(line_list[0]);
        boost::trim(line_list[1]);
        if (line_list[0] == OSS_ACCESS_ID) {
          access_id_ = line_list[1];
        } else if (line_list[0] == OSS_ACCESS_KEY) {
          access_key_ = line_list[1];
        } else if (line_list[0] == OSS_ENDPOINT) {
          oss_endpoint_ = line_list[1];
        }
      }
    }
    credentials.close();
  } catch (std::exception& e) {
    VLOG(2) << "[OSS] .osscredentials file not exist, skip search";
  }
}

void OSSIOAdaptor::parseOssEnvironmentVariables() {
  char* s = NULL;
  s = getenv(OSS_ACCESS_ID);
  if (s != NULL && strlen(s) != 0) {
    access_id_ = s;
  }
  s = getenv(OSS_ACCESS_KEY);
  if (s != NULL && strlen(s) != 0) {
    access_key_ = s;
  }
  s = getenv(OSS_ENDPOINT);
  if (s != NULL && strlen(s) != 0) {
    oss_endpoint_ = s;
  }
}

void OSSIOAdaptor::parseGFlags() {
  if (!FLAGS_oss_access_key_id.empty()) {
    access_id_ = FLAGS_oss_access_key_id;
  }
  if (!FLAGS_oss_access_key_secret.empty()) {
    access_key_ = FLAGS_oss_access_key_secret;
  }
  if (!FLAGS_oss_endpoint.empty()) {
    oss_endpoint_ = FLAGS_oss_endpoint;
  }
}

void OSSIOAdaptor::parseYamlLocation(const std::string& location) {
  // location format:
  // oss://<access_id>:<access_key>@<endpoint>/<bucket_name>/<prefix>
  // oss:///<bucket_name>/<prefix>
  // oss://bucket_name/prefix

  network::uri instance(location);

  std::string user_info = instance.user_info().to_string();
  std::string host = instance.host().to_string();
  std::string path = instance.path().to_string();

  if (!user_info.empty()) {
    std::vector<std::string> infos;
    boost::split(infos, user_info, boost::is_any_of(":"));
    CHECK_EQ(infos.size(), 2);
    access_id_ = infos[0];
    access_key_ = infos[1];
  }
  bool bucket_name_as_host = false;
  if (host.find(".com") != std::string::npos) {
    oss_endpoint_ = host;
  } else if (!host.empty()) {  // oss://bucket/prefix
    bucket_name_ = host;
    bucket_name_as_host = true;
  }

  std::vector<std::string> locs;
  boost::split(locs, path, boost::is_any_of("/"));
  if (bucket_name_as_host) {
    // path format is '/prefix'
    prefix_ = path.substr(1);  // skip one '/'
  } else {
    // path format is '/bucket/prefix'
    // so the locs[0] will be an empty string.
    if (locs.size() >= 3) {  // oss:///bucket/prefix
      bucket_name_ = locs[1];
      prefix_ = path.substr(bucket_name_.size() + 2);  // skip two '/'
    } else {
      LOG(FATAL) << "Invalid uri: " << location;
    }
  }
}

void OSSIOAdaptor::Init() { InitializeSdk(); }

void OSSIOAdaptor::Finalize() { ShutdownSdk(); }

// if mode has 'a' or 'w', then create upload request.
// else create download request.
// I think in practice, we don't need to both read and write a table,
// so I don't support the rw, to make the performance better.
Status OSSIOAdaptor::Open(const char* mode) {
  if (strchr(mode, 'w') != NULL || strchr(mode, 'a') != NULL) {
    VLOG(2) << "Open OSS Adaptor, mode: " << mode;
    return Status::OK();
  } else {
    return Open();
  }
}

Status OSSIOAdaptor::Open() {
  opened_ = true;

  if (!listAllObjects(prefix_, suffix_)) {
    LOG(FATAL) << "IOException: OSS Exception: list object failed.";
  }
  if (partial_read_) {
    selectObjects();
  }
  producer_num_ = std::min(producer_num_, objects_.size());
  producers_.resize(producer_num_);
  queue_.SetProducerNum(producer_num_);
  for (size_t i = 0; i < producer_num_; ++i) {
    producers_[i] = std::thread(&OSSIOAdaptor::producerRoutine, this);
  }
  return Status::OK();
}

Status OSSIOAdaptor::ReadLine(std::string& line) {
  while (true) {
    if (buffer_next_ >= current_buffer_.size()) {
      if (exited_producers_ == producer_num_ && queue_.Size() == 0) {
        return Status::EndOfFile();
      }
      queue_.Get(current_buffer_);
      buffer_next_ = 0;
    } else {
      auto next = current_buffer_.find('\n', buffer_next_);
      line = current_buffer_.substr(buffer_next_, next - buffer_next_);
      buffer_next_ = next == std::string::npos ? next : next + 1;
      return Status::OK();
    }
  }
}

Status OSSIOAdaptor::ReadTable(std::shared_ptr<arrow::Table>* table) {
  arrow::BufferBuilder builder;
  std::string buffer;
  while (!(exited_producers_ == producer_num_ && queue_.Size() == 0)) {
    queue_.Get(buffer);
    RETURN_ON_ARROW_ERROR(builder.Append(buffer.c_str(), buffer.size()));
  }
  std::shared_ptr<arrow::Buffer> buf;
  RETURN_ON_ARROW_ERROR(builder.Finish(&buf));
  auto file = std::make_shared<arrow::io::BufferReader>(buf);
  auto stream = arrow::io::RandomAccessFile::GetStream(file, 0, buf->size());
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  auto read_options = arrow::csv::ReadOptions::Defaults();
  auto parse_options = arrow::csv::ParseOptions::Defaults();

  auto convert_options = arrow::csv::ConvertOptions::Defaults();
  auto reader_result = arrow::csv::TableReader::Make(
      pool, stream, read_options, parse_options, convert_options);

  RETURN_ON_ARROW_ERROR(reader_result.status());

  auto reader = reader_result.ValueOrDie();
  auto table_result = reader->Read();
  RETURN_ON_ARROW_ERROR(table_result.status());
  *table = table_result.ValueOrDie();
  return Status::OK();
}

Status OSSIOAdaptor::Write(void* buffer, size_t size) {
  OssClient client(oss_endpoint_, access_id_, access_key_, conf_);
  std::shared_ptr<std::iostream> content =
      std::make_shared<std::stringstream>();
  *content << static_cast<char*>(buffer);
  PutObjectRequest request(bucket_name_, prefix_, content);
  auto outcome = client.PutObject(request);
  if (!outcome.isSuccess()) {
    LOG(ERROR) << "Put object failed, code: " << outcome.error().Code()
               << ", message: " << outcome.error().Message()
               << ", requestId: " << outcome.error().RequestId();
    return Status::IOError(outcome.error().Message());
  }
  return Status::OK();
}

void OSSIOAdaptor::selectObjects() {
  VLOG(2) << "Using partial read strategy: index = " << part_id_
          << " total = " << part_num_;
  size_t object_num = objects_.size();
  std::vector<std::string> selected;
  for (size_t i = 0; i < object_num; ++i) {
    if (i % part_num_ == part_id_) {
      selected.emplace_back(objects_[i]);
    }
  }
  objects_.swap(selected);
}

Status OSSIOAdaptor::listAllObjects(const std::string& prefix,
                                    const std::string& suffix) {
  OssClient client(oss_endpoint_, access_id_, access_key_, conf_);

  std::string next_marker;
  bool IsTruncated = false;
  do {
    ListObjectsRequest request(bucket_name_);
    request.setPrefix(prefix);
    request.setMarker(next_marker);
    auto outcome = client.ListObjects(request);
    if (!outcome.isSuccess()) {
      LOG(ERROR) << "List object fail, code: " << outcome.error().Code()
                 << ", message: " << outcome.error().Message()
                 << ", requestId: " << outcome.error().RequestId();
      break;
    }
    for (const auto& object : outcome.result().ObjectSummarys()) {
      auto name = object.Key();
      if (suffix.empty() || boost::ends_with(name, suffix)) {
        objects_.push_back(name);
      }
    }
    next_marker = outcome.result().NextMarker();
    IsTruncated = outcome.result().IsTruncated();
  } while (IsTruncated);

  if (objects_.empty()) {
    return Status::IOError("Cannot find object in the specified prefix: " +
                           prefix);
  }
  return Status::OK();
}

Status OSSIOAdaptor::producerRoutine() {
  while (true) {
    size_t index = __sync_fetch_and_add(&current_object_, 1);
    if (index >= objects_.size()) {
      break;
    }
    std::string content;
    auto st = getObjectToBuffer(objects_[index], content);
    if (!st.ok()) {
      LOG(ERROR) << "IOException: OSS Exception: get object failed.";
      return st;
    }
    queue_.Put(std::move(content));
  }
  ++exited_producers_;
  queue_.DecProducerNum();
  return Status::OK();
}

Status OSSIOAdaptor::getObjectToBuffer(const std::string& object_name,
                                       std::string& content) {
  OssClient client(oss_endpoint_, access_id_, access_key_, conf_);
  GetObjectRequest request(bucket_name_, object_name);
  auto outcome = client.GetObject(request);

  if (outcome.isSuccess()) {
    int content_length = outcome.result().Metadata().ContentLength();
    VLOG(2) << "getObjectToBuffer success, Content-Length: " << content_length;
    content.clear();
    content.resize(content_length);
    auto stream = outcome.result().Content();
    for (int i = 0; i < content_length; ++i) {
      stream->get(content[i]);
    }
  } else {
    LOG(ERROR) << "getObjectToBuffer fail, code: " << outcome.error().Code()
               << ", message: " << outcome.error().Message()
               << ", requestId: " << outcome.error().RequestId();
    return Status::IOError(outcome.error().Message());
  }
  return Status::OK();
}

Status OSSIOAdaptor::Close() {
  for (auto& thrd : producers_) {
    if (thrd.joinable()) {
      thrd.join();
    }
  }
  producers_.clear();
  return Status::OK();
}

Status OSSIOAdaptor::SetPartialRead(int index, int total_parts) {
  if (opened_) {
    LOG(WARNING) << "WARNING!! Set partial read after open have no effect,"
                    "You probably want to set partial before open!";
    return Status::Invalid("Set partial read after open have no effect.");
  }
  part_id_ = index;
  part_num_ = total_parts;
  partial_read_ = true;
  return Status::OK();
}

bool OSSIOAdaptor::IsExist(const std::string& path) {
  OssClient client(oss_endpoint_, access_id_, access_key_, conf_);
  return client.DoesObjectExist(bucket_name_, prefix_);
}
}  // namespace vineyard

#endif  // OSS_ENABLED
