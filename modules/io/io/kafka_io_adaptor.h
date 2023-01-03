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

#ifndef MODULES_IO_IO_KAFKA_IO_ADAPTOR_H_
#define MODULES_IO_IO_KAFKA_IO_ADAPTOR_H_

#ifdef KAFKA_ENABLED

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "librdkafka/rdkafka.h"
#include "librdkafka/rdkafkacpp.h"

#include "common/util/blocking_queue.h"
#include "common/util/functions.h"
#include "common/util/status.h"
#include "io/io/i_io_adaptor.h"
#include "io/io/io_factory.h"

namespace vineyard {

class KafkaIOAdaptor : public IIOAdaptor {
 public:
  /** Constructor.
   * @param location the location containing brokers, topic, group_id and
   * partition_num of kafka.
   */
  explicit KafkaIOAdaptor(const std::string& location);

  /** Default destructor. */
  ~KafkaIOAdaptor();

  static std::unique_ptr<IIOAdaptor> Make(const std::string& location,
                                          Client* client);

  Status Open() override;

  Status Open(const char* mode) override;

  Status Close() override;

  Status ReadLine(std::string& line) override;

  Status SetPartialRead(const int index, const int total_parts) override;

  Status GetPartialReadDetail(int64_t& offset, int64_t& nbytes) {
    return Status::NotImplemented();
  }

  Status Configure(const std::string& key, const std::string& value) override;

  Status WriteLine(const std::string& line) override;

  Status Read(void* buffer, size_t size) override {
    return Status::NotImplemented();
  }

  Status Write(void* buffer, size_t size) override;

  Status ListDirectory(const std::string& path,
                       std::vector<std::string>& files) override {
    return Status::NotImplemented();
  }

  Status MakeDirectory(const std::string& path) override {
    return Status::NotImplemented();
  }

  bool IsExist(const std::string& path) override { return true; }

 private:
  void parseLocation(const std::string& location);

  void startFetch();

  void fetchMessage(int partition, std::vector<std::string>& messages);

  static const constexpr int internal_buffer_size_ = 1024 * 1024;

  bool consumer_;
  int batch_size_ = 50;
  int partition_num_;
  int local_partition_num_;
  int batch_size_per_partition_;
  int time_interval_ms_ = 1000 * 10;
  size_t message_offset_ = 0;

  bool partial_read_ = false;
  int partial_index_;
  int total_parts_;

  template <typename T>
  using mq_t = std::shared_ptr<PCBlockingQueue<std::vector<T>>>;
  std::vector<mq_t<std::string>> message_queue_;
  std::vector<std::string> message_list_;
  std::string group_id_;
  std::string brokers_;
  std::string topic_;
  std::unique_ptr<RdKafka::Producer> producer_;
  std::map<int, std::shared_ptr<RdKafka::KafkaConsumer>> consumer_ptrs_;

  // register
  static const bool registered_;
};
}  // namespace vineyard

#endif  // KAFKA_ENABLED
#endif  // MODULES_IO_IO_KAFKA_IO_ADAPTOR_H_
