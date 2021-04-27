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

#ifdef KAFKA_ENABLED

#include "io/io/kafka_io_adaptor.h"

#include <iosfwd>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "librdkafka/rdkafka.h"
#include "librdkafka/rdkafkacpp.h"

namespace vineyard {

KafkaIOAdaptor::KafkaIOAdaptor(const std::string& location) {
  LOG(INFO) << "Parse location here";
  parseLocation(location);
}

KafkaIOAdaptor::~KafkaIOAdaptor() {}

std::unique_ptr<IIOAdaptor> KafkaIOAdaptor::Make(const std::string& location,
                                                 Client* client) {
  // use `registered` to avoid it being optimized out.
  VLOG(999) << "Kafka IO adaptor has been registered: " << registered_;
  return std::unique_ptr<IIOAdaptor>(new KafkaIOAdaptor(location));
}

Status KafkaIOAdaptor::Open() {
  if (partial_read_) {
    local_partition_num_ = partition_num_ / total_parts_;
    if (partition_num_ % total_parts_ > partial_index_) {
      local_partition_num_++;
    }
  } else {
    local_partition_num_ = partition_num_;
    group_id_ = group_id_ + std::to_string(partial_index_);
  }
  batch_size_per_partition_ = batch_size_ / local_partition_num_;
  RdKafka::Conf* conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
  std::string rdkafka_err;
  if (conf->set("metadata.broker.list", brokers_, rdkafka_err) !=
      RdKafka::Conf::CONF_OK) {
    LOG(WARNING) << "Failed to set metadata.broker.list: " << rdkafka_err;
  }
  if (conf->set("group.id", group_id_, rdkafka_err) != RdKafka::Conf::CONF_OK) {
    LOG(WARNING) << "Failed to set group.id: " << rdkafka_err;
  }
  if (conf->set("enable.auto.commit", "false", rdkafka_err) !=
      RdKafka::Conf::CONF_OK) {
    LOG(WARNING) << "Failed to set enable.auto.commit: " << rdkafka_err;
  }
  if (conf->set("auto.offset.reset", "earliest", rdkafka_err) !=
      RdKafka::Conf::CONF_OK) {
    LOG(WARNING) << "Failed to set auto.offset.reset: " << rdkafka_err;
  }

  message_queue_.resize(local_partition_num_);
  for (int i = 0; i < local_partition_num_; ++i) {
    consumer_ptrs_[i] = std::shared_ptr<RdKafka::KafkaConsumer>(
        RdKafka::KafkaConsumer::create(conf, rdkafka_err));
    if (!consumer_ptrs_[i]) {
      LOG(ERROR) << "Failed to create rdkafka consumer for partition "
                 << rdkafka_err;
    }
    std::vector<RdKafka::TopicPartition*> topic_partitions;
    RdKafka::TopicPartition* topic_partition;
    if (partial_read_) {
      topic_partition = RdKafka::TopicPartition::create(
          topic_, partial_index_ + i * total_parts_);
    } else {
      topic_partition = RdKafka::TopicPartition::create(topic_, i);
    }
    consumer_ptrs_[i]->assign({topic_partition});
    delete topic_partition;
    topic_partition = nullptr;
    consumer_ptrs_[i]->subscribe({topic_});

    message_queue_[i] =
        std::make_shared<PCBlockingQueue<std::vector<std::string>>>();
    message_queue_[i]->SetLimit(16);
    message_queue_[i]->SetProducerNum(1);
  }
  delete conf;  // release the memory resource
  startFetch();

  return Status::OK();
}

// if mode has 'w', then create kafka producer.
// else create kafka consumer.
// A KafkaIOAdaptor can not be both producer and consumer.
Status KafkaIOAdaptor::Open(const char* mode) {
  if (strchr(mode, 'w') != NULL) {
    consumer_ = false;
    RdKafka::Conf* conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
    std::string rdkafka_err;
    if (conf->set("metadata.broker.list", brokers_, rdkafka_err) !=
        RdKafka::Conf::CONF_OK) {
      LOG(WARNING) << "Failed to set metadata.broker.list: " << rdkafka_err;
    }
    // for producer's internal queue.
    if (conf->set("queue.buffering.max.messages",
                  std::to_string(internal_buffer_size_),
                  rdkafka_err) != RdKafka::Conf::CONF_OK) {
      LOG(WARNING) << "Failed to set queue.buffering.max.messages: "
                   << rdkafka_err;
    }

    producer_ = std::unique_ptr<RdKafka::Producer>(
        RdKafka::Producer::create(conf, rdkafka_err));
    if (!producer_) {
      LOG(ERROR) << "Failed to create kafka producer: " << rdkafka_err;
    }
    delete conf;  // release the memory resource
    return Status::OK();
  } else {
    consumer_ = true;
    return Open();
  }
}

Status KafkaIOAdaptor::Configure(const std::string& key,
                                 const std::string& value) {
  if (key == "group_id") {
    group_id_ = value;
  } else if (key == "batch_size") {
    batch_size_ = std::stoi(value);
    batch_size_per_partition_ = batch_size_ / local_partition_num_;
  } else if (key == "time_interval") {
    time_interval_ms_ = std::stoi(value) * 1000;
  }
  return Status::OK();
}

Status KafkaIOAdaptor::SetPartialRead(const int index, const int total_parts) {
  partial_read_ = true;
  partial_index_ = index;
  total_parts_ = total_parts;
  return Status::OK();
}

Status KafkaIOAdaptor::ReadLine(std::string& line) {
  if (message_offset_ < message_list_.size()) {
    line = message_list_[message_offset_];
    ++message_offset_;
    return Status::OK();
  } else {
    message_list_.clear();
    message_offset_ = 0;
    bool end = false;
    while (!end) {
      end = true;
      for (int i = 0; i < local_partition_num_; ++i) {
        end = end & message_queue_[i]->End();
        if (message_queue_[i]->Size()) {
          message_queue_[i]->Get(message_list_);
          break;
        }
      }
      if (message_list_.size()) {
        line = message_list_[message_offset_];
        ++message_offset_;
        return Status::OK();
      }
    }
    return Status::EndOfFile();
  }
}

Status KafkaIOAdaptor::WriteLine(const std::string& line) {
  if (line.empty()) {
    return Status::OK();
  }
  RdKafka::ErrorCode err = producer_->produce(
      topic_, RdKafka::Topic::PARTITION_UA, RdKafka::Producer::RK_MSG_COPY,
      static_cast<void*>(const_cast<char*>(line.c_str())) /* value */,
      line.size() /* size */, NULL, 0, 0 /* timestamp */,
      NULL /* delivery report */);
  if (err != RdKafka::ERR_NO_ERROR) {
    LOG(ERROR) << "Failed to writeline to kafka: " << RdKafka::err2str(err);
  }
  producer_->flush(time_interval_ms_);
  return Status::OK();
}

Status KafkaIOAdaptor::Write(void* buffer, size_t size) {
  {
    if (size == 0) {
      return Status::OK();
    }
    RdKafka::ErrorCode err = producer_->produce(
        topic_, RdKafka::Topic::PARTITION_UA, RdKafka::Producer::RK_MSG_COPY,
        buffer /* value */, size /* size */, NULL, 0, 0 /* timestamp */,
        NULL /* delivery report */);
    if (err != RdKafka::ERR_NO_ERROR) {
      LOG(ERROR) << "Failed to output to kafka: " << RdKafka::err2str(err);
    }
    producer_->flush(time_interval_ms_);
  }
  return Status::OK();
}

Status KafkaIOAdaptor::Close() { return Status::OK(); }

void KafkaIOAdaptor::parseLocation(const std::string& location) {
  std::string tmp_location(location);
  std::replace(tmp_location.begin(), tmp_location.end(), ';', ',');
  std::vector<std::string> kafka_params;
  std::string::size_type pos, last_pos = 0, length = tmp_location.length();
  while (last_pos < length) {
    pos = tmp_location.find_first_of("/", last_pos);
    if (pos == std::string::npos) {
      pos = length;
    }
    if (pos != last_pos) {
      std::string v;
      v = std::string(tmp_location.data() + last_pos, pos - last_pos);
      kafka_params.emplace_back(v);
    }
    last_pos = pos + 1;
  }
  brokers_ = kafka_params[1];
  topic_ = kafka_params[2];
  group_id_ = kafka_params[3];
  partition_num_ = std::stoi(kafka_params[4]);
}

void KafkaIOAdaptor::startFetch() {
  for (int i = 0; i < local_partition_num_; ++i) {
    std::thread t = std::thread([&, i] {
      while (!message_queue_[i]->End()) {
        std::vector<std::string> msg;
        fetchMessage(i, msg);
        message_queue_[i]->Put(std::move(msg));
      }
    });
    t.detach();
    LOG(INFO) << "[partition" << partial_index_
              << "] start fetch thread on partition " << i;
  }
}

void KafkaIOAdaptor::fetchMessage(int partition_index,
                                  std::vector<std::string>& messages) {
  messages.reserve(batch_size_per_partition_);
  // Create a consumer dispatcher
  auto consumer_ptr_ = consumer_ptrs_[partition_index];

  int msg_cnt = 0;
  int64_t first_msg_ts = 0, cur_msg_ts = 0;

  int msg_len;
  const char* msg_payload;
  int64_t timestamp;
  std::string msg_data;

  auto process = [&](int partition_index, RdKafka::Message* message) -> bool {
    switch (message->err()) {
    case RdKafka::ERR__TIMED_OUT:
      if (msg_cnt) {
        msg_cnt = 0;
        first_msg_ts = 0;
        return false;
      }
      message_queue_[partition_index]->DecProducerNum();
      return true;

    case RdKafka::ERR_NO_ERROR:
      /* process message */
      msg_len = message->len();
      msg_payload = static_cast<char*>(message->payload());
      timestamp = message->timestamp().timestamp;

      msg_data = std::string(msg_payload, msg_len);
      if (!msg_data.empty()) {
        messages.push_back(msg_data);
        ++msg_cnt;
      }
      if (!first_msg_ts) {
        first_msg_ts = timestamp;
      }
      cur_msg_ts = timestamp;
      if (msg_cnt >= this->batch_size_per_partition_ ||
          cur_msg_ts - first_msg_ts > this->time_interval_ms_) {
        msg_cnt = 0;
        first_msg_ts = 0;
        return false;
      } else {
        return true;
      }

    case RdKafka::ERR__PARTITION_EOF:
      LOG(ERROR) << "Reached EOF on partition";
      message_queue_[partition_index]->DecProducerNum();
      return true;

    case RdKafka::ERR__UNKNOWN_TOPIC:
    case RdKafka::ERR__UNKNOWN_PARTITION:
      LOG(ERROR) << "Topic or partition error: " << message->errstr();
      message_queue_[partition_index]->DecProducerNum();
      return true;

    default:
      LOG(ERROR) << "Unhandled kafka error: " << message->errstr();
      return false;
    }
  };

  while (true) {
    RdKafka::Message* message = consumer_ptr_->consume(time_interval_ms_);
    bool cont = process(partition_index, message);
    delete message;
    if (!cont) {
      break;
    }
  }
}

const bool KafkaIOAdaptor::registered_ = IOFactory::Register(
    "kafka", static_cast<IOFactory::io_initializer_t>(&KafkaIOAdaptor::Make));

}  // namespace vineyard

#endif  // KAFKA_ENABLED
