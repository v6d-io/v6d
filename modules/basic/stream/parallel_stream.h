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
*/

#ifndef MODULES_BASIC_STREAM_PARALLEL_STREAM_H_
#define MODULES_BASIC_STREAM_PARALLEL_STREAM_H_

#include <memory>
#include <string>
#include <vector>

#include "client/client.h"

namespace vineyard {

class ParallelStreamBuilder;

class ParallelStream : public Registered<ParallelStream>, GlobalObject {
 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<ParallelStream>{new ParallelStream()});
  }

  void Construct(const ObjectMeta& meta) override;

  ObjectMeta GetStreamMeta(int index) { return streams_[index]->meta(); }

  size_t GetStreamSize() { return size_; }

  template <typename T>
  std::shared_ptr<T> GetStream(int index) {
    return std::dynamic_pointer_cast<T>(streams_[index]);
  }

  template <typename T>
  std::vector<std::shared_ptr<T>> GetLocalStreams() {
    std::vector<std::shared_ptr<T>> local_streams;
    for (auto const& s : streams_) {
      if (s->IsLocal()) {
        local_streams.emplace_back(std::dynamic_pointer_cast<T>(s));
      }
    }
    return local_streams;
  }

 private:
  size_t size_;
  std::vector<std::shared_ptr<Object>> streams_;

  friend class Client;
  friend class ParallelStreamBuilder;
};

/**
 * @brief ParallelStreamBuilder is desinged for building parallel stremas
 *
 */
class ParallelStreamBuilder : public ObjectBuilder {
 public:
  explicit ParallelStreamBuilder(Client& client) {}

  void AddStream(const ObjectID stream_id);

  Status Build(Client& client) override;

  std::shared_ptr<Object> _Seal(Client& client) override;

 private:
  std::vector<ObjectID> streams_;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_STREAM_PARALLEL_STREAM_H_
