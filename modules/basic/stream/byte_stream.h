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

#ifndef MODULES_BASIC_STREAM_BYTE_STREAM_H_
#define MODULES_BASIC_STREAM_BYTE_STREAM_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "arrow/builder.h"
#include "arrow/status.h"

#include "basic/ds/arrow_utils.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "client/ds/stream.h"
#include "common/util/uuid.h"

namespace vineyard {

class ByteStream : public BareRegistered<ByteStream>, public Stream<Blob> {
 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<ByteStream>{new ByteStream()});
  }

  void SetBufferSizeLimit(size_t limit) { buffer_size_limit_ = limit; }

  Status WriteBytes(const char* ptr, size_t len);

  Status WriteLine(const std::string& line);

  Status FlushBuffer();

  Status ReadLine(std::string& line);

 protected:
  size_t buffer_size_limit_ = 1024 * 1024 * 256;  // 256Mi

  arrow::BufferBuilder builder_;  // for write
  std::stringstream ss_;          // for read
};

template <>
struct stream_type<Blob> {
  using type = ByteStream;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_STREAM_BYTE_STREAM_H_
