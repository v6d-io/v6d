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

#include "basic/stream/byte_stream.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/arrow_utils.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/uuid.h"

namespace vineyard {

Status ByteStream::WriteBytes(const char* ptr, size_t len) {
  RETURN_ON_ARROW_ERROR(builder_.Append(ptr, len));
  if (builder_.length() + len > buffer_size_limit_) {
    RETURN_ON_ERROR(FlushBuffer());
  }
  return Status::OK();
}

Status ByteStream::WriteLine(const std::string& line) {
  RETURN_ON_ARROW_ERROR(builder_.Append(line.c_str(), line.size()));
  if (builder_.length() + line.length() > buffer_size_limit_) {
    RETURN_ON_ERROR(FlushBuffer());
  }
  return Status::OK();
}

Status ByteStream::FlushBuffer() {
  std::shared_ptr<arrow::Buffer> buf;
  RETURN_ON_ARROW_ERROR(builder_.Finish(&buf));

  if (buf->size() > 0) {
    std::unique_ptr<BlobWriter> buffer;
    RETURN_ON_ERROR(this->client_->CreateBlob(buf->size(), buffer));
    memcpy(buffer->data(), buf->data(), buf->size());
  }
  return Status::OK();
}

Status ByteStream::ReadLine(std::string& line) {
  if (std::getline(ss_, line)) {
    return Status::OK();
  }

  std::shared_ptr<Blob> buffer;
  if (!this->Next(buffer).ok()) {
    return Status::EndOfFile();
  }
  std::string buf_str = std::string(
      reinterpret_cast<const char*>(buffer->data()), buffer->size());
  ss_.str(buf_str);
  std::getline(ss_, line);

  return Status::OK();
}

}  // namespace vineyard
