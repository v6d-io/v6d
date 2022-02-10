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

Status ByteStreamWriter::GetNext(
    size_t const size, std::unique_ptr<arrow::MutableBuffer>& buffer) {
  return client_.GetNextStreamChunk(id_, size, buffer);
}

Status ByteStreamWriter::Abort() {
  if (stoped_) {
    return Status::OK();
  }
  stoped_ = true;
  return client_.StopStream(id_, true);
}

Status ByteStreamWriter::Finish() {
  if (stoped_) {
    return Status::OK();
  }
  RETURN_ON_ERROR(flushBuffer());
  stoped_ = true;
  return client_.StopStream(id_, false);
}

Status ByteStreamWriter::WriteBytes(const char* ptr, size_t len) {
  if (builder_.length() + len > buffer_size_limit_) {
    RETURN_ON_ERROR(flushBuffer());
  }
  RETURN_ON_ARROW_ERROR(builder_.Append(ptr, len));
  return Status::OK();
}

Status ByteStreamWriter::WriteLine(const std::string& line) {
  if (builder_.length() + line.length() > buffer_size_limit_) {
    RETURN_ON_ERROR(flushBuffer());
  }
  RETURN_ON_ARROW_ERROR(builder_.Append(line.c_str(), line.size()));
  return Status::OK();
}

Status ByteStreamWriter::flushBuffer() {
  std::shared_ptr<arrow::Buffer> buf;
  RETURN_ON_ARROW_ERROR(builder_.Finish(&buf));
  std::unique_ptr<arrow::MutableBuffer> mb;
  if (buf->size() > 0) {
    RETURN_ON_ERROR(GetNext(buf->size(), mb));
    memcpy(mb->mutable_data(), buf->data(), buf->size());
  }
  return Status::OK();
}

Status ByteStreamReader::GetNext(std::unique_ptr<arrow::Buffer>& buffer) {
  return client_.PullNextStreamChunk(id_, buffer);
}

Status ByteStreamReader::ReadLine(std::string& line) {
  if (std::getline(ss_, line)) {
    return Status::OK();
  }

  std::unique_ptr<arrow::Buffer> buffer;
  if (!GetNext(buffer).ok()) {
    return Status::EndOfFile();
  }
  std::string buf_str = std::string(
      reinterpret_cast<const char*>(buffer->data()), buffer->size());
  ss_.str(buf_str);
  std::getline(ss_, line);

  return Status::OK();
}

Status ByteStream::OpenReader(Client& client,
                              std::unique_ptr<ByteStreamReader>& reader) {
  RETURN_ON_ERROR(client.OpenStream(id_, StreamOpenMode::read));
  reader = std::unique_ptr<ByteStreamReader>(
      new ByteStreamReader(client, id_, meta_));
  return Status::OK();
}

Status ByteStream::OpenWriter(Client& client,
                              std::unique_ptr<ByteStreamWriter>& writer) {
  RETURN_ON_ERROR(client.OpenStream(id_, StreamOpenMode::write));
  writer = std::unique_ptr<ByteStreamWriter>(
      new ByteStreamWriter(client, id_, meta_));
  return Status::OK();
}

}  // namespace vineyard
