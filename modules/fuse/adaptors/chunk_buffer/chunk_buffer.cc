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
#include "modules/fuse/adaptors/chunk_buffer/chunk_buffer.h"
#include <algorithm>
#include <memory>
#include "arrow/buffer.h"
#include "common/util/logging.h"
#include "iostream"

namespace vineyard {
namespace fuse {
namespace internal {
ChunkBuffer::ChunkBuffer() : size_(0), position_(0), open(true) {}
ChunkBuffer::~ChunkBuffer() {}
arrow::Status ChunkBuffer::Close() {
  this->open = false;
  return arrow::Status::OK();
}
arrow::Status ChunkBuffer::Abort() {
  this->chunks.~map();
  this->open = false;
  this->size_ = 0;
  return arrow::Status::OK();
}
arrow::Status ChunkBuffer::Flush() { return arrow::Status::OK(); }
arrow::Result<int64_t> ChunkBuffer::Tell() const {
  DLOG(INFO) << "Write(const std::shared_ptr<arrow::Buffer>& data):"
             << this->position_;

  return this->position_;
}
bool ChunkBuffer::closed() const { return !this->open; }
// in arrow::ipc::MakeFileWriter, this function is often used to fill in the
// padding bytes, so the overhead should be small
arrow::Status ChunkBuffer::Write(const void* data, int64_t nbytes) {
  DLOG(INFO) << "Write(const void* data, int64_t nbytes):" << nbytes;
  if (nbytes == 0) {
    return arrow::Status::OK();
  }
  void* copy = malloc(nbytes);
  memcpy(copy, data, nbytes);

  auto chunk = arrow::Buffer::Wrap(reinterpret_cast<uint8_t*>(copy), nbytes);
  std::pair<int64_t, int64_t> id = {size_, size_ + nbytes - 1};
  this->chunks.emplace(id, chunk);
  this->size_ += nbytes;
  this->position_ = this->size_;
  return arrow::Status::OK();
}
arrow::Status ChunkBuffer::Write(const std::shared_ptr<arrow::Buffer>& data) {
  DLOG(INFO) << "Write(const std::shared_ptr<arrow::Buffer>& data):"
             << data->size();

  auto chunk_size = data->size();
  std::pair<int64_t, int64_t> id = {size_, size_ + chunk_size - 1};
  this->chunks.emplace(id, data);
  this->size_ += chunk_size;
  this->position_ = this->size_;
  return arrow::Status::OK();
}
template <typename C>
typename C::const_iterator find_key_less_equal(
    const C& c, const typename C::key_type& key) {
  auto iter = c.upper_bound(key);
  if (iter == c.cbegin())
    return c.cend();
  return --iter;
}
int64_t ChunkBuffer::readAt(int64_t position, int64_t nbytes, void* out) {
  if (nbytes == 0) {
    return 0;
  }
  auto it = find_key_less_equal(this->chunks, {position, INT64_MAX});

  if (it == this->chunks.cend()) {
    return 0;
  } else {
    if (position + nbytes > this->size_) {
      nbytes = (this->size_ - 1) - position + 1;
    }
    auto byteout = reinterpret_cast<uint8_t*>(out);
    int64_t remaining = nbytes;
    {
      auto chunk_start = it->first.first;
      auto chunk_end = it->first.second;
      // load the first buffer
      auto chunk = it->second;
      auto chunk_p = chunk->data();
      chunk_p += position - chunk_start;
      auto chunk_copied_len = chunk_end - position + 1;
      memcpy(byteout, chunk_p, chunk_copied_len);
      byteout += chunk_copied_len;
      remaining -= chunk_copied_len;
      it++;
    }

    while (remaining > 0 && it != this->chunks.end()) {
      auto chunk = it->second;
      auto chunk_p = chunk->data();
      auto readByte = std::min(remaining, chunk->size());
      memcpy(byteout, chunk_p, readByte);
      byteout += readByte;

      remaining -= readByte;
      it++;
    }

    return nbytes;
  }
}
int64_t ChunkBuffer::size() const { return this->size_; }

}  // namespace internal

}  // namespace fuse

}  // namespace vineyard
