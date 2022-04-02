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

#ifndef MODULES_MSGPACK_VBUFFER_H_
#define MODULES_MSGPACK_VBUFFER_H_

#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "basic/ds/array.vineyard.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/logging.h"
#include "common/util/uuid.h"

namespace vineyard {

class Buffer {
 public:
  Buffer(uint8_t* buffer, size_t size, bool owned = false)
      : buffer_(buffer), size_(size), owned_(owned) {}

  Buffer(const uint8_t* buffer, size_t size, bool owned = false)
      : buffer_(const_cast<uint8_t*>(buffer)), size_(size), owned_(owned) {}

  explicit Buffer(std::string const& buffer) {
    buffer_ = static_cast<uint8_t*>(malloc(buffer.size()));
    size_ = buffer.size();
    owned_ = true;
    memcpy(buffer_, buffer.data(), buffer.size());
  }

  template <typename T>
  explicit Buffer(std::vector<T> const& buffer) {
    size_t elements_size = buffer.size() * sizeof(T);
    buffer_ = static_cast<uint8_t*>(malloc(elements_size));
    size_ = elements_size;
    owned_ = true;
    memcpy(buffer_, buffer.data(), elements_size);
  }

  Buffer(const Buffer& other) {
    buffer_ = other.buffer_;
    size_ = other.size_;
    owned_ = false;
  }

  Buffer(Buffer&& other) {
    buffer_ = other.buffer_;
    size_ = other.size_;
    if (other.owned_) {
      owned_ = true;
    }
    other.buffer_ = nullptr;
    other.size_ = 0;
    other.owned_ = false;
  }

  Buffer& operator=(const Buffer& other) {
    buffer_ = other.buffer_;
    size_ = other.size_;
    owned_ = false;
    return *this;
  }

  Buffer& operator=(Buffer&& other) {
    buffer_ = other.buffer_;
    size_ = other.size_;
    if (other.owned_) {
      owned_ = true;
    }
    other.buffer_ = nullptr;
    other.size_ = 0;
    other.owned_ = false;
    return *this;
  }

  ~Buffer() {
    if (owned_ && buffer_ != nullptr) {
      free(buffer_);
      buffer_ = nullptr;
    }
  }

  size_t size() const { return size_; }

  uint8_t* ptr() const { return buffer_; }

  std::string ToString() const {
    static const char cs[] = {'0', '1', '2', '3', '4', '5', '6', '7',
                              '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    std::ostringstream ss;
    for (size_t index = 0; index < size_; ++index) {
      ss << "\\x" << cs[static_cast<size_t>(buffer_[index] >> 4)]
         << cs[static_cast<size_t>(buffer_[index] & 0xF)];
    }
    return ss.str();
  }

 private:
  uint8_t* buffer_ = nullptr;
  size_t size_;
  bool owned_;
};

// little endian
class BufferBuilder {
 private:
  union bs32_t {
    uint8_t c[4];
    int32_t i;
    uint32_t u;
    float d;
  };

  union bs64_t {
    uint8_t c[8];
    int64_t i;
    uint64_t u;
    double d;
  };

 public:
  BufferBuilder() {}

  void PutByte(uint8_t value) { stream_.put(value); }

  void PutChars(char const* cbytes) {
    stream_.write(cbytes, std::strlen(cbytes));
  }

  void PutChars(char const* cbytes, size_t count) {
    stream_.write(cbytes, count);
  }

  void PutBytes(const uint8_t* bytes) {
    char const* cbytes = reinterpret_cast<const char*>(bytes);
    stream_.write(cbytes, std::strlen(cbytes));
  }

  void PutBytes(const uint8_t* bytes, size_t count) {
    char const* cbytes = reinterpret_cast<const char*>(bytes);
    stream_.write(cbytes, count);
  }

  void PutInt32(int32_t value) {
    bs32_t bs;
    bs.i = value;
    putBS32(bs);
  }

  void PutUInt32(uint32_t value) {
    bs32_t bs;
    bs.u = value;
    putBS32(bs);
  }

  void PutInt64(int64_t value) {
    bs64_t bs;
    bs.i = value;
    putBS64(bs);
  }

  void PutUInt64(uint64_t value) {
    bs64_t bs;
    bs.u = value;
    putBS64(bs);
  }

  void Append(Buffer const& buffer) { PutBytes(buffer.ptr(), buffer.size()); }

  void Append(BufferBuilder& builder) {
    auto const buffer = builder.Finish();
    PutBytes(buffer.ptr(), buffer.size());
  }

  Buffer Finish() { return Buffer(stream_.str()); }

 private:
  std::ostringstream stream_;

  void putBS32(bs32_t const& bs) {
    for (size_t i = 0; i < 4; ++i) {
      PutByte(bs.c[i]);
    }
  }

  void putBS64(bs64_t const& bs) {
    for (size_t i = 0; i < 8; ++i) {
      PutByte(bs.c[i]);
    }
  }
};

class VBuffer {
 public:
  VBuffer() { offsets_.emplace_back(0); }

  void Append(Buffer const& buffer) {
    buffers_.emplace_back(buffer);
    size_ += buffer.size();
    offsets_.emplace_back(size_);
  }

 private:
  std::vector<size_t> offsets_;
  std::vector<Buffer> buffers_;
  size_t size_ = 0;
};

}  // namespace vineyard

#endif  // MODULES_MSGPACK_VBUFFER_H_
