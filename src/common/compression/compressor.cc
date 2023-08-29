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

#include "common/compression/compressor.h"

#include <algorithm>
#include <string>

#include "zstd/lib/zstd.h"

namespace vineyard {

// return the status if the boost::system:error_code is not OK.
#ifndef RETURN_ON_ZSTD_ERROR
#define RETURN_ON_ZSTD_ERROR(expr, message)                                \
  do {                                                                     \
    auto _ret = (expr);                                                    \
    if (ZSTD_isError(_ret)) {                                              \
      return Status::IOError(std::string("Error in zstd in '") + message + \
                             "'" + ZSTD_getErrorName(_ret));               \
    }                                                                      \
  } while (0)
#endif  // RETURN_ON_ZSTD_ERROR

Compressor::Compressor() {
  stream = ZSTD_createCStream();
  in_size_ = ZSTD_CStreamInSize();
  out_size_ = ZSTD_CStreamOutSize();
  accumulated_ = 0;
  input_ = new ZSTD_inBuffer{nullptr, 0, 0};
  output_ = new ZSTD_outBuffer{malloc(out_size_), out_size_, 0};
}

Compressor::~Compressor() {
  if (stream) {
    if (input_->src != nullptr) {
      ZSTD_compressStream2(stream, output_, input_,
                           ZSTD_EndDirective::ZSTD_e_end);
    }
    ZSTD_freeCStream(stream);
    if (output_->dst) {
      free(output_->dst);
      output_->dst = nullptr;
    }
    stream = nullptr;
  }
  if (input_) {
    delete input_;
    input_ = nullptr;
  }
  if (output_) {
    delete output_;
    output_ = nullptr;
  }
}

Status Compressor::Compress(const void* data, const size_t size) {
  if (!finished_) {
    return Status::Invalid("Compressor: the zstd stream is not finished yet");
  }
  *input_ = ZSTD_inBuffer{data, size, 0};
  finished_ = false;
  return Status::OK();
}

Status Compressor::Pull(void*& data, size_t& size) {
  if (finished_ && !flushing_) {  // finished and nothing to flush
    size = 0;
    return Status::StreamDrained();
  }

  // reset output pointer
  output_->pos = 0;

  // if reach a flush point, flush util empty to make the decompressor
  // can start work and avoid much memory consumption.
  if (accumulated_ >= maximum_accumulated_bytes) {
    flushing_ = true;
    accumulated_ = 0;
  }
  if (flushing_) {
    size_t ret = ZSTD_compressStream2(stream, output_, input_,
                                      ZSTD_EndDirective::ZSTD_e_flush);
    RETURN_ON_ZSTD_ERROR(ret, "ZSTD compress flush");
    if (ret == 0) {  // stop flushing
      flushing_ = false;
    }
    if (output_->pos > 0) {  // maybe nothing flushed
      data = output_->dst;
      size = output_->pos;
      return Status::OK();
    }
    if (finished_) {
      size = 0;
      return Status::OK();
    }
  }

  // if not a flush point, but reaching the end
  if (input_->pos >= input_->size) {
    flushing_ = true;
    finished_ = true;  // finishing this block
    // starting flushing
    return Pull(data, size);
  }

  // continue to compress
  size_t ret = ZSTD_compressStream2(stream, output_, input_,
                                    ZSTD_EndDirective::ZSTD_e_continue);
  RETURN_ON_ZSTD_ERROR(ret, "ZSTD compress continue");

  data = output_->dst;
  size = output_->pos;
  accumulated_ += size;  // update accumulate statistics
  return Status::OK();
}

Decompressor::Decompressor() {
  stream = ZSTD_createDStream();
  in_size_ = std::max(ZSTD_CStreamOutSize(), ZSTD_DStreamInSize());
  out_size_ = ZSTD_DStreamOutSize();
  input_ = new ZSTD_inBuffer{malloc(in_size_), in_size_, 0};
  output_ = new ZSTD_outBuffer{nullptr, 0, 0};
}

Decompressor::~Decompressor() {
  if (stream) {
    ZSTD_freeDStream(stream);
    if (input_ && input_->src) {
      free(const_cast<void*>(input_->src));
      input_->src = nullptr;
    }
    stream = nullptr;
  }
  if (input_) {
    delete input_;
    input_ = nullptr;
  }
  if (output_) {
    delete output_;
    output_ = nullptr;
  }
}

Status Decompressor::Buffer(void*& data, size_t& size) {
  if (!finished_) {
    return Status::Invalid(
        "Decompressor: the zstd stream is not finished yet, the next input "
        "cannot be fed");
  }
  data = const_cast<void*>(input_->src);
  size = input_->size;
  return Status::OK();
}

Status Decompressor::Decompress(const size_t size) {
  if (!finished_) {
    // pull again, if the returned size is zero, mark is as finished
    char buffer[1024];
    size_t size = 0;
    auto s = this->Pull(buffer, 1024, size);
    if (!(s.IsStreamDrained() || size == 0)) {
      // indicates an error, as there's error, the consumed 1 byte doesn't
      // matter
      return Status::Invalid(
          "Decompressor: the zstd stream is not finished yet, new decompress "
          "process cannot be started");
    }
  }
  input_->size = size;
  input_->pos = 0;
  finished_ = false;
  return Status::OK();
}

Status Decompressor::Pull(void* data, const size_t capacity, size_t& size) {
  if (capacity == 0) {
    size = 0;
    return Status::OK();
  }
  if (finished_) {
    size = 0;
    return Status::StreamDrained();
  }

  // reset output pointer
  *output_ = ZSTD_outBuffer{data, capacity, 0};

  // decompress
  size_t ret = ZSTD_decompressStream(stream, output_, input_);
  RETURN_ON_ZSTD_ERROR(ret, "ZSTD decompress");
  size = output_->pos;
  if (size == 0) {
    finished_ = true;
    // reset the input buffer
    input_->size = in_size_;
    return Status::StreamDrained();
  }
  return Status::OK();
}

}  // namespace vineyard
