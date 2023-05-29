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

#ifndef SRC_COMMON_COMPRESSION_COMPRESSOR_H_
#define SRC_COMMON_COMPRESSION_COMPRESSOR_H_

#include <memory>
#include <string>

#include "common/util/status.h"

// forward declaration to avoid including zstd.h
struct ZSTD_inBuffer_s;
struct ZSTD_outBuffer_s;
struct ZSTD_CCtx_s;
struct ZSTD_DCtx_s;

namespace vineyard {

/**
 * Usage of the ZSTD compressor:
 *
 *  auto compressor = Compressor();
 *  compressor.Compress(data, data_size);
 *
 *  void *chunk;
 *  size_t size;
 *  while (compressor.Pull(chunk, size).ok()) {
 *      // do something with resulted chunk and size
 *  }
 */
class Compressor {
 public:
  /**
   * Whether to set the first word as the buffer length in `Pull`.
   */
  Compressor();
  ~Compressor();

  size_t input_size() const { return in_size_; }

  size_t output_size() const { return out_size_; }

  bool Finished() const { return finished_; }

  Status Compress(const void* data, const size_t size);

  Status Pull(void*& data, size_t& size);

 private:
  const size_t maximum_accumulated_bytes = 64 * 1024 * 1024;  // 64MB

  size_t in_size_, out_size_, accumulated_;
  bool finished_ = true, flushing_ = false;
  struct ZSTD_inBuffer_s* input_ = nullptr;
  struct ZSTD_outBuffer_s* output_ = nullptr;
  ZSTD_CCtx_s* stream = nullptr;
};

/**
 * Usage of the ZSTD decompressor:
 *
 *  auto decompressor = Decompressor();
 *
 *  void *data;
 *  size_t data_size;
 *  decompressor.Buffer(data, data_size);
 *
 *  // fill data with compressed data ...
 *  size_t actual_data_size = ...;
 *
 *  decompressor.Decompress(actual_data_size);
 *
 *  void *chunk;
 *  size_t size;
 *  while (decompressor.Pull(chunk, chunk_capacity, size)) {
 *      // do something with resulted chunk and size
 *  }
 */
class Decompressor {
 public:
  Decompressor();
  ~Decompressor();

  size_t input_size() const { return in_size_; }

  size_t output_size() const { return out_size_; }

  // expose the buffer
  Status Buffer(void*& data, size_t& size);

  Status Decompress(const size_t size);

  Status Pull(void* data, const size_t capacity, size_t& size);

 private:
  const size_t maximum_accumulated_bytes = 64 * 1024 * 1024;  // 64MB

  size_t in_size_, out_size_;
  bool finished_ = true;
  struct ZSTD_inBuffer_s* input_ = nullptr;
  struct ZSTD_outBuffer_s* output_ = nullptr;
  ZSTD_DCtx_s* stream = nullptr;
};

}  // namespace vineyard

#endif  // SRC_COMMON_COMPRESSION_COMPRESSOR_H_
