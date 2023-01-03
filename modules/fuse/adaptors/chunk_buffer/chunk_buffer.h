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
#ifndef MODULES_FUSE_ADAPTORS_CHUNK_BUFFER_CHUNK_BUFFER_H_
#define MODULES_FUSE_ADAPTORS_CHUNK_BUFFER_CHUNK_BUFFER_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include "arrow/io/interfaces.h"
#include "arrow/io/type_fwd.h"
#include "arrow/ipc/writer.h"
#include "arrow/status.h"

namespace vineyard {
namespace fuse {
namespace internal {
// this is an append only file buffer
/*
this implementation of outputstream is only guaranteed to be functional when it
is passed as the sink in MakeFileWriter.
*/
class ChunkBuffer : public arrow::io::OutputStream {
 public:
  ChunkBuffer();
  ~ChunkBuffer() override;
  arrow::Status Close() override;
  arrow::Status Abort() override;
  arrow::Result<int64_t> Tell() const override;
  bool closed() const override;
  arrow::Status Write(const void* data, int64_t nbytes) override;
  arrow::Status Write(const std::shared_ptr<arrow::Buffer>& data) override;
  arrow::Status Flush() override;
  int64_t readAt(int64_t position, int64_t nbytes, void* out);
  int64_t size() const;

 private:
  std::map<std::pair<int64_t, int64_t>, std::shared_ptr<arrow::Buffer>> chunks;
  int64_t size_;
  int64_t position_;
  bool open;
};
}  // namespace internal

}  // namespace fuse

}  // namespace vineyard
#endif  // MODULES_FUSE_ADAPTORS_CHUNK_BUFFER_CHUNK_BUFFER_H_
