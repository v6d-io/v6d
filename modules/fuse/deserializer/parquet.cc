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

#include "fuse/adaptors/parquet.h"

#if defined(WITH_PARQUET)

#include <limits>
#include <memory>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "parquet/api/reader.h"
#include "parquet/api/schema.h"
#include "parquet/api/writer.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/schema.h"
#include "parquet/arrow/writer.h"

namespace vineyard {
namespace fuse {

std::shared_ptr<arrow::Buffer> parquet_view(
    std::shared_ptr<vineyard::DataFrame>& df) {
  // Add writer properties
  ::parquet::WriterProperties::Builder builder;
  builder.encoding(::parquet::Encoding::PLAIN);
  builder.disable_dictionary();
  builder.compression(::parquet::Compression::UNCOMPRESSED);
  builder.disable_statistics();
  builder.write_batch_size(std::numeric_limits<size_t>::max());
  builder.max_row_group_length(std::numeric_limits<size_t>::max());
  std::shared_ptr<::parquet::WriterProperties> props = builder.build();

  auto batch = df->AsBatch(false);
  std::shared_ptr<arrow::Table> table;
  VINEYARD_CHECK_OK(RecordBatchesToTable({batch}, &table));
  std::shared_ptr<arrow::io::BufferOutputStream> sink;
  CHECK_ARROW_ERROR_AND_ASSIGN(sink, arrow::io::BufferOutputStream::Create());
  ::parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink,
                               std::numeric_limits<size_t>::max(), props);
  std::shared_ptr<arrow::Buffer> buffer;
  CHECK_ARROW_ERROR_AND_ASSIGN(buffer, sink->Finish());
  return buffer;
}

}  // namespace fuse
}  // namespace vineyard

#endif
