#ifndef MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_
#define MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_
#include <stdio.h>

#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/util/env.h"

#include <glog/logging.h>
#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/macros.h"
#include "basic/ds/array.h"
#include "basic/ds/dataframe.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/core_types.h"
#include "client/ds/i_object.h"
#include "common/util/logging.h"
#include "common/util/typename.h"

#include "common/util/uuid.h"

namespace vineyard {
namespace fuse {
using vineyard_deserializer_nt = std::shared_ptr<arrow::Buffer> (*)(
    const std::shared_ptr<vineyard::Object>&);

template <typename T>
std::shared_ptr<arrow::Buffer> numeric_array_arrow_ipc_view(
    const std::shared_ptr<vineyard::Object>& p) {
  auto arr = std::dynamic_pointer_cast<vineyard::NumericArray<T>>(p);
  LOG(INFO) << "numeric_array_arrow_ipc_view" << type_name<T>() << " is called";
  std::shared_ptr<arrow::io::BufferOutputStream> ssink;

  CHECK_ARROW_ERROR_AND_ASSIGN(ssink, arrow::io::BufferOutputStream::Create());
  LOG(INFO) << "buffer successfully created";

  auto kvmeta = std::shared_ptr<arrow::KeyValueMetadata>(
      new arrow::KeyValueMetadata({}, {}));

  auto meta = arr->meta();

  for (auto i : meta) {
    std::string v = i.value().dump();
    kvmeta->Append(i.key(), v);
  }

  auto schema = arrow::schema(
      {arrow::field("a", ConvertToArrowType<T>::TypeValue())}, kvmeta);
  std::shared_ptr<arrow::Table> my_table =
      arrow::Table::Make(schema, {arr->GetArray()});

  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
  CHECK_ARROW_ERROR_AND_ASSIGN(writer,
                               arrow::ipc::MakeStreamWriter(ssink, schema));

  VINEYARD_CHECK_OK(writer->WriteTable(*my_table));
  LOG(INFO) << "table is written";
  writer->Close();
  std::shared_ptr<arrow::Buffer> buffer_;
  LOG(INFO) << "writer is closed";
  CHECK_ARROW_ERROR_AND_ASSIGN(buffer_, ssink->Finish());
  LOG(INFO) << "buffer is extracted";
  return buffer_;
}

std::shared_ptr<arrow::Buffer> string_array_arrow_ipc_view(
    const std::shared_ptr<vineyard::Object>& p) {
  auto arr = std::dynamic_pointer_cast<
      vineyard::BaseBinaryArray<arrow::LargeStringArray>>(p);
  LOG(INFO) << "string_array_arrow_ipc_view is called";
  std::shared_ptr<arrow::io::BufferOutputStream> ssink;

  CHECK_ARROW_ERROR_AND_ASSIGN(ssink, arrow::io::BufferOutputStream::Create());
  LOG(INFO) << "buffer successfully created";

  auto kvmeta = std::shared_ptr<arrow::KeyValueMetadata>(
      new arrow::KeyValueMetadata({}, {}));

  auto meta = arr->meta();

  for (auto i : meta) {
    std::string v = i.value().dump();
    kvmeta->Append(i.key(), v);
  }

  auto schema = arrow::schema(
      {arrow::field("a", ConvertToArrowType<std::string>::TypeValue())},
      kvmeta);
  std::shared_ptr<arrow::Table> my_table =
      arrow::Table::Make(schema, {arr->GetArray()});

  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
  CHECK_ARROW_ERROR_AND_ASSIGN(writer,
                               arrow::ipc::MakeStreamWriter(ssink, schema));

  VINEYARD_CHECK_OK(writer->WriteTable(*my_table));
  LOG(INFO) << "table is written";
  writer->Close();
  std::shared_ptr<arrow::Buffer> buffer_;
  LOG(INFO) << "writer is closed";
  CHECK_ARROW_ERROR_AND_ASSIGN(buffer_, ssink->Finish());
  LOG(INFO) << "buffer is extracted";
  return buffer_;
}

std::shared_ptr<arrow::Buffer> bool_array_arrow_ipc_view(
    const std::shared_ptr<vineyard::Object>& p) {
  auto arr = std::dynamic_pointer_cast<vineyard::BooleanArray>(p);
  // std::clog << "new registry way" << std::endl;
  LOG(INFO) << "bool_array_arrow_ipc_view is called";
  std::shared_ptr<arrow::io::BufferOutputStream> ssink;

  CHECK_ARROW_ERROR_AND_ASSIGN(ssink, arrow::io::BufferOutputStream::Create());
  LOG(INFO) << "buffer successfully created";

  auto kvmeta = std::shared_ptr<arrow::KeyValueMetadata>(
      new arrow::KeyValueMetadata({}, {}));

  auto meta = arr->meta();

  for (auto i : meta) {
    std::string v = i.value().dump();
    kvmeta->Append(i.key(), v);
  }

  auto schema = arrow::schema(
      {arrow::field("a", ConvertToArrowType<bool>::TypeValue())}, kvmeta);
  std::shared_ptr<arrow::Table> my_table =
      arrow::Table::Make(schema, {arr->GetArray()});

  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
  CHECK_ARROW_ERROR_AND_ASSIGN(writer,
                               arrow::ipc::MakeStreamWriter(ssink, schema));

  VINEYARD_CHECK_OK(writer->WriteTable(*my_table));
  LOG(INFO) << "table is written";
  writer->Close();
  std::shared_ptr<arrow::Buffer> buffer_;
  LOG(INFO) << "writer is closed";
  CHECK_ARROW_ERROR_AND_ASSIGN(buffer_, ssink->Finish());
  LOG(INFO) << "buffer is extracted";
  return buffer_;
}

std::shared_ptr<arrow::Buffer> dataframe_arrow_ipc_view(
    const std::shared_ptr<vineyard::Object>& p) {
  // Add writer properties
  auto df = std::dynamic_pointer_cast<vineyard::DataFrame>(p);

  // ::parquet::WriterProperties::Builder builder;
  // builder.encoding(::parquet::Encoding::PLAIN);
  // builder.disable_dictionary();
  // builder.compression(::parquet::Compression::UNCOMPRESSED);
  // builder.disable_statistics();
  // builder.write_batch_size(std::numeric_limits<size_t>::max());
  // builder.max_row_group_length(std::numeric_limits<size_t>::max());
  // std::shared_ptr<::parquet::WriterProperties> props = builder.build();

  auto batch = df->AsBatch(true);
  std::shared_ptr<arrow::Table> table;
  VINEYARD_CHECK_OK(RecordBatchesToTable({batch}, &table));
  std::shared_ptr<arrow::io::BufferOutputStream> sink;
  CHECK_ARROW_ERROR_AND_ASSIGN(sink, arrow::io::BufferOutputStream::Create());
  std::clog << batch->column_data(2)->GetValues<_Float64>(1) << std::endl;
  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
  CHECK_ARROW_ERROR_AND_ASSIGN(
      writer, arrow::ipc::MakeStreamWriter(sink, batch->schema()));

  VINEYARD_CHECK_OK(writer->WriteTable(*table));
  // ::parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink,
  //                             std::numeric_limits<size_t>::max(), props);
  std::shared_ptr<arrow::Buffer> buffer;
  writer->Close();

  CHECK_ARROW_ERROR_AND_ASSIGN(buffer, sink->Finish());
  return buffer;
}

std::unordered_map<std::string, vineyard::fuse::vineyard_deserializer_nt>
arrow_ipc_register_once() {
  std::unordered_map<std::string, vineyard::fuse::vineyard_deserializer_nt>
      d_array_registry;
  // std::string array_prefix = "vineyard::NumericArray";
#define MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGSITER(T)                                                    \
  {                                                                         \
    std::string array_type = "vineyard::NumericArray";                      \
    array_type.append("<").append(type_name<T>()).append(">");              \
    LOG(INFO) << "register type: " << array_type << std::endl;              \
    d_array_registry.emplace(array_type, &numeric_array_arrow_ipc_view<T>); \
  }


  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGSITER(int8_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGSITER(int32_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGSITER(int16_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGSITER(int64_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGSITER(uint16_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGSITER(uint8_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGSITER(uint32_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGSITER(uint64_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGSITER(float);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGSITER(double);
  // d_array_registry.emplace(type_name<vineyard::NumericArray<int64_t>>(),&arrow_ipc_view<vineyard::NumericArray<int64_t>>);

  {
    std::string t_name = type_name<vineyard::BooleanArray>();
    LOG(INFO) << "register type: " << t_name << std::endl;

    d_array_registry.emplace(t_name, &bool_array_arrow_ipc_view);
  }

  {
    std::string t_name =
        type_name<vineyard::BaseBinaryArray<arrow::LargeStringArray>>();
    LOG(INFO) << "register type: " << t_name << std::endl;

    d_array_registry.emplace(t_name, &string_array_arrow_ipc_view);
  }
  {
    std::string t_name = type_name<vineyard::DataFrame>();
    LOG(INFO) << "register type: " << t_name << std::endl;

    d_array_registry.emplace(t_name, &dataframe_arrow_ipc_view);
  }
  return d_array_registry;
}
}  // namespace fuse
}  // namespace vineyard
#endif  // MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_
