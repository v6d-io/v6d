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

#include "fuse/adaptors/arrow_ipc/deserializer_registry.h"
namespace vineyard {
namespace fuse {

std::shared_ptr<arrow::KeyValueMetadata> extractVineyardMetaToArrowMeta(
    std::shared_ptr<vineyard::Object> obj) {
  auto kvmeta = std::shared_ptr<arrow::KeyValueMetadata>(
      new arrow::KeyValueMetadata({}, {}));

  auto meta = obj->meta();

  for (auto i : meta) {
    std::string v = i.value().dump();
    kvmeta->Append(i.key(), v);
  }
  return kvmeta;
}

template <typename T>
std::shared_ptr<internal::ChunkBuffer> numeric_array_arrow_ipc_view(
    const std::shared_ptr<vineyard::Object>& p) {
  auto arr = std::dynamic_pointer_cast<vineyard::NumericArray<T>>(p);
  DLOG(INFO) << "numeric_array_arrow_ipc_view" << type_name<T>()
             << " is called";
  auto cbuffer = std::make_shared<internal::ChunkBuffer>();
  DLOG(INFO) << "buffer successfully created";

  auto kvmeta = extractVineyardMetaToArrowMeta(arr);
  auto schema = arrow::schema(
      {arrow::field("a", ConvertToArrowType<T>::TypeValue())}, kvmeta);
  std::shared_ptr<arrow::Table> my_table =
      arrow::Table::Make(schema, {arr->GetArray()});
  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
  CHECK_ARROW_ERROR_AND_ASSIGN(writer,
                               arrow::ipc::MakeFileWriter(cbuffer, schema));

  VINEYARD_CHECK_OK(writer->WriteTable(*my_table));
  DLOG(INFO) << "table is written";
  writer->Close();
  DLOG(INFO) << "buffer is extracted";
  return cbuffer;
}

std::shared_ptr<internal::ChunkBuffer> string_array_arrow_ipc_view(
    const std::shared_ptr<vineyard::Object>& p) {
  auto arr = std::dynamic_pointer_cast<
      vineyard::BaseBinaryArray<arrow::LargeStringArray>>(p);
  DLOG(INFO) << "string_array_arrow_ipc_view is called";
  auto cbuffer = std::make_shared<internal::ChunkBuffer>();

  DLOG(INFO) << "buffer successfully created";

  auto kvmeta = extractVineyardMetaToArrowMeta(arr);
  auto schema = arrow::schema(
      {arrow::field("a", ConvertToArrowType<std::string>::TypeValue())},
      kvmeta);
  std::shared_ptr<arrow::Table> my_table =
      arrow::Table::Make(schema, {arr->GetArray()});

  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
  CHECK_ARROW_ERROR_AND_ASSIGN(writer,
                               arrow::ipc::MakeFileWriter(cbuffer, schema));

  VINEYARD_CHECK_OK(writer->WriteTable(*my_table));
  DLOG(INFO) << "table is written";
  writer->Close();
  std::shared_ptr<internal::ChunkBuffer> buffer_;
  DLOG(INFO) << "writer is closed";
  return cbuffer;
}

std::shared_ptr<internal::ChunkBuffer> bool_array_arrow_ipc_view(
    const std::shared_ptr<vineyard::Object>& p) {
  auto arr = std::dynamic_pointer_cast<vineyard::BooleanArray>(p);
  DLOG(INFO) << "bool_array_arrow_ipc_view is called";
  auto cbuffer = std::make_shared<internal::ChunkBuffer>();
  auto kvmeta = extractVineyardMetaToArrowMeta(arr);
  DLOG(INFO) << "buffer successfully created";

  auto schema = arrow::schema(
      {arrow::field("a", ConvertToArrowType<bool>::TypeValue())}, kvmeta);
  std::shared_ptr<arrow::Table> my_table =
      arrow::Table::Make(schema, {arr->GetArray()});

  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
  CHECK_ARROW_ERROR_AND_ASSIGN(writer,
                               arrow::ipc::MakeFileWriter(cbuffer, schema));

  VINEYARD_CHECK_OK(writer->WriteTable(*my_table));
  DLOG(INFO) << "table is written";
  writer->Close();

  return cbuffer;
}

std::shared_ptr<internal::ChunkBuffer> dataframe_arrow_ipc_view(
    const std::shared_ptr<vineyard::Object>& p) {
  auto df = std::dynamic_pointer_cast<vineyard::DataFrame>(p);
  auto kvmeta = extractVineyardMetaToArrowMeta(df);

  auto batch = df->AsBatch(true);
  std::shared_ptr<arrow::Table> table;
  VINEYARD_CHECK_OK(RecordBatchesToTable({batch}, &table));
  auto cbuffer = std::make_shared<internal::ChunkBuffer>();
  std::clog << batch->column_data(2)->GetValues<_Float64>(1) << std::endl;
  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
  CHECK_ARROW_ERROR_AND_ASSIGN(
      writer, arrow::ipc::MakeFileWriter(cbuffer, batch->schema()));
  VINEYARD_CHECK_OK(writer->WriteTable(*table));
  std::shared_ptr<internal::ChunkBuffer> buffer;
  writer->Close();
  return cbuffer;
}

std::unordered_map<std::string, vineyard::fuse::vineyard_deserializer_nt>
arrow_ipc_register_once() {
  std::unordered_map<std::string, vineyard::fuse::vineyard_deserializer_nt>
      d_array_registry;
#define MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGISTER( \
    T)                                                                         \
  {                                                                            \
    std::string array_type = "vineyard::NumericArray";                         \
    array_type.append("<").append(type_name<T>()).append(">");                 \
    DLOG(INFO) << "register type: " << array_type << std::endl;                \
    d_array_registry.emplace(array_type, &numeric_array_arrow_ipc_view<T>);    \
  }

  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGISTER(int8_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGISTER(
      int32_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGISTER(
      int16_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGISTER(
      int64_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGISTER(
      uint16_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGISTER(
      uint8_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGISTER(
      uint32_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGISTER(
      uint64_t);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGISTER(float);
  MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_FUSE_REGISTER(double);

  {
    std::string t_name = type_name<vineyard::BooleanArray>();
    DLOG(INFO) << "register type: " << t_name << std::endl;

    d_array_registry.emplace(t_name, &bool_array_arrow_ipc_view);
  }

  {
    std::string t_name =
        type_name<vineyard::BaseBinaryArray<arrow::LargeStringArray>>();
    DLOG(INFO) << "register type: " << t_name << std::endl;

    d_array_registry.emplace(t_name, &string_array_arrow_ipc_view);
  }
  {
    std::string t_name = type_name<vineyard::DataFrame>();
    DLOG(INFO) << "register type: " << t_name << std::endl;

    d_array_registry.emplace(t_name, &dataframe_arrow_ipc_view);
  }
  return d_array_registry;
}

}  // namespace fuse
}  // namespace vineyard
