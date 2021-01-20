/** Copyright 2020 Alibaba Group Holding Limited.

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

#include <memory>
#include <string>
#include <thread>

#include "arrow/status.h"
#include "arrow/stl.h"
#include "arrow/util/config.h"
#include "arrow/util/io_util.h"
#include "arrow/util/logging.h"

#include "basic/ds/arrow.h"
#include "basic/ds/serialization.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "glog/logging.h"

using namespace vineyard;  // NOLINT(build/namespaces)

Status reconstruct(Client& client, ObjectID in_id, ObjectID* out_id) {
  ObjectID stream_id;
  RETURN_ON_ERROR(Serialize(client, in_id, &stream_id));
  RETURN_ON_ERROR(Deserialize(client, stream_id, out_id));
  return Status::OK();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./arrow_data_structure_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  std::string prefix = "/tmp/serialization-test/array";

  {
    LOG(INFO) << "#########  Int64 Test #############";
    arrow::Int64Builder b1;
    CHECK_ARROW_ERROR(b1.AppendValues({1, 2, 3, 4}));
    CHECK_ARROW_ERROR(b1.AppendNull());
    CHECK_ARROW_ERROR(b1.AppendValues({5}));
    std::shared_ptr<arrow::Int64Array> a1;
    CHECK_ARROW_ERROR(b1.Finish(&a1));
    NumericArrayBuilder<int64_t> array_builder(client, a1);
    auto r1 = std::dynamic_pointer_cast<NumericArray<int64_t>>(
        array_builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(r1->id()));
    ObjectID restored_id;
    VINEYARD_CHECK_OK(reconstruct(client, r1->id(), &restored_id));
    auto r2 = std::dynamic_pointer_cast<NumericArray<int64_t>>(
        client.GetObject(restored_id));

    auto internal_array = r2->GetArray();
    CHECK_EQ(internal_array->length(), a1->length());
    for (int64_t i = 0; i < a1->length(); ++i) {
      CHECK_EQ(a1->Value(i), internal_array->Value(i));
    }

    // test sliced array.
    auto a3 = std::dynamic_pointer_cast<arrow::Int64Array>(a1->Slice(2, 2));
    CHECK_EQ(a3->length(), 2);
    NumericArrayBuilder<int64_t> sliced_array_builder(client, a3);
    auto r3 = std::dynamic_pointer_cast<NumericArray<int64_t>>(
        sliced_array_builder.Seal(client));
    VINEYARD_CHECK_OK(reconstruct(client, r3->id(), &restored_id));
    auto r4 = std::dynamic_pointer_cast<NumericArray<int64_t>>(
        client.GetObject(restored_id));

    auto sliced_internal_array = r4->GetArray();
    CHECK_EQ(sliced_internal_array->length(), a3->length());
    for (int64_t i = 0; i < a3->length(); ++i) {
      CHECK_EQ(a3->Value(i), sliced_internal_array->Value(i));
    }

    LOG(INFO) << "Passed int64 array wrapper tests...";
  }

  {
    LOG(INFO) << "#########  Double Test #############";
    arrow::DoubleBuilder b1;
    CHECK_ARROW_ERROR(b1.AppendValues({1.5, 2.5, 3.5, 4.5}));
    CHECK_ARROW_ERROR(b1.AppendNull());
    CHECK_ARROW_ERROR(b1.AppendValues({5.5}));
    std::shared_ptr<arrow::DoubleArray> a1;
    CHECK_ARROW_ERROR(b1.Finish(&a1));
    NumericArrayBuilder<double> array_builder(client, a1);
    auto r1 = std::dynamic_pointer_cast<NumericArray<double>>(
        array_builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(r1->id()));
    ObjectID restored_id;
    VINEYARD_CHECK_OK(reconstruct(client, r1->id(), &restored_id));
    auto r2 = std::dynamic_pointer_cast<NumericArray<double>>(
        client.GetObject(restored_id));
    auto internal_array = r2->GetArray();
    CHECK(internal_array->Equals(*a1));

    LOG(INFO) << "Passed double array wrapper tests...";
  }

  {
    LOG(INFO) << "#########  Binary Array Test ##########";
    struct S {
      int64_t a;
      double b;
    };
    arrow::FixedSizeBinaryBuilder b1(arrow::fixed_size_binary(sizeof(S)));
    S s1{1, 1.5}, s2{2, 2.5}, s3{3, 3.5}, s4{4, 4.5}, s5{5, 5.5};
    CHECK_ARROW_ERROR(
        b1.AppendValues(reinterpret_cast<uint8_t*>(&s1), sizeof(S)));
    CHECK_ARROW_ERROR(
        b1.AppendValues(reinterpret_cast<uint8_t*>(&s2), sizeof(S)));
    CHECK_ARROW_ERROR(
        b1.AppendValues(reinterpret_cast<uint8_t*>(&s3), sizeof(S)));
    CHECK_ARROW_ERROR(
        b1.AppendValues(reinterpret_cast<uint8_t*>(&s4), sizeof(S)));
    CHECK_ARROW_ERROR(b1.AppendNull());
    CHECK_ARROW_ERROR(
        b1.AppendValues(reinterpret_cast<uint8_t*>(&s5), sizeof(S)));
    std::shared_ptr<arrow::FixedSizeBinaryArray> a1;
    CHECK_ARROW_ERROR(b1.Finish(&a1));

    FixedSizeBinaryArrayBuilder array_builder(client, a1);
    auto r1 = std::dynamic_pointer_cast<FixedSizeBinaryArray>(
        array_builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(r1->id()));
    ObjectID restored_id;
    VINEYARD_CHECK_OK(reconstruct(client, r1->id(), &restored_id));
    auto r2 = std::dynamic_pointer_cast<FixedSizeBinaryArray>(
        client.GetObject(restored_id));
    auto internal_array = r2->GetArray();
    CHECK(internal_array->Equals(*a1));

    // test sliced array.
    auto a3 =
        std::dynamic_pointer_cast<arrow::FixedSizeBinaryArray>(a1->Slice(2, 2));
    CHECK_EQ(a3->length(), 2);
    FixedSizeBinaryArrayBuilder sliced_array_builder(client, a3);
    auto r3 = std::dynamic_pointer_cast<FixedSizeBinaryArray>(
        sliced_array_builder.Seal(client));
    VINEYARD_CHECK_OK(reconstruct(client, r3->id(), &restored_id));
    auto r4 = std::dynamic_pointer_cast<FixedSizeBinaryArray>(
        client.GetObject(restored_id));
    auto sliced_internal_array = r4->GetArray();
    CHECK(sliced_internal_array->Equals(a3));

    LOG(INFO) << "Passed binary array wrapper tests...";
  }

  {
    LOG(INFO) << "#########  String Array Test ########";
    arrow::StringBuilder b1;
    CHECK_ARROW_ERROR(b1.AppendValues({"a", "bb", "ccc", "dddd"}));
    CHECK_ARROW_ERROR(b1.AppendNull());
    CHECK_ARROW_ERROR(b1.AppendValues({"eeeee"}));
    std::shared_ptr<arrow::StringArray> a1;
    CHECK_ARROW_ERROR(b1.Finish(&a1));
    StringArrayBuilder array_builder(client, a1);
    auto r1 =
        std::dynamic_pointer_cast<StringArray>(array_builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(r1->id()));
    ObjectID restored_id;
    VINEYARD_CHECK_OK(reconstruct(client, r1->id(), &restored_id));
    auto r2 =
        std::dynamic_pointer_cast<StringArray>(client.GetObject(restored_id));
    auto internal_array = r2->GetArray();
    CHECK(internal_array->Equals(*a1));

    // test sliced array.
    auto a3 = std::dynamic_pointer_cast<arrow::StringArray>(a1->Slice(2, 2));
    CHECK_EQ(a3->length(), 2);
    StringArrayBuilder sliced_array_builder(client, a3);
    auto r3 = std::dynamic_pointer_cast<StringArray>(
        sliced_array_builder.Seal(client));
    VINEYARD_CHECK_OK(reconstruct(client, r3->id(), &restored_id));
    auto r4 =
        std::dynamic_pointer_cast<StringArray>(client.GetObject(restored_id));
    auto sliced_internal_array = r4->GetArray();
    CHECK(sliced_internal_array->Equals(a3));

    LOG(INFO) << "Passed string array wrapper tests...";
  }

  {
    LOG(INFO) << "######### Large String Array Test ######";
    arrow::LargeStringBuilder b1;
    CHECK_ARROW_ERROR(b1.AppendValues({"a", "bb", "ccc", "dddd"}));
    CHECK_ARROW_ERROR(b1.AppendNull());
    CHECK_ARROW_ERROR(b1.AppendValues({"eeeee"}));
    std::shared_ptr<arrow::LargeStringArray> a1;
    CHECK_ARROW_ERROR(b1.Finish(&a1));
    LargeStringArrayBuilder array_builder(client, a1);
    auto r1 =
        std::dynamic_pointer_cast<LargeStringArray>(array_builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(r1->id()));
    ObjectID restored_id;
    VINEYARD_CHECK_OK(reconstruct(client, r1->id(), &restored_id));
    auto r2 = std::dynamic_pointer_cast<LargeStringArray>(
        client.GetObject(restored_id));
    auto internal_array = r2->GetArray();
    CHECK(internal_array->Equals(*a1));

    // test sliced array.
    auto a3 =
        std::dynamic_pointer_cast<arrow::LargeStringArray>(a1->Slice(2, 2));
    CHECK_EQ(a3->length(), 2);
    LargeStringArrayBuilder sliced_array_builder(client, a3);
    auto r3 = std::dynamic_pointer_cast<LargeStringArray>(
        sliced_array_builder.Seal(client));
    VINEYARD_CHECK_OK(reconstruct(client, r3->id(), &restored_id));
    auto r4 = std::dynamic_pointer_cast<LargeStringArray>(
        client.GetObject(restored_id));
    auto sliced_internal_array = r4->GetArray();
    CHECK(sliced_internal_array->Equals(a3));

    LOG(INFO) << "Passed string array wrapper tests...";
  }

  {
    LOG(INFO) << "#########  Boolean Test #############";
    arrow::BooleanBuilder b1;
    CHECK_ARROW_ERROR(
        b1.AppendValues(std::vector<bool>{true, false, true, false}));
    CHECK_ARROW_ERROR(b1.AppendNull());
    CHECK_ARROW_ERROR(b1.AppendValues(std::vector<bool>{true}));
    std::shared_ptr<arrow::BooleanArray> a1;
    CHECK_ARROW_ERROR(b1.Finish(&a1));
    BooleanArrayBuilder array_builder(client, a1);
    auto r1 =
        std::dynamic_pointer_cast<BooleanArray>(array_builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(r1->id()));
    ObjectID restored_id;
    VINEYARD_CHECK_OK(reconstruct(client, r1->id(), &restored_id));
    auto r2 =
        std::dynamic_pointer_cast<BooleanArray>(client.GetObject(restored_id));
    auto internal_array = r2->GetArray();
    CHECK_EQ(internal_array->length(), a1->length());
    for (int64_t i = 0; i < a1->length(); ++i) {
      CHECK_EQ(a1->Value(i), internal_array->Value(i));
    }

    // test sliced array.
    auto a3 = std::dynamic_pointer_cast<arrow::BooleanArray>(a1->Slice(2, 2));
    CHECK_EQ(a3->length(), 2);
    BooleanArrayBuilder sliced_array_builder(client, a3);
    auto r3 = std::dynamic_pointer_cast<BooleanArray>(
        sliced_array_builder.Seal(client));
    VINEYARD_CHECK_OK(reconstruct(client, r3->id(), &restored_id));
    auto r4 =
        std::dynamic_pointer_cast<BooleanArray>(client.GetObject(restored_id));
    auto sliced_internal_array = r4->GetArray();
    CHECK_EQ(sliced_internal_array->length(), a3->length());
    for (int64_t i = 0; i < a3->length(); ++i) {
      CHECK_EQ(a3->Value(i), sliced_internal_array->Value(i));
    }

    LOG(INFO) << "Passed boolean array wrapper tests...";
  }

  {
    LOG(INFO) << "#########  Record Batch Test #######";
    arrow::LargeStringBuilder key_builder;
    arrow::Int64Builder value_builder;
    arrow::StringBuilder string_builder;
    std::shared_ptr<arrow::Array> array1;
    std::shared_ptr<arrow::Array> array2;
    std::shared_ptr<arrow::Array> array3;
    for (int64_t j = 0; j < 100; j++) {
      CHECK_ARROW_ERROR(key_builder.AppendValues({std::to_string(j)}));
      CHECK_ARROW_ERROR(value_builder.AppendValues({j}));
      CHECK_ARROW_ERROR(string_builder.AppendValues({std::to_string(j * j)}));
    }
    CHECK_ARROW_ERROR(key_builder.Finish(&array1));
    CHECK_ARROW_ERROR(value_builder.Finish(&array2));
    CHECK_ARROW_ERROR(string_builder.Finish(&array3));

    auto arrowSchema = arrow::schema(
        {std::make_shared<arrow::Field>("f1", arrow::large_utf8()),
         std::make_shared<arrow::Field>("f2", arrow::int64()),
         std::make_shared<arrow::Field>("f3", arrow::utf8())});
    std::shared_ptr<arrow::RecordBatch> batch = arrow::RecordBatch::Make(
        arrowSchema, array1->length(), {array1, array2, array3});
    RecordBatchBuilder builder(client, batch);
    auto r1 = std::dynamic_pointer_cast<RecordBatch>(builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(r1->id()));
    ObjectID restored_id;
    VINEYARD_CHECK_OK(reconstruct(client, r1->id(), &restored_id));
    auto r2 =
        std::dynamic_pointer_cast<RecordBatch>(client.GetObject(restored_id));
    auto internal_batch = r2->GetRecordBatch();
    CHECK(internal_batch->Equals(*batch));

    LOG(INFO) << "#########  Record Batch Extender Test #######";
    RecordBatchExtender extender(client, r2);
    VINEYARD_CHECK_OK(extender.AddColumn(client, "f7", array1));
    VINEYARD_CHECK_OK(extender.AddColumn(client, "f8", array2));
    VINEYARD_CHECK_OK(extender.AddColumn(client, "f9", array3));
    auto r3 = std::dynamic_pointer_cast<RecordBatch>(extender.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(r3->id()));
    VINEYARD_CHECK_OK(reconstruct(client, r3->id(), &restored_id));
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    CHECK_ARROW_ERROR(
        batch->AddColumn(batch->num_columns(), "f7", array1, &batch));
    CHECK_ARROW_ERROR(
        batch->AddColumn(batch->num_columns(), "f8", array2, &batch));
    CHECK_ARROW_ERROR(
        batch->AddColumn(batch->num_columns(), "f9", array3, &batch));
#else
    CHECK_ARROW_ERROR_AND_ASSIGN(
        batch, batch->AddColumn(batch->num_columns(), "f7", array1));
    CHECK_ARROW_ERROR_AND_ASSIGN(
        batch, batch->AddColumn(batch->num_columns(), "f8", array2));
    CHECK_ARROW_ERROR_AND_ASSIGN(
        batch, batch->AddColumn(batch->num_columns(), "f9", array3));
#endif
    auto r4 =
        std::dynamic_pointer_cast<RecordBatch>(client.GetObject(restored_id));
    CHECK(r4->GetRecordBatch()->Equals(*batch));
    LOG(INFO) << "Passed record batch wrapper tests...";
  }

  {
    LOG(INFO) << "#########  Table Test #############";
    arrow::LargeStringBuilder key_builder;
    arrow::Int64Builder value_builder;
    std::shared_ptr<arrow::Array> array1;
    std::shared_ptr<arrow::Array> array2;
    for (int64_t j = 0; j < 5; j++) {
      CHECK_ARROW_ERROR(key_builder.AppendValues({std::to_string(j)}));
      CHECK_ARROW_ERROR(value_builder.AppendValues({j}));
    }
    CHECK_ARROW_ERROR(key_builder.Finish(&array1));
    CHECK_ARROW_ERROR(value_builder.Finish(&array2));

    std::vector<std::string> names = {"f1", "f2"};
    std::vector<std::tuple<int64_t, double>> rows = {
        std::tuple<int64_t, double>(1, 1.5),
        std::tuple<int64_t, double>(2, 2.5),
        std::tuple<int64_t, double>(3, 3.5),
        std::tuple<int64_t, double>(4, 4.5),
        std::tuple<int64_t, double>(5, 5.5)};
    std::shared_ptr<arrow::Table> table;
    CHECK_ARROW_ERROR(arrow::stl::TableFromTupleRange(
        arrow::default_memory_pool(), rows, names, &table));
    TableBuilder builder(client, table);
    auto r1 = std::dynamic_pointer_cast<Table>(builder.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(r1->id()));
    ObjectID restored_id;
    VINEYARD_CHECK_OK(reconstruct(client, r1->id(), &restored_id));
    auto r2 = std::dynamic_pointer_cast<Table>(client.GetObject(restored_id));
    auto internal_table = r2->GetTable();
    CHECK(internal_table->Equals(*table));

    LOG(INFO) << "#########  Table Extender Test #############";
    TableExtender extender(client, r2);
    VINEYARD_CHECK_OK(extender.AddColumn(client, "f7", array1));
    VINEYARD_CHECK_OK(extender.AddColumn(client, "f8", array2));
    auto r3 = std::dynamic_pointer_cast<Table>(extender.Seal(client));
    VINEYARD_CHECK_OK(client.Persist(r3->id()));
    VINEYARD_CHECK_OK(reconstruct(client, r1->id(), &restored_id));
    auto field = ::arrow::field("f7", array1->type());
    auto chunked_array1 = std::make_shared<arrow::ChunkedArray>(array1);
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    CHECK_ARROW_ERROR(
        table->AddColumn(table->num_columns(), field, chunked_array1, &table));
#else
    CHECK_ARROW_ERROR_AND_ASSIGN(
        table, table->AddColumn(table->num_columns(), field, chunked_array1));
#endif
    field = ::arrow::field("f8", array2->type());
    auto chunked_array2 = std::make_shared<arrow::ChunkedArray>(array2);
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    CHECK_ARROW_ERROR(
        table->AddColumn(table->num_columns(), field, chunked_array2, &table));
#else
    CHECK_ARROW_ERROR_AND_ASSIGN(
        table, table->AddColumn(table->num_columns(), field, chunked_array2));
#endif
    auto r4 = std::dynamic_pointer_cast<Table>(client.GetObject(restored_id));
    CHECK(r4->GetTable()->Equals(*table));

    LOG(INFO) << "Passed Table wrapper tests...";
  }
  client.Disconnect();

  return 0;
}
