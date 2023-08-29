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

#ifndef MODULES_HOSSEINMOEIN_DATAFRAME_HOSSEINMOEIN_DATAFRAME_H_
#define MODULES_HOSSEINMOEIN_DATAFRAME_HOSSEINMOEIN_DATAFRAME_H_

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "DataFrame/DataFrame.h"  // Main DataFrame header
#include "DataFrame/DataFrameTypes.h"

#include "basic/ds/dataframe.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/logging.h"

using namespace hmdf;  // NOLINT(build/namespaces)

#define ACCEPT_TYPE double, float, int32_t, int64_t, uint32_t, uint64_t

#define LOAD_INDEX_FROM_VINEYARD_TO_SDF(T, num_rows)                         \
  do {                                                                       \
    auto index = std::dynamic_pointer_cast<Tensor<T>>(vineyard_df->Index()); \
    std::vector<T> vec(num_rows);                                            \
    memcpy(vec.data(), index->data(), sizeof(T) * num_rows);                 \
    df.load_index(std::move(vec));                                           \
  } while (0)

#define LOAD_COLUMN_FROM_VINEYARD_TO_SDF(type, filed_name)                   \
  do {                                                                       \
    auto df_col =                                                            \
        std::dynamic_pointer_cast<Tensor<type>>(vineyard_df->Column(cname)); \
    std::vector<type> vec(num_rows);                                         \
    memcpy(vec.data(), df_col->data(), num_rows * sizeof(type));             \
    df.load_column(field_name.c_str(), vec);                                 \
  } while (0)

#define LOAD_COLUMN_FROM_SDF_TO_VINEYARD(type, citer)                \
  do {                                                               \
    const char* col_name = std::get<0>(citer).c_str();               \
    std::vector<type>& vec = df.template get_column<type>(col_name); \
    auto tb = std::make_shared<TensorBuilder<type>>(                 \
        client, std::vector<int64_t>{(int64_t) vec.size()});         \
    vineyard_df_builder->AddColumn(col_name, tb);                    \
    auto data = (std::dynamic_pointer_cast<TensorBuilder<type>>(     \
                     vineyard_df_builder->Column(col_name)))         \
                    ->data();                                        \
    memcpy(data, vec.data(), vec.size() * sizeof(type));             \
  } while (0)

namespace vineyard {

template <typename T>
class HDataFrameBuilder;

template <typename T>
class HDataFrame : public vineyard::Registered<HDataFrame<T>> {
 private:
  StdDataFrame<T> df;
  uint64_t vineyard_df_id;

 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<HDataFrame<T>>{new HDataFrame<T>()});
  }

  void Construct(const ObjectMeta& meta) override {
    Object::Construct(meta);
    this->vineyard_df_id = meta.GetKeyValue<uint64_t>("vineyard_df_id");
  }

  StdDataFrame<T> Resolve(Client& client);

  friend class HDataFrameBuilder<T>;
};

template <typename T>
class HDataFrameBuilder : public vineyard::ObjectBuilder {
 private:
  vineyard::DataFrameBuilder* vineyard_df_builder;
  StdDataFrame<T> df;
  uint64_t vineyard_df_id;

 public:
  HDataFrameBuilder() {}

  ~HDataFrameBuilder() { delete vineyard_df_builder; }

  Status Build(Client& client) override {
    vineyard_df_builder = new DataFrameBuilder(client);
    return Status::OK();
  }

  void Put(StdDataFrame<T>& df) { this->df = df; }

  std::shared_ptr<Object> _Seal(Client& client) override;
};

template <typename T>
StdDataFrame<T> HDataFrame<T>::Resolve(Client& client) {
  auto vineyard_df = std::dynamic_pointer_cast<vineyard::DataFrame>(
      client.GetObject(vineyard_df_id));
  int64_t num_rows = vineyard_df->shape().first;
  size_t num_columns = vineyard_df->shape().second;

  /* Fill the index data from vineyard::DataFrame to StdDataFrame. */
  switch (vineyard_df->Index()->value_type()) {
  case AnyType::Int32:
  case AnyType::Int64:
  case AnyType::UInt32:
  case AnyType::UInt64:
  case AnyType::Double:
  case AnyType::Float: {
    LOAD_INDEX_FROM_VINEYARD_TO_SDF(T, num_rows);
    break;
  }
  default: {
    LOG(INFO) << __func__ << std::endl;
    LOG(INFO) << "The support of this type: "
              << vineyard_df->Index()->value_type() << " need to be finished."
              << std::endl;
  }
  }

  /* Fill the column data from vineyard::DataFrame to StdDataFrame. */
  for (size_t i = 0; i < num_columns; i++) {
    auto cname = vineyard_df->Columns()[i];
    std::string field_name;
    if (cname.is_string()) {
      field_name = cname.template get_ref<std::string const&>();
    } else {
      field_name = json_to_string(cname);
    }

    switch (vineyard_df->Column(cname)->value_type()) {
    case AnyType::Int32: {
      LOAD_COLUMN_FROM_VINEYARD_TO_SDF(int32_t, field_name);
      break;
    }
    case AnyType::Int64: {
      LOAD_COLUMN_FROM_VINEYARD_TO_SDF(int64_t, field_name);
      break;
    }
    case AnyType::UInt32: {
      LOAD_COLUMN_FROM_VINEYARD_TO_SDF(uint32_t, field_name);
      break;
    }
    case AnyType::UInt64: {
      LOAD_COLUMN_FROM_VINEYARD_TO_SDF(uint64_t, field_name);
      break;
    }
    case AnyType::Double: {
      LOAD_COLUMN_FROM_VINEYARD_TO_SDF(double, field_name);
      break;
    }
    case AnyType::Float: {
      LOAD_COLUMN_FROM_VINEYARD_TO_SDF(float, field_name);
      break;
    }
    default: {
      LOG(INFO) << __func__ << std::endl;
      LOG(INFO) << "The support of type: "
                << vineyard_df->Column(cname)->value_type()
                << " need to be finished." << std::endl;
      break;
    }
    }
  }

  return df;
}

template <typename T>
std::shared_ptr<Object> HDataFrameBuilder<T>::_Seal(Client& client) {
  VINEYARD_CHECK_OK(this->Build(client));
  auto hn_df = std::make_shared<HDataFrame<T>>();

  /* Get index info. */
  std::vector<T>& index_vec = df.get_index();
  auto tb = std::make_shared<TensorBuilder<T>>(
      client, std::vector<int64_t>{(int64_t) index_vec.size()});
  vineyard_df_builder->set_index(tb);

  auto data = (std::dynamic_pointer_cast<TensorBuilder<T>>(
                   vineyard_df_builder->Column("index_")))
                  ->data();
  memcpy(data, index_vec.data(), sizeof(T) * index_vec.size());

  /* Fill column data of StdDataFrame into vineyard::DataFrame. */
  auto result = df.template get_columns_info<ACCEPT_TYPE>();
  for (auto citer : result) {
    if (std::get<2>(citer) == std::type_index(typeid(double))) {
      LOAD_COLUMN_FROM_SDF_TO_VINEYARD(double, citer);
    } else if (std::get<2>(citer) == std::type_index(typeid(float))) {
      LOAD_COLUMN_FROM_SDF_TO_VINEYARD(float, citer);
    } else if (std::get<2>(citer) == std::type_index(typeid(int32_t))) {
      LOAD_COLUMN_FROM_SDF_TO_VINEYARD(int32_t, citer);
    } else if (std::get<2>(citer) == std::type_index(typeid(int64_t))) {
      LOAD_COLUMN_FROM_SDF_TO_VINEYARD(int64_t, citer);
    } else if (std::get<2>(citer) == std::type_index(typeid(uint32_t))) {
      LOAD_COLUMN_FROM_SDF_TO_VINEYARD(uint32_t, citer);
    } else if (std::get<2>(citer) == std::type_index(typeid(uint64_t))) {
      LOAD_COLUMN_FROM_SDF_TO_VINEYARD(uint64_t, citer);
    } else {
      LOG(INFO) << __func__ << std::endl;
      LOG(INFO) << "The support of this type need to be finished." << std::endl;
      assert(0);
    }
  }

  auto vineyard_df_result = vineyard_df_builder->Seal(client);
  vineyard_df_id = vineyard_df_result->id();

  hn_df->vineyard_df_id = vineyard_df_id;
  hn_df->meta_.AddKeyValue("vineyard_df_id", vineyard_df_id);
  hn_df->meta_.SetTypeName(vineyard::type_name<HDataFrame<T>>());
  VINEYARD_CHECK_OK(client.CreateMetaData(hn_df->meta_, hn_df->id_));
  return hn_df;
}

}  // namespace vineyard

#endif  // MODULES_HOSSEINMOEIN_DATAFRAME_HOSSEINMOEIN_DATAFRAME_H_
