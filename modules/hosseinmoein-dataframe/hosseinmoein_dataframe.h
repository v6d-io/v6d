#pragma once
#include <iostream>
#include <vector>

#include "DataFrame/DataFrame.h"  // Main DataFrame header
#include "DataFrame/DataFrameTypes.h"

#include "client/ds/i_object.h"
#include "client/ds/blob.h"
#include "client/client.h"
// #include "basic/ds/dataframe.vineyard.h"
#include "basic/ds/dataframe.h"

using namespace vineyard;
using namespace hmdf;

template <typename T>
class HNDataFrameBuilder;

/*
 * @T: The index type.
 */
template<typename T>
class HNDataFrame: public vineyard::Registered<HNDataFrame<T>> {
 private:
  StdDataFrame<T> df;
  uint64_t vineyard_df_id;
 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
      std::unique_ptr<HNDataFrame<T>>{
        new HNDataFrame<T>()});
  }

  void Construct(const ObjectMeta& meta) override {
    this->vineyard_df_id = meta.GetKeyValue<uint64_t>("vineyard_df_id");
  }

  StdDataFrame<T> &Resolve(Client &client) {
    std::cout<< __func__ << std::endl;
    auto vineyard_df = std::dynamic_pointer_cast<vineyard::DataFrame>(client.GetObject(vineyard_df_id));
    int64_t num_rows = vineyard_df->shape().first;
    size_t num_columns = vineyard_df->shape().second;

    std::cout << num_rows << " " << num_columns << std::endl;

    //todo: abstract a function
    switch(vineyard_df->Index()->value_type()) {
    case AnyType::Int32 : {
      std::cout << "int32 index" << std::endl;
      auto index = std::dynamic_pointer_cast<Tensor<int32_t>>(vineyard_df->Index());
      std::vector<T> vec(num_rows);
      memcpy(vec.data(), index->data(), sizeof(int32_t) * num_rows);
      df.load_index(std::move(vec));
      break;
    }
    case AnyType::Int64 : {
      std::cout << "int64 index" << std::endl;
      auto index = std::dynamic_pointer_cast<Tensor<int64_t>>(vineyard_df->Index());
      std::vector<T> vec(num_rows);
      memcpy(vec.data(), index->data(), sizeof(int64_t) * num_rows);
      df.load_index(std::move(vec));
      break;
    }
    case AnyType::UInt32 : {
      std::cout << "uint32 index" << std::endl;
      auto index = std::dynamic_pointer_cast<Tensor<uint32_t>>(vineyard_df->Index());
      std::vector<T> vec(num_rows);
      memcpy(vec.data(), index->data(), sizeof(uint32_t) * num_rows);
      df.load_index(std::move(vec));
      break;
    }
    case AnyType::UInt64 : {
      std::cout << "uint64 index" << std::endl;
      auto index = std::dynamic_pointer_cast<Tensor<uint64_t>>(vineyard_df->Index());
      std::vector<T> vec(num_rows);
      memcpy(vec.data(), index->data(), sizeof(uint64_t) * num_rows);
      df.load_index(std::move(vec));
      break;
    }
    default : {
      std::cout << "TBF" << std::endl;
    }
    }

    for (int i = 0; i < num_columns; i++) {
      auto cname = vineyard_df->Columns()[i];
      std::string field_name;
      if (cname.is_string()) {
        field_name = cname.template get_ref<std::string const&>();
      } else {
        field_name = json_to_string(cname);
      }

      // abstract function
      switch(vineyard_df->Column(cname)->value_type()) {
      case AnyType::Int32 : {
        auto df_col = std::dynamic_pointer_cast<Tensor<int32_t>>(vineyard_df->Column(cname));
        std::cout << field_name << std::endl;
        std::vector<int32_t> vec(num_rows);
        for (int j = 0; j < num_rows; j++) {
          memcpy(vec.data(), df_col->data(), num_rows * sizeof(double));
          std::cout << vec[j] << std::endl;
        }
        df.load_column(field_name.c_str(), vec);
        break;
      }
      case AnyType::Int64 : {
        auto df_col = std::dynamic_pointer_cast<Tensor<int64_t>>(vineyard_df->Column(cname));
        std::cout << field_name << std::endl;
        std::vector<int64_t> vec(num_rows);
        for (int j = 0; j < num_rows; j++) {
          memcpy(vec.data(), df_col->data(), num_rows * sizeof(double));
          std::cout << vec[j] << std::endl;
        }
        df.load_column(field_name.c_str(), vec);
        break;
      }
      case AnyType::UInt32 : {
        auto df_col = std::dynamic_pointer_cast<Tensor<uint32_t>>(vineyard_df->Column(cname));
        std::cout << field_name << std::endl;
        std::vector<uint32_t> vec(num_rows);
        for (int j = 0; j < num_rows; j++) {
          memcpy(vec.data(), df_col->data(), num_rows * sizeof(double));
          std::cout << vec[j] << std::endl;
        }
        df.load_column(field_name.c_str(), vec);
        break;
      }
      case AnyType::UInt64 : {
        auto df_col = std::dynamic_pointer_cast<Tensor<uint64_t>>(vineyard_df->Column(cname));
        std::cout << field_name << std::endl;
        std::vector<uint64_t> vec(num_rows);
        for (int j = 0; j < num_rows; j++) {
          memcpy(vec.data(), df_col->data(), num_rows * sizeof(double));
          std::cout << vec[j] << std::endl;
        }
        df.load_column(field_name.c_str(), vec);
        break;
      }
      case AnyType::Double : {
        auto df_col = std::dynamic_pointer_cast<Tensor<double>>(vineyard_df->Column(cname));
        std::cout << field_name << std::endl;
        std::vector<double> vec(num_rows);
        for (int j = 0; j < num_rows; j++) {
          memcpy(vec.data(), df_col->data(), num_rows * sizeof(double));
          std::cout << vec[j] << std::endl;
        }
        df.load_column(field_name.c_str(), vec);
        break;
      }
      case AnyType::Float : {
        auto df_col = std::dynamic_pointer_cast<Tensor<float>>(vineyard_df->Column(cname));
        std::cout << field_name << std::endl;
        std::vector<float> vec(num_rows);
        for (int j = 0; j < num_rows; j++) {
          memcpy(vec.data(), df_col->data(), num_rows * sizeof(double));
          std::cout << vec[j] << std::endl;
        }
        df.load_column(field_name.c_str(), vec);
        break;
      }
      default : {
        std::cout << "TBF" << std::endl;
        break;
      }
      }
    }

    return df;
  }

  HNDataFrame() {

  }

  friend class HNDataFrameBuilder<T>;
};

template<typename T>
class HNDataFrameBuilder: public vineyard::ObjectBuilder {
 private:
  // std::unique_ptr<BlobWriter> buffer_builder;
  vineyard::DataFrameBuilder *vineyard_df_builder;
  StdDataFrame<T> df;
  uint64_t vineyard_df_id;

 public:
  HNDataFrameBuilder() {
  }

  Status Build(Client& client) override {
    std::cout << __func__ << std::endl;
    vineyard_df_builder = new DataFrameBuilder(client);
    return Status::OK();
  }

  void Put(StdDataFrame<T> &df) {
    this->df = df;
  }

  std::shared_ptr<Object> _Seal(Client& client) override {
    std::cout << __func__ << std::endl;
    VINEYARD_CHECK_OK(this->Build(client));
    auto hn_df = std::make_shared<HNDataFrame <T>>();

    {
      std::vector<T> &index_vec = df.get_index();
      // check value_type
      auto tb = std::make_shared<TensorBuilder<int64_t>>(client, std::vector<int64_t>{(int64_t)index_vec.size()});
      // what if there is not exist index column?
      vineyard_df_builder->set_index(tb);

      auto data = (std::dynamic_pointer_cast<TensorBuilder<int64_t>>(vineyard_df_builder->Column("index_")))->data();
      for (int i = 0; i < index_vec.size(); i++) {
        data[i] = index_vec[i];
        std::cout << data[i];
      }
      std::cout << std::endl;
    }

    //auto result = df->get_columns_info<>();
    auto result = df.template get_columns_info<int, double, std::string>();
    for (auto citer: result)  {
      if(std::get<2>(citer) == std::type_index(typeid(double)))
      {
        const char *col_name = std::get<0>(citer).c_str();
        std::vector<double> &vec = df.template get_column<double>(col_name);
        auto tb = std::make_shared<TensorBuilder<double>>(client, std::vector<int64_t>{(int64_t)vec.size()});
        std::cout << col_name << std::endl;
        vineyard_df_builder->AddColumn(col_name, tb);

        auto data = (std::dynamic_pointer_cast<TensorBuilder<double>>(vineyard_df_builder->Column(col_name)))->data();
        for (int i = 0; i < vec.size(); i++) {
          data[i] = vec[i];
          std::cout<< data[i];
        }
        std::cout<< std::endl;
      } else if (std::get<2>(citer) == std::type_index(typeid(int))) {
        const char *col_name = std::get<0>(citer).c_str();
        std::vector<int> &vec = df.template get_column<int>(col_name);
        auto tb = std::make_shared<TensorBuilder<int>>(client, std::vector<int64_t>{(int64_t)vec.size()});
        std::cout << col_name << std::endl;
        vineyard_df_builder->AddColumn(col_name, tb);

        auto data = (std::dynamic_pointer_cast<TensorBuilder<int>>(vineyard_df_builder->Column(col_name)))->data();
        for (int i = 0; i < vec.size(); i++) {
          data[i] = vec[i];
          std::cout<< data[i] << " ";
        }
        std::cout<< std::endl;
      } else {
        std::cout << "TBF" << std::endl;
        return nullptr;
      }
    }

    auto vineyard_df_result = vineyard_df_builder->Seal(client);
    vineyard_df_id = vineyard_df_result->id();

    hn_df->vineyard_df_id = vineyard_df_id;
    hn_df->meta_.AddKeyValue("vineyard_df_id", vineyard_df_id);
    hn_df->meta_.SetTypeName(vineyard::type_name<HNDataFrame<T>>());
    VINEYARD_CHECK_OK(client.CreateMetaData(hn_df->meta_, hn_df->id_));
    return hn_df;
  }
};
