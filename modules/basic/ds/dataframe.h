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

#ifndef MODULES_BASIC_DS_DATAFRAME_H_
#define MODULES_BASIC_DS_DATAFRAME_H_

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "basic/ds/arrow.h"  // IWYU pragma: keep
#include "basic/ds/dataframe.vineyard.h"
#include "basic/ds/tensor.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/i_object.h"
#include "common/util/json.h"

namespace vineyard {

/**
 * @brief DataFrameBuilder is used for constructing dataframes that supported by
 * vineyard
 *
 */
class DataFrameBuilder : public DataFrameBaseBuilder {
 public:
  explicit DataFrameBuilder(Client& client);

  /**
   * @brief Get the partition index of the global dataframe.
   *
   * @return The pair of the partition_index on rows and the partition_index on
   * columns.
   */
  const std::pair<size_t, size_t> partition_index() const;

  /**
   * @brief Set the index in the global dataframe.
   *
   * @param partition_index_row The row index.
   * @param partition_index_column The column index.
   */
  void set_partition_index(size_t partition_index_row,
                           size_t partition_index_column);

  /**
   * @brief Set the row batch index in the global dataframe.
   * Note that the row batch index gives the order of
   * batches on rows.
   *
   * @param row_batch_index The row batch index.
   */
  void set_row_batch_index(size_t row_batch_index);

  /**
   * @brief Set the index of dataframe by add a index column to dataframe.
   *
   * @param builder The index tensor builder.
   */
  void set_index(std::shared_ptr<ITensorBuilder> builder);

  /**
   * @brief Get the column of the given column name.
   *
   * @param column The given column name.
   * @return The shared pointer to the column tensor.
   */
  std::shared_ptr<ITensorBuilder> Column(json const& column) const;

  /**
   * @brief Add a column to the dataframe by registering a tensor builder to the
   * column name. When the dataframe is built, the tensor builder will be
   * employed to build the column.
   *
   * @param column The name of the column.
   * @param builder The tensor builder for the column.
   */
  void AddColumn(json const& column, std::shared_ptr<ITensorBuilder> builder);

  /**
   * @brief Drop the column with the given column name.
   *
   * @param column The name of column to be dropped.
   */
  void DropColumn(json const& column);

  /**
   * @brief Build the dataframe object.
   * @param client The client connected to the vineyard server.
   */
  Status Build(Client& client) override;

 private:
  std::vector<json> columns_;
  std::unordered_map<json, std::shared_ptr<ITensorBuilder>> values_;
};

class GlobalDataFrameBaseBuilder;

/**
 * @brief GlobalDataFrame is a DataFrame that refers a set of dataframe chunks
 * in many vineyardd nodes.
 */
class GlobalDataFrame : public BareRegistered<GlobalDataFrame>,
                        public Collection<DataFrame> {
 public:
  static std::unique_ptr<Object> Create() __attribute__((used)) {
    return std::static_pointer_cast<Object>(
        std::unique_ptr<GlobalDataFrame>{new GlobalDataFrame()});
  }

  void PostConstruct(const ObjectMeta& meta) override;

  /**
   * @brief Set the partition shape of the global dataframe.
   *
   * @param partition_shape_row The number of partitions on rows.
   * @param partition_shape_column The number of partitions on columns.
   */
  const std::pair<size_t, size_t> partition_shape() const;

  /// backwards compatibility
  const std::vector<std::shared_ptr<DataFrame>> LocalPartitions(
      Client& client) const;

 private:
  size_t partition_shape_row_;
  size_t partition_shape_column_;

  friend class GlobalDataFrameBuilder;
};

template <>
struct collection_type<DataFrame> {
  using type = GlobalDataFrame;
};

/**
 * @brief GlobalDataFrameBuilder is designed for building global dataframes
 *
 */
class GlobalDataFrameBuilder : public CollectionBuilder<DataFrame> {
 public:
  explicit GlobalDataFrameBuilder(Client& client)
      : CollectionBuilder<DataFrame>(client) {}

  /**
   * @brief Get the partition shape of the global dataframe.
   *
   * @return The pair of <number_of_partitions_on_rows,
   * number_of_partitions_on_columns>.
   */
  const std::pair<size_t, size_t> partition_shape() const;

  /**
   * @brief Set the partition shape of the global dataframe.
   *
   * @param partition_shape_row The number of partitions on rows.
   * @param partition_shape_column The number of partitions on columns.
   */
  void set_partition_shape(const size_t partition_shape_row,
                           const size_t partition_shape_column);

  /// Backwards compatibility
  void AddPartition(const ObjectID partition_id);
  void AddPartitions(const std::vector<ObjectID>& partition_ids);

 private:
  size_t partition_shape_row_;
  size_t partition_shape_column_;
};

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_DATAFRAME_H_
