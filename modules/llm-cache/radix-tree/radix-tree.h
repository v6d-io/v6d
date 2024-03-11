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

#ifndef MODULES_LLM_CACHE_RADIX_TREE_RADIX_TREE_H_
#define MODULES_LLM_CACHE_RADIX_TREE_RADIX_TREE_H_

#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "rax/radix.h"

#include "common/util/base64.h"
#include "common/util/logging.h"

namespace vineyard {

struct DataWrapper {
  void* data;
  int dataLength;
};

struct NodeData {
  DataWrapper* nodeData;
  DataWrapper* treeData;
  bool cleanTreeData = false;

  NodeData(DataWrapper* nodeData, DataWrapper* treeData) {
    this->nodeData = nodeData;
    this->treeData = treeData;
  }

  void RecycleSource() {
    if (this->nodeData != nullptr) {
      delete this->nodeData;
    }
    if (cleanTreeData && this->treeData != nullptr) {
      delete this->treeData;
    }
  }
};

class RadixTree : public std::enable_shared_from_this<RadixTree> {
 public:
  rax* tree;
  int cacheCapacity;
  int nodeCount;
  std::set<void*> subTreeDataSet;
  std::vector<int> rootToken;

 private:
  std::shared_ptr<NodeData> InsertInternal(
      const std::vector<int>& tokens, std::shared_ptr<NodeData>& evictedNode);

  void DeleteInternal(const std::vector<int>& tokens,
                      std::shared_ptr<NodeData>& evictedNode);

  std::shared_ptr<NodeData> QueryInternal(const std::vector<int>& tokens);

  std::vector<std::shared_ptr<NodeData>> SplitInternal(
      const std::vector<int>& tokens, std::shared_ptr<NodeData>& header);

 public:
  explicit RadixTree(int cacheCapacity, bool withRoot = true);

  ~RadixTree();

  std::shared_ptr<NodeData> Insert(const std::vector<int>& tokens,
                                   std::shared_ptr<NodeData>& evictedNode);

  void Delete(const std::vector<int>& tokens,
              std::shared_ptr<NodeData>& evictedNode);

  std::shared_ptr<NodeData> Query(const std::vector<int>& tokens);

  std::vector<std::shared_ptr<NodeData>> Split(
      const std::vector<int>& tokens, std::shared_ptr<NodeData>& header);

  std::string Serialize();

  static std::shared_ptr<RadixTree> Deserialize(std::string data);

  // Get child node list from this tree.
  static std::vector<std::shared_ptr<NodeData>> TraverseTreeWithoutSubTree(
      raxNode* headNode);

  void SetSubtreeData(void* data);

  void ClearSubtreeData(void* data);

  rax* GetRootTree() { return this->tree; }

  int GetCacheCapacity() { return cacheCapacity - 1; }

  std::set<void*> GetSubTreeDataSet() { return subTreeDataSet; }

  std::shared_ptr<NodeData> GetRootNode();

  static void MergeTree(std::shared_ptr<RadixTree> tree_1,
                        std::shared_ptr<RadixTree> tree_2,
                        std::vector<std::vector<int>>& evicted_tokens,
                        std::set<std::vector<int>>& insert_tokens);

  std::set<void*> GetAllNodeData();
};

}  // namespace vineyard

#endif  //  MODULES_LLM_CACHE_RADIX_TREE_RADIX_TREE_H_
