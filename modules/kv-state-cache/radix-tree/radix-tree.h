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

#ifndef RADIX_TREE_H
#define RADIX_TREE_H

#include "radix.h"

#include "common/util/base64.h"
#include "common/util/logging.h"
#include "lz4.h"

#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <vector>

using namespace vineyard;

struct DataWrapper {
  void* data;
  int dataLength;
};

struct NodeData {
  DataWrapper* nodeData;
  DataWrapper* treeData;

  NodeData(DataWrapper* nodeData, DataWrapper* treeData) {
    this->nodeData = nodeData;
    this->treeData = treeData;
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
      std::vector<int> tokens, std::shared_ptr<NodeData>& evictedNode);

  void DeleteInternal(std::vector<int> tokens,
                      std::shared_ptr<NodeData>& evictedNode);

  std::shared_ptr<NodeData> QueryInternal(std::vector<int> key);

  std::vector<std::shared_ptr<NodeData>> SplitInternal(
      std::vector<int> tokens, std::shared_ptr<NodeData>& header);

 public:
  RadixTree(int cacheCapacity);

  ~RadixTree();

  std::shared_ptr<NodeData> Insert(std::vector<int> tokens,
                                   std::shared_ptr<NodeData>& evictedNode);

  void Delete(std::vector<int> tokens, std::shared_ptr<NodeData>& evictedNode);

  std::shared_ptr<NodeData> Query(std::vector<int> key);

  std::vector<std::shared_ptr<NodeData>> Split(
      std::vector<int> tokens, std::shared_ptr<NodeData>& header);

  std::string Serialize();

  static std::shared_ptr<RadixTree> Deserialize(std::string data);

  // Get child node list from this tree.
  static std::vector<std::shared_ptr<NodeData>> TraverseTreeWithoutSubTree(
      raxNode* headNode);

  void SetSubtreeData(void* data, int dataLength);

  rax* GetRootTree() { return this->tree; }

  int GetCacheCapacity() { return cacheCapacity; }

  std::set<void*> GetSubTreeDataSet() { return subTreeDataSet; }

  std::shared_ptr<NodeData> GetRootNode();
};

#endif
