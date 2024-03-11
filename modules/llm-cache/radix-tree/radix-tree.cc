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

#include "llm-cache/radix-tree/radix-tree.h"

#include "common/util/status.h"

#include "zstd/lib/zstd.h"

namespace vineyard {

RadixTree::RadixTree(int cacheCapacity, bool withRoot) {
  this->tree = raxNew();
  // add one to the cache capacity because we insert a root node to the tree.
  this->cacheCapacity = cacheCapacity + 1;
  this->nodeCount = 0;

  // prepare root node
  if (withRoot) {
    std::vector<int> rootToken = {INT32_MAX};
    std::shared_ptr<NodeData> evictedNode;
    this->InsertInternal(rootToken, evictedNode);
    if (VLOG_IS_ON(100)) {
      VLOG(100) << raxShow(this->tree);
    }
    raxNode* dataNode =
        raxFindAndReturnDataNode(this->tree, rootToken, NULL, false);
    DataWrapper* data = new DataWrapper();
    data->data = nullptr;
    data->dataLength = 0;
    dataNode->custom_data = data;
    VLOG(100) << "root data wrapper:" << data;
    dataNode->issubtree = true;
    this->rootToken = rootToken;
  }
}

RadixTree::~RadixTree() {
  VLOG(100) << "~RadixTree";
  if (VLOG_IS_ON(100)) {
    VLOG(100) << raxShow(this->tree);
  }

  raxNode* dataNode =
      raxFindAndReturnDataNode(this->tree, rootToken, NULL, false);
  if (dataNode != nullptr) {
    delete reinterpret_cast<DataWrapper*>(dataNode->custom_data);
    delete reinterpret_cast<DataWrapper*>(raxGetData(dataNode));
  }

  raxFree(this->tree);
}

std::shared_ptr<NodeData> RadixTree::Insert(
    const std::vector<int>& tokens, std::shared_ptr<NodeData>& evictedNode) {
  return InsertInternal(tokens, evictedNode);
}

void RadixTree::Delete(const std::vector<int>& tokens,
                       std::shared_ptr<NodeData>& evictedNode) {
  DeleteInternal(tokens, evictedNode);
}

std::shared_ptr<NodeData> RadixTree::Query(const std::vector<int>& tokens) {
  return QueryInternal(tokens);
}

std::vector<std::shared_ptr<NodeData>> RadixTree::Split(
    const std::vector<int>& tokens, std::shared_ptr<NodeData>& header) {
  return SplitInternal(tokens, header);
}

std::shared_ptr<NodeData> RadixTree::InsertInternal(
    const std::vector<int>& tokens, std::shared_ptr<NodeData>& evictedNode) {
  // get the sub vector of the tokens
  std::vector<int> rootToken =
      std::vector<int>(tokens.begin(), tokens.end() - 1);
  if (rootToken.size() > 0 && QueryInternal(rootToken) == nullptr) {
    return nullptr;
  }

  // insert the token vector to the radix tree
  DataWrapper* dummyData = new DataWrapper();
  DataWrapper* oldData;
  raxNode* dataNode = NULL;
  int retval = raxInsertAndReturnDataNode(this->tree, tokens, dummyData,
                                          reinterpret_cast<void**>(&dataNode),
                                          reinterpret_cast<void**>(&oldData));
  if (dataNode == NULL) {
    return NULL;
  }
  if (retval == 1) {
    VLOG(100) << "node count++:" << this->nodeCount;
    nodeCount++;
  }
  if (VLOG_IS_ON(100)) {
    VLOG(100) << raxShow(this->tree);
  }
  if (this->nodeCount > this->cacheCapacity) {
    VLOG(100) << "cache capacity is full, evict the last recent node";
    VLOG(100) << "cache capacity:" << this->cacheCapacity
              << " node count:" << this->nodeCount;
    // evict the last recent node (the node with the largest lru index-
    std::vector<int> evictedTokensVector;
    raxFindLastRecentNode(this->tree->head, evictedTokensVector);
    std::string evicted_str = "";
    for (size_t i = 0; i < evictedTokensVector.size(); i++) {
      evicted_str += std::to_string(evictedTokensVector[i]);
    }
    this->DeleteInternal(evictedTokensVector, evictedNode);
  }

  raxNode* subTreeNode = nullptr;
  dataNode = raxFindAndReturnDataNode(this->tree, tokens, &subTreeNode, false);
  VLOG(100) << "sub tree node:" << subTreeNode << " data node:" << dataNode;
  /**
   * if the data node is null, it means the evicted node is the same node as
   * the inserted node.
   */
  if (dataNode == NULL) {
    return NULL;
  }

  if (subTreeNode == nullptr) {
    return std::make_shared<NodeData>(dummyData, nullptr);
  }
  return std::make_shared<NodeData>(
      dummyData, reinterpret_cast<DataWrapper*>(subTreeNode->custom_data));
}

void RadixTree::DeleteInternal(const std::vector<int>& tokens,
                               std::shared_ptr<NodeData>& evictedNode) {
  DataWrapper* oldData;
  raxNode* subTreeNode;
  std::vector<int> pre;
  raxNode* dataNode =
      raxFindAndReturnDataNode(this->tree, tokens, &subTreeNode, false);
  bool nodeIsSubTree = false;
  if (dataNode != nullptr && dataNode->issubtree) {
    nodeIsSubTree = true;
  }
  int retval =
      raxRemove(this->tree, tokens, reinterpret_cast<void**>(&oldData), false);
  if (retval == 1) {
    evictedNode = std::make_shared<NodeData>(
        oldData, reinterpret_cast<DataWrapper*>(subTreeNode->custom_data));
    nodeCount--;
    if (nodeIsSubTree) {
      evictedNode->cleanTreeData = true;
    }
  } else {
    LOG(ERROR) << "remove failed";
  }
}

std::shared_ptr<NodeData> RadixTree::QueryInternal(
    const std::vector<int>& tokens) {
  VLOG(100) << "Query";

  if (this->tree == nullptr) {
    return NULL;
  }

  raxNode* subTreeNode;
  raxNode* dataNode =
      raxFindAndReturnDataNode(this->tree, tokens, &subTreeNode);
  if (dataNode == NULL) {
    return NULL;
  }

  return std::make_shared<NodeData>(
      reinterpret_cast<DataWrapper*>(raxGetData(dataNode)),
      reinterpret_cast<DataWrapper*>(subTreeNode->custom_data));
}

std::string RadixTree::Serialize() {
  VLOG(100) << "Serialize......";
  if (VLOG_IS_ON(100)) {
    VLOG(100) << raxShow(this->tree);
  }
  std::vector<std::vector<int>> tokenList;
  std::vector<void*> dataList;
  std::vector<uint64_t> timestampList;
  std::vector<std::vector<int>> subTreeTokenList;
  std::vector<void*> subTreeDataList;
  raxSerialize(this->tree, tokenList, dataList, timestampList,
               &subTreeTokenList, &subTreeDataList);

  if (VLOG_IS_ON(100)) {
    VLOG(100) << raxShow(this->tree);
  }
  std::string serializedStr;

  for (size_t index = 0; index < tokenList.size(); index++) {
    for (size_t j = 0; j < tokenList[index].size(); j++) {
      serializedStr += std::to_string(tokenList[index][j]);
      if (j < tokenList[index].size() - 1) {
        serializedStr += ",";
      }
    }
    serializedStr += "|";

    // convert timestamp(uint64) to hex string
    uint64_t timestamp = timestampList[index];
    std::ostringstream timestampOSS;
    timestampOSS << std::hex << timestamp;

    serializedStr += timestampOSS.str() + "|";

    raxNode* node =
        raxFindAndReturnDataNode(this->tree, tokenList[index], NULL, false);
    uint32_t numNodes = node->numnodes;
    std::ostringstream subTreeSizeOSS;
    subTreeSizeOSS << std::hex << numNodes;

    serializedStr += subTreeSizeOSS.str() + "|";

    // convert data to hex string
    char* bytes = reinterpret_cast<char*>(
        (reinterpret_cast<DataWrapper*>(dataList[index]))->data);
    std::ostringstream dataOSS;

    for (int i = 0;
         i < (reinterpret_cast<DataWrapper*>(dataList[index]))->dataLength;
         i++) {
      dataOSS << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(bytes[i]));
    }
    serializedStr += dataOSS.str() + "\n";
  }

  serializedStr += "\t\n";

  VLOG(100) << "sub tree token list size:" << subTreeTokenList.size();
  for (size_t index = 0; index < subTreeTokenList.size(); index++) {
    for (size_t j = 0; j < subTreeTokenList[index].size(); j++) {
      serializedStr += std::to_string(subTreeTokenList[index][j]);
      if (j < subTreeTokenList[index].size() - 1) {
        serializedStr += ",";
      }
    }
    serializedStr += "|";

    // convert custom data to hex string
    char* bytes = reinterpret_cast<char*>(
        (reinterpret_cast<DataWrapper*>(subTreeDataList[index]))->data);
    std::ostringstream dataOSS;

    VLOG(100)
        << "data length:"
        << (reinterpret_cast<DataWrapper*>(subTreeDataList[index]))->dataLength;
    for (int i = 0;
         i <
         (reinterpret_cast<DataWrapper*>(subTreeDataList[index]))->dataLength;
         ++i) {
      dataOSS << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(bytes[i]));
    }
    VLOG(100) << "data:"
              << (reinterpret_cast<DataWrapper*>(subTreeDataList[index]))->data;
    VLOG(100) << "data oss:" << dataOSS.str();
    serializedStr += dataOSS.str() + "\n";
  }
  VLOG(100) << "serializedStr:" << serializedStr;

  // use ZSTD to compress the serialized string
  size_t srcSize = serializedStr.size();
  size_t dstSize = ZSTD_compressBound(srcSize);
  std::string compressedStr(dstSize + 1, '\0');
  VLOG(100) << "src size:" << srcSize << " dst size:" << dstSize;
  int compressedSize = ZSTD_compress(compressedStr.data(), compressedStr.size(),
                                     serializedStr.c_str(), srcSize, 3);
  if (ZSTD_isError(compressedSize)) {
    LOG(ERROR) << "ZSTD compression failed: "
               << ZSTD_getErrorName(compressedSize);
  }
  int cacheCapacity = this->cacheCapacity - 1;

  std::string result =
      std::string(reinterpret_cast<char*>(&compressedSize), sizeof(int)) +
      std::string(reinterpret_cast<char*>(&cacheCapacity), sizeof(int)) +
      std::string(reinterpret_cast<char*>(&(this->tree->head->numnodes)),
                  sizeof(uint32_t)) +
      compressedStr;

  return result;
}

std::shared_ptr<RadixTree> RadixTree::Deserialize(std::string data) {
  VLOG(100) << "Deserialize......";
  // use LZ4 to decompress the serialized string
  int compressedSize = *reinterpret_cast<int*>(data.data());
  data.erase(0, sizeof(int));
  int cacheCapacity = *reinterpret_cast<int*>(data.data());
  data.erase(0, sizeof(int));
  int rootNumNodes = *reinterpret_cast<uint32_t*>(data.data());
  data.erase(0, sizeof(uint32_t));
  uint64_t ds = ZSTD_getFrameContentSize(data.c_str(), data.size());
  if (ds == ZSTD_CONTENTSIZE_ERROR) {
    LOG(ERROR) << "Error: not a valid compressed frame";
  } else if (ds == ZSTD_CONTENTSIZE_UNKNOWN) {
    LOG(ERROR)
        << "Error: original size unknown. Use streaming decompression instead.";
  }

  std::string decompressedStr(ds + 1, '\0');
  int decompressedSize =
      ZSTD_decompress(decompressedStr.data(), ds, data.c_str(), compressedSize);
  if (ZSTD_isError(decompressedSize)) {
    LOG(ERROR) << "ZSTD decompression failed: "
               << ZSTD_getErrorName(decompressedSize);
  }
  data = decompressedStr.substr(0, decompressedSize);

  std::vector<std::vector<int>> tokenList;
  std::vector<void*> dataList;
  std::vector<int> dataSizeList;
  std::vector<uint64_t> timestampList;
  std::vector<std::vector<int>> subTreeTokenList;
  std::vector<void*> subTreeDataList;
  std::vector<int> subTreeDataSizeList;
  std::vector<int> subTreeSizeList;
  std::istringstream iss(data);
  std::string line;
  bool isMainTree = true;

  while (std::getline(iss, line)) {
    if (!line.empty() && line[0] == '\t') {
      isMainTree = false;
      line.pop_back();
      continue;
    }
    VLOG(100) << "data line:" << line << std::endl;
    std::istringstream lineStream(line);
    std::string tokenListPart, timestampPart, dataPart, subTreeSizePart;

    if (!std::getline(lineStream, tokenListPart, '|')) {
      LOG(ERROR) << "Invalid serialized string format in token list part.";
    }
    if (isMainTree) {
      if (!std::getline(lineStream, timestampPart, '|')) {
        LOG(ERROR) << "Invalid serialized string format in timestamp part.";
      }
      if (!std::getline(lineStream, subTreeSizePart, '|')) {
        LOG(ERROR) << "Invalid serialized string format in sub tree size part.";
      }
    }
    if (!std::getline(lineStream, dataPart)) {
      VLOG(100) << "data length is 0";
    }

    std::istringstream keyStream(tokenListPart);
    std::string token;
    std::vector<int> keys;
    while (std::getline(keyStream, token, ',')) {
      keys.push_back(std::stoi(token));
    }

    uint64_t timestamp;
    if (isMainTree) {
      std::istringstream timestampStream(timestampPart);
      if (!(timestampStream >> std::hex >> timestamp)) {
        LOG(ERROR) << "Invalid timestamp format.";
      }

      std::istringstream subTreeSizeStream(subTreeSizePart);
      uint32_t subTreeSize;
      if (!(subTreeSizeStream >> std::hex >> subTreeSize)) {
        LOG(ERROR) << "Invalid sub tree size format.";
      }
      VLOG(100) << "Deserialize sub tree size:" << subTreeSize;
      subTreeSizeList.push_back(subTreeSize);
    }

    size_t dataSize = dataPart.length() /
                      2;  // Each byte is represented by two hex characters
    if (isMainTree) {
      dataSizeList.push_back(dataSize);
    } else {
      subTreeDataSizeList.push_back(dataSize);
    }
    // This pointer will be freed by upper layer. Because this data
    // is created by upper layer. Here just recover it from serialized
    // string.
    char* data = nullptr;
    VLOG(100) << "data size:" << dataSize;
    if (dataSize != 0) {
      data = new char[dataSize];
      std::istringstream dataStream(dataPart);
      for (size_t i = 0; i < dataSize; ++i) {
        // Temporary buffer to store two hexadecimal chars + null
        char hex[3] = {};
        // Read two characters for one byte
        if (!dataStream.read(hex, 2)) {
          delete[] data;
          LOG(ERROR) << "Invalid data format.";
        }
        // Convert the two hex characters to one byte
        unsigned int byte;
        std::istringstream hexStream(hex);
        if (!(hexStream >> std::hex >> byte)) {
          delete[] data;
          LOG(ERROR) << "Invalid data format.";
        }
        reinterpret_cast<unsigned char*>(data)[i] =
            static_cast<unsigned char>(byte);
      }
    }
    if (isMainTree) {
      tokenList.push_back(keys);
      timestampList.push_back(timestamp);
      dataList.push_back(data);
    } else {
      subTreeTokenList.push_back(keys);
      subTreeDataList.push_back(data);
    }
  }

  // This pointer will be freed by upper layer. Because this data
  // is created by upper layer. Here just recover it from serialized
  // string.
  std::shared_ptr<RadixTree> radixTree =
      std::make_shared<RadixTree>(cacheCapacity, false);
  radixTree->nodeCount = tokenList.size();

  if (VLOG_IS_ON(100)) {
    VLOG(100) << raxShow(radixTree->tree);
  }
  for (size_t i = 0; i < tokenList.size(); i++) {
    std::string token_str = "";
    for (size_t j = 0; j < tokenList[i].size(); j++) {
      token_str += std::to_string(tokenList[i][j]);
    }
    VLOG(100) << "token:" << token_str;
    DataWrapper* data = new DataWrapper();
    data->data = dataList[i];
    data->dataLength = dataSizeList[i];
    raxNode* dataNode = NULL;

    int retval =
        raxInsertAndReturnDataNode(radixTree->tree, tokenList[i], data,
                                   reinterpret_cast<void**>(&dataNode), NULL);
    if (retval == 0) {
      LOG(WARNING) << "Overwrite the data of the node";
    }
    dataNode->timestamp = timestampList[i];
    dataNode->numnodes = subTreeSizeList[i];
  }

  radixTree->tree->head->numnodes = rootNumNodes;
  if (VLOG_IS_ON(100)) {
    VLOG(100) << raxShow(radixTree->tree);
  }

  VLOG(100) << "start to insert sub tree token list" << std::endl;
  for (size_t i = 0; i < subTreeTokenList.size(); i++) {
    for (size_t j = 0; j < subTreeTokenList[i].size(); j++) {
      VLOG(100) << subTreeTokenList[i][j];
    }

    raxNode* node = nullptr;
    VLOG(100) << "stage 1";

    // TBD refactor this code.
    node = raxFindAndReturnDataNode(radixTree->tree, subTreeTokenList[i], NULL,
                                    false);
    VLOG(100) << "stage 2";
    DataWrapper* data = new DataWrapper();
    data->data = subTreeDataList[i];
    VLOG(100) << subTreeDataList[i];
    data->dataLength = subTreeDataSizeList[i];

    VLOG(100) << "stage 3";
    node->issubtree = true;
    raxSetCustomData(node, data);

    radixTree->SetSubtreeData(subTreeDataList[i]);
  }
  VLOG(100) << "Deserialize success";
  if (VLOG_IS_ON(100)) {
    VLOG(100) << raxShow(radixTree->tree);
  }
  return radixTree;
}

std::vector<std::shared_ptr<NodeData>> RadixTree::SplitInternal(
    const std::vector<int>& tokens, std::shared_ptr<NodeData>& header) {
  std::vector<int> rootToken;
  raxNode* subTreeRootNode = raxSplit(this->tree, tokens, rootToken);

  subTreeRootNode->issubtree = true;
  DataWrapper* treeData = new DataWrapper();
  treeData->data = nullptr;
  treeData->dataLength = 0;
  subTreeRootNode->custom_data = treeData;
  header = std::make_shared<NodeData>(
      reinterpret_cast<DataWrapper*>(raxGetData(subTreeRootNode)), treeData);
  return TraverseTreeWithoutSubTree(subTreeRootNode);
}

// Get child node list from this tree.
std::vector<std::shared_ptr<NodeData>> RadixTree::TraverseTreeWithoutSubTree(
    raxNode* headNode) {
  std::vector<std::shared_ptr<NodeData>> nodes;
  if (headNode == NULL) {
    VLOG(100) << "traverse failed";
    return nodes;
  }

  std::vector<raxNode*> dataNodeList;
  std::vector<int> pre_tmp;
  raxTraverseSubTree(headNode, dataNodeList);
  VLOG(100) << "data node list:" << dataNodeList.size();
  for (size_t i = 0; i < dataNodeList.size(); i++) {
    nodes.push_back(std::make_shared<NodeData>(
        reinterpret_cast<DataWrapper*>(raxGetData(dataNodeList[i])),
        reinterpret_cast<DataWrapper*>(dataNodeList[i]->custom_data)));
  }
  return nodes;
}

void RadixTree::SetSubtreeData(void* data) {
  VLOG(100) << "set subtree data:" << data;
  subTreeDataSet.insert(data);
}

void RadixTree::ClearSubtreeData(void* data) {
  VLOG(100) << "clear subtree data:" << data;
  subTreeDataSet.erase(data);
}

std::shared_ptr<NodeData> RadixTree::GetRootNode() {
  raxNode* node = raxFindAndReturnDataNode(this->tree, rootToken, NULL);
  return std::make_shared<NodeData>(
      reinterpret_cast<DataWrapper*>(raxGetData(node)),
      reinterpret_cast<DataWrapper*>(node->custom_data));
}

void RadixTree::MergeTree(std::shared_ptr<RadixTree> tree_1,
                          std::shared_ptr<RadixTree> tree_2,
                          std::vector<std::vector<int>>& evicted_tokens,
                          std::set<std::vector<int>>& insert_tokens) {
  std::set<std::vector<int>> insert_tokens_set;
  std::vector<std::vector<int>> evicted_tokens_vec;
  mergeTree(tree_1->tree, tree_2->tree, evicted_tokens_vec, insert_tokens_set,
            tree_1->cacheCapacity);
  for (size_t i = 0; i < evicted_tokens_vec.size(); i++) {
    std::vector<int> tmp(evicted_tokens_vec[i].begin() + 1,
                         evicted_tokens_vec[i].end());
    evicted_tokens.push_back(tmp);
  }

  for (auto& vec : insert_tokens_set) {
    std::vector<int> tmp(vec.begin() + 1, vec.end());
    insert_tokens.insert(tmp);
  }
}

std::set<void*> RadixTree::GetAllNodeData() {
  raxIterator iter;
  raxStart(&iter, this->tree);
  raxSeek(&iter, "^", NULL, 0);
  std::set<void*> nodeDataSet;
  while (raxNext(&iter)) {
    raxNode* node = iter.node;
    if (node->isnull) {
      continue;
    }
    nodeDataSet.insert(
        (reinterpret_cast<DataWrapper*>(raxGetData(node)))->data);
  }
  return nodeDataSet;
}

}  // namespace vineyard
