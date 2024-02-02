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

#include "radix-tree.h"
#include "common/util/base64.h"
#include "common/util/logging.h"
#include "common/util/status.h"

using namespace vineyard;

RadixTree::RadixTree(int cacheCapacity) {
  this->tree = raxNew();
  // add one to the cache capacity because we insert a root node to the tree.
  this->cacheCapacity = cacheCapacity + 1;
  this->nodeCount = 0;

  // prepare root node
  std::vector<int> rootToken = {INT32_MAX};
  std::shared_ptr<NodeData> evictedNode;
  this->InsertInternal(rootToken, evictedNode);
  raxShow(this->tree);
  raxNode* dataNode = raxFindAndReturnDataNode(this->tree, rootToken.data(),
                                               rootToken.size(), NULL, false);
  DataWrapper* data = new DataWrapper();
  data->data = nullptr;
  data->dataLength = 0;
  dataNode->custom_data = data;
  dataNode->issubtree = true;
  this->rootToken = rootToken;
}

RadixTree::~RadixTree() {
  // TBD
  // raxFreeWithCallback(this->tree, [](raxNode *n) {
  //   if (n->iskey && !n->isnull) {
  //     nodeData* nodedata = (nodeData*) raxGetData(n);
  //     delete nodedata;
  //   }
  //   if (n->issubtree && n->iscustomallocated && !n->iscustomnull) {
  //     customData* customdata = (customData*) raxGetCustomData(n);
  //     delete customdata;
  //   }
  // });
}

std::shared_ptr<NodeData> RadixTree::Insert(
    std::vector<int> tokens, std::shared_ptr<NodeData>& evictedNode) {
  tokens.insert(tokens.begin(), INT32_MAX);
  return InsertInternal(tokens, evictedNode);
}

void RadixTree::Delete(std::vector<int> tokens,
                       std::shared_ptr<NodeData>& evictedNode) {
  tokens.insert(tokens.begin(), INT32_MAX);
  DeleteInternal(tokens, evictedNode);
}

std::shared_ptr<NodeData> RadixTree::Query(std::vector<int> key) {
  key.insert(key.begin(), INT32_MAX);
  return QueryInternal(key);
}

std::vector<std::shared_ptr<NodeData>> RadixTree::Split(
    std::vector<int> tokens, std::shared_ptr<NodeData>& header) {
  tokens.insert(tokens.begin(), INT32_MAX);
  return SplitInternal(tokens, header);
}

std::shared_ptr<NodeData> RadixTree::InsertInternal(
    std::vector<int> tokens, std::shared_ptr<NodeData>& evictedNode) {
  // get the sub vector of the tokens
  std::vector<int> rootToken =
      std::vector<int>(tokens.begin(), tokens.end() - 1);
  if (rootToken.size() > 0 && QueryInternal(rootToken) == nullptr) {
    return nullptr;
  }

  // insert the token vector to the radix tree
  int* insertTokensArray = tokens.data();
  size_t insertTokensArrayLen = tokens.size();
  DataWrapper* dummyData = new DataWrapper();
  DataWrapper* oldData;
  raxNode* dataNode = NULL;
  int retval = raxInsertAndReturnDataNode(
      this->tree, insertTokensArray, insertTokensArrayLen, dummyData,
      (void**) &dataNode, (void**) &oldData);
  if (dataNode == NULL) {
    throw std::runtime_error("Insert token list failed");
    return NULL;
  }
  if (retval == 1) {
    LOG(INFO) << "node count++:" << this->nodeCount;
    nodeCount++;
  }

  raxShow(this->tree);
  if (this->nodeCount > this->cacheCapacity) {
    LOG(INFO) << "cache capacity is full, evict the last recent node";
    LOG(INFO) << "cache capacity:" << this->cacheCapacity
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
  dataNode = raxFindAndReturnDataNode(
      this->tree, insertTokensArray, insertTokensArrayLen, &subTreeNode, false);
  LOG(INFO) << "sub tree node:" << subTreeNode << " data node:" << dataNode;
  /**
   * if the data node is null, it means the evicted node is the same node as
   * the inserted node.
   */
  if (dataNode == NULL) {
    LOG(INFO) << "get failed";
    return NULL;
  }

  if (subTreeNode == nullptr) {
    return std::make_shared<NodeData>(dummyData, nullptr);
  }
  return std::make_shared<NodeData>(dummyData,
                                    (DataWrapper*) subTreeNode->custom_data);
}

void RadixTree::DeleteInternal(std::vector<int> tokens,
                               std::shared_ptr<NodeData>& evictedNode) {
  // remove the token vector from the radix tree
  // TBD
  // If the evicted node is the root node of sub tree, recycle the tree.
  int* deleteTokensArray = tokens.data();
  size_t deleteTokensArrayLen = tokens.size();

  DataWrapper* oldData;
  raxNode* subTreeNode;
  std::vector<int> pre;
  // raxFindAndReturnDataNode(this->tree, deleteTokensArray,
  // deleteTokensArrayLen,
  //                          &subTreeNode, false);
  int retval = raxRemove(this->tree, deleteTokensArray, deleteTokensArrayLen,
                         (void**) &oldData, &subTreeNode);
  if (retval == 1) {
    evictedNode = std::make_shared<NodeData>(
        oldData, (DataWrapper*) subTreeNode->custom_data);
    nodeCount--;
  } else {
    LOG(INFO) << "remove failed";
  }
}

std::shared_ptr<NodeData> RadixTree::QueryInternal(std::vector<int> key) {
  LOG(INFO) << "Query";
  int* tokens = key.data();
  size_t tokensLen = key.size();

  if (this->tree == nullptr) {
    return NULL;
  }

  raxNode* subTreeNode;
  raxNode* dataNode =
      raxFindAndReturnDataNode(this->tree, tokens, tokensLen, &subTreeNode);
  LOG(INFO) << "query subtree node:" << subTreeNode;
  if (dataNode == NULL) {
    LOG(INFO) << "get failed";
    return NULL;
  }

  return std::make_shared<NodeData>((DataWrapper*) raxGetData(dataNode),
                                    (DataWrapper*) subTreeNode->custom_data);
}

std::string RadixTree::Serialize() {
  LOG(INFO) << "Serialize......";
  raxShow(this->tree);
  std::vector<std::vector<int>> tokenList;
  std::vector<void*> dataList;
  std::vector<uint64_t> timestampList;
  std::vector<std::vector<int>> subTreeTokenList;
  std::vector<void*> subTreeDataList;
  raxSerialize(this->tree, tokenList, dataList, timestampList,
               &subTreeTokenList, &subTreeDataList);

  raxShow(this->tree);
  std::string serializedStr;

  if (tokenList.size() != dataList.size()) {
    throw std::runtime_error(
        "The size of token list and data list is not equal");
  }
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

    // convert data to hex string
    char* bytes = (char*) ((DataWrapper*) dataList[index])->data;
    std::ostringstream dataOSS;

    for (int i = 0; i < ((DataWrapper*) dataList[index])->dataLength; i++) {
      dataOSS << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(bytes[i]));
    }
    serializedStr += dataOSS.str() + "\n";
  }

  serializedStr += "\t\n";

  LOG(INFO) << "sub tree token list size:" << subTreeTokenList.size();
  for (size_t index = 0; index < subTreeTokenList.size(); index++) {
    for (size_t j = 0; j < subTreeTokenList[index].size(); j++) {
      serializedStr += std::to_string(subTreeTokenList[index][j]);
      if (j < subTreeTokenList[index].size() - 1) {
        serializedStr += ",";
      }
    }
    serializedStr += "|";

    // convert custom data to hex string
    char* bytes = (char*) ((DataWrapper*) subTreeDataList[index])->data;
    std::ostringstream dataOSS;

    LOG(INFO) << "data lengtπh:"
              << ((DataWrapper*) subTreeDataList[index])->dataLength;
    for (int i = 0; i < ((DataWrapper*) subTreeDataList[index])->dataLength;
         ++i) {
      dataOSS << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(static_cast<unsigned char>(bytes[i]));
    }
    LOG(INFO) << "data:" << ((DataWrapper*) subTreeDataList[index])->data;
    LOG(INFO) << "data oss:" << dataOSS.str();
    serializedStr += dataOSS.str() + "\n";
  }
  LOG(INFO) << "serializedStr:" << serializedStr;

  // use LZ4 to compress the serialized string
  const char* const src = serializedStr.c_str();
  const int srcSize = serializedStr.size();
  const int maxDstSize = LZ4_compressBound(srcSize);
  char* compressedData = new char[maxDstSize];
  if (compressedData == NULL) {
    LOG(INFO) << "Failed to allocate memory for *compressedData.";
  }

  const int compressedDataSize =
      LZ4_compress_default(src, compressedData, srcSize, maxDstSize);
  if (compressedDataSize <= 0) {
    LOG(INFO) << "A 0 or negative result from LZ4_compress_default() "
                 "indicates a failure trying to compress the data. ";
  }

  if (compressedDataSize > 0) {
    LOG(INFO) << "We successfully compressed some data! Ratio: "
              << ((float) compressedDataSize / srcSize);
  }

  if (compressedData == NULL) {
    LOG(INFO) << "Failed to re-alloc memory for compressedData.  Sad :(";
  }

  std::string compressedStr = std::string(compressedData, compressedDataSize);
  int cacheCapacity = this->cacheCapacity - 1;
  std::string result = std::string((char*) &srcSize, sizeof(int)) +
                       std::string((char*) &cacheCapacity, sizeof(int)) +
                       compressedStr;
  delete[] compressedData;
  return result;
}

std::shared_ptr<RadixTree> RadixTree::Deserialize(std::string data) {
  LOG(INFO) << "Deserialize......";
  // use LZ4 to decompress the serialized string
  int srcSize = *(int*) data.c_str();
  data.erase(0, sizeof(int));
  int cacheCapacity = *(int*) data.c_str();
  data.erase(0, sizeof(int));
  char* const decompressBuffer = new char[srcSize];
  if (decompressBuffer == NULL) {
    LOG(INFO) << "Failed to allocate memory for *decompressBuffer.";
  }

  const int decompressedSize =
      LZ4_decompress_safe(data.c_str(), decompressBuffer, data.size(), srcSize);
  if (decompressedSize < 0) {
    LOG(INFO) << "A negative result from LZ4_decompress_safe indicates a "
                 "failure trying to decompress the data.  See exit code "
                 "(echo $?) for value returned.";
  }
  if (decompressedSize >= 0) {
    LOG(INFO) << "We successfully decompressed some data!";
  }
  // if (decompressedSize != data.size()) {
  //     LOG(INFO) << "Decompressed data is different from original! \n";
  // }
  data = std::string(decompressBuffer, decompressedSize);
  delete[] decompressBuffer;

  std::vector<std::vector<int>> tokenList;
  std::vector<void*> dataList;
  std::vector<int> dataSizeList;
  std::vector<uint64_t> timestampList;
  std::vector<std::vector<int>> subTreeTokenList;
  std::vector<void*> subTreeDataList;
  std::vector<int> subTreeDataSizeList;
  std::istringstream iss(data);
  std::string line;
  bool isMainTree = true;

  while (std::getline(iss, line)) {
    if (!line.empty() && line[0] == '\t') {
      isMainTree = false;
      line.pop_back();
      continue;
    }
    LOG(INFO) << "data line:" << line << std::endl;
    std::istringstream lineStream(line);
    std::string tokenListPart, timestampPart, dataPart;

    if (!std::getline(lineStream, tokenListPart, '|')) {
      throw std::runtime_error(
          "Invalid serialized string format in token list part.");
    }
    if (isMainTree) {
      if (!std::getline(lineStream, timestampPart, '|')) {
        throw std::runtime_error(
            "Invalid serialized string format in timestamp part.");
      }
    }
    if (!std::getline(lineStream, dataPart)) {
      LOG(INFO) << "data length is 0";
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
        LOG(INFO) << "Invalid timestamp format.";
        throw std::runtime_error("Invalid timestamp format.");
      }
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
    LOG(INFO) << "data size:" << dataSize;
    if (dataSize != 0) {
      data = new char[dataSize];
      std::istringstream dataStream(dataPart);
      for (size_t i = 0; i < dataSize; ++i) {
        // Temporary buffer to store two hexadecimal chars + null
        char hex[3] = {};
        // Read two characters for one byte
        if (!dataStream.read(hex, 2)) {
          delete[] data;
          LOG(INFO) << "Invalid data format.";
          throw std::runtime_error("Invalid data format.");
        }
        // Convert the two hex characters to one byte
        unsigned int byte;
        std::istringstream hexStream(hex);
        if (!(hexStream >> std::hex >> byte)) {
          delete[] data;
          LOG(INFO) << "Invalid data format.";
          throw std::runtime_error("Invalid data format.");
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
      std::make_shared<RadixTree>(cacheCapacity);
  radixTree->nodeCount = tokenList.size();

  raxShow(radixTree->tree);
  for (size_t i = 0; i < tokenList.size(); i++) {
    std::string token_str = "";
    for (size_t j = 0; j < tokenList[i].size(); j++) {
      token_str += std::to_string(tokenList[i][j]);
    }
    LOG(INFO) << "token:" << token_str;
    int* insertTokensArray = tokenList[i].data();
    size_t insertTokensArrayLen = tokenList[i].size();
    DataWrapper* data = new DataWrapper();
    data->data = dataList[i];
    data->dataLength = dataSizeList[i];
    raxNode* dataNode = NULL;

    // TBD
    // check retval
    raxInsertAndReturnDataNode(radixTree->tree, insertTokensArray,
                               insertTokensArrayLen, data, (void**) &dataNode,
                               NULL);

    if (dataNode == NULL) {
      throw std::runtime_error("Insert token list failed");
    }
    dataNode->timestamp = timestampList[i];
  }
  LOG(INFO) << "start to insert sub tree token list" << std::endl;
  for (size_t i = 0; i < subTreeTokenList.size(); i++) {
    for (size_t j = 0; j < subTreeTokenList[i].size(); j++) {
      LOG(INFO) << subTreeTokenList[i][j];
    }

    raxNode* node = nullptr;
    LOG(INFO) << "stage 1";
    VINEYARD_ASSERT(radixTree->tree != nullptr);

    // TBD refator this code.
    node = raxFindAndReturnDataNode(radixTree->tree, subTreeTokenList[i].data(),
                                    subTreeTokenList[i].size(), NULL, false);
    VINEYARD_ASSERT(node != nullptr);
    LOG(INFO) << "stage 2";
    DataWrapper* data = new DataWrapper();
    data->data = subTreeDataList[i];
    LOG(INFO) << subTreeDataList[i];
    data->dataLength = subTreeDataSizeList[i];

    LOG(INFO) << "stage 3";
    node->issubtree = true;
    raxSetCustomData(node, data);

    // TBD
    // refactor this code.
    radixTree->subTreeDataSet.insert(data);
  }
  LOG(INFO) << "Deserialize success";
  return radixTree;
}

std::vector<std::shared_ptr<NodeData>> RadixTree::SplitInternal(
    std::vector<int> tokens, std::shared_ptr<NodeData>& header) {
  std::vector<int> rootToken;
  DataWrapper* dummyData = new DataWrapper();
  raxNode* subTreeRootNode =
      raxSplit(this->tree, tokens.data(), tokens.size(), dummyData, rootToken);

  raxShow(this->tree);
  subTreeRootNode->issubtree = true;
  DataWrapper* treeData = new DataWrapper();
  treeData->data = nullptr;
  treeData->dataLength = 0;
  subTreeRootNode->custom_data = treeData;
  header = std::make_shared<NodeData>(
      (DataWrapper*) raxGetData(subTreeRootNode), treeData);
  return TraverseTreeWithoutSubTree(subTreeRootNode);
}

// Get child node list from this tree.
std::vector<std::shared_ptr<NodeData>> RadixTree::TraverseTreeWithoutSubTree(
    raxNode* headNode) {
  std::vector<std::shared_ptr<NodeData>> nodes;
  if (headNode == NULL) {
    LOG(INFO) << "traverse failed";
    return nodes;
  }

  std::vector<raxNode*> dataNodeList;
  std::vector<int> pre_tmp;
  raxTraverseSubTree(headNode, dataNodeList);
  LOG(INFO) << "data node list:" << dataNodeList.size();
  for (size_t i = 0; i < dataNodeList.size(); i++) {
    nodes.push_back(std::make_shared<NodeData>(
        (DataWrapper*) raxGetData(dataNodeList[i]),
        (DataWrapper*) dataNodeList[i]->custom_data));
  }
  return nodes;
}

void RadixTree::SetSubtreeData(void* data, int dataLength) {
  LOG(INFO) << "set subtree data";
  DataWrapper* dataWrapper = new DataWrapper();
  dataWrapper->data = data;
  dataWrapper->dataLength = dataLength;
  subTreeDataSet.insert(dataWrapper);
}

std::shared_ptr<NodeData> RadixTree::GetRootNode() {
  raxNode* node = raxFindAndReturnDataNode(this->tree, rootToken.data(),
                                           rootToken.size(), NULL);
  return std::make_shared<NodeData>((DataWrapper*) raxGetData(node),
                                    (DataWrapper*) node->custom_data);
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
    VINEYARD_ASSERT(evicted_tokens_vec[i][0] == INT32_MAX);
    std::vector<int> tmp(evicted_tokens_vec[i].begin() + 1,
                         evicted_tokens_vec[i].end());
    evicted_tokens.push_back(tmp);
  }

  for (auto& vec : insert_tokens_set) {
    VINEYARD_ASSERT(vec[0] == INT32_MAX);
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
    nodeDataSet.insert(raxGetData(node));
  }
  return nodeDataSet;
}