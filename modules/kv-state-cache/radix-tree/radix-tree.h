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
#include "kv-state-cache/strategy/LRU_strategy.h"
#include "lz4.h"

#include <iomanip>
#include <map>
#include <memory>
#include <vector>
#include <set>

using namespace vineyard;

typedef struct customData {
  int data_length;
  void* data;
} customData;

typedef struct nodeData {
  int data_length;
  void* data;
} nodeData;

class Node {
 private:
  nodeData* data;
  raxNode* node;

 public:
  Node(raxNode* node) {
    this->data = (nodeData*) raxGetData(node);
    this->node = node;
  }

  Node(nodeData* data) {
    this->data = data;
    this->node = NULL;
  }

  void set_data(void* data, int data_length) {
    if (this->node == NULL) {
      LOG(INFO) << "set data failed, node is null";
      return;
    }
    this->data->data = data;
    this->data->data_length = data_length;
    raxSetData(this->node, this->data);
  }

  void* get_data() { return this->data->data; }

  int get_data_length() { return this->data->data_length; }
};

class RadixTree;

class NodeWithTreeAttri {
 private:
  std::shared_ptr<Node> node;
  std::shared_ptr<RadixTree> belong_to;

 public:
  NodeWithTreeAttri(std::shared_ptr<Node> node,
                    std::shared_ptr<RadixTree> belong_to) {
    this->node = node;
    this->belong_to = belong_to;
  }

  std::shared_ptr<Node> get_node() { return node; }

  std::shared_ptr<RadixTree> get_tree() { return belong_to; }
};

class RadixTree : public std::enable_shared_from_this<RadixTree> {
 public:
  // the whole radix tree for prefix match
  rax* tree;
  int cache_capacity;
  int node_count;
  std::set<std::shared_ptr<RadixTree>> sub_tree;
  std::vector<int> prefix;

 public:
  RadixTree(int cache_capacity) {
    this->tree = raxNew();
    this->cache_capacity = cache_capacity;

    std::vector<int> root_token = {INT32_MAX};
    std::shared_ptr<NodeWithTreeAttri> evicted_node;
    this->InsertInternal(root_token, evicted_node);
    this->prefix = root_token;
  }

  RadixTree(rax* rax_tree, int cache_capacity, std::vector<int> prefix) {
    this->tree = rax_tree;
    this->cache_capacity = cache_capacity;
    this->prefix = prefix;
  }

  RadixTree(void* custom_data, int custom_data_length, int cache_capacity) {
    this->tree = raxNew();
    this->tree->head->issubtree = true;
    customData* custom_data_struct = new customData();
    custom_data_struct->data = custom_data;
    custom_data_struct->data_length = custom_data_length;
    raxSetCustomData(this->tree->head, custom_data_struct);
    this->cache_capacity = cache_capacity;
  }

  ~RadixTree() {
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

  std::shared_ptr<NodeWithTreeAttri> Insert(
      std::vector<int> tokens,
      std::shared_ptr<NodeWithTreeAttri>& evicted_node) {
    tokens.insert(tokens.begin(), INT32_MAX);
    return InsertInternal(tokens, evicted_node);
  }

  void Delete(std::vector<int> tokens,
              std::shared_ptr<NodeWithTreeAttri>& evicted_node) {
    tokens.insert(tokens.begin(), INT32_MAX);
    DeleteInternal(tokens, evicted_node);
  }

  std::shared_ptr<NodeWithTreeAttri> Query(std::vector<int> key) {
    key.insert(key.begin(), INT32_MAX);
    return QueryInternal(key);
  }

  std::shared_ptr<RadixTree> Split(std::vector<int> tokens) {
    tokens.insert(tokens.begin(), INT32_MAX);
    return SplitInternal(tokens);
  }

  std::shared_ptr<NodeWithTreeAttri> InsertInternal(
      std::vector<int> tokens,
      std::shared_ptr<NodeWithTreeAttri>& evicted_node) {
    // get the sub vector of the tokens
    // TBD
    // change the api to Insert(prefix, tokens, evicted_node);
    std::vector<int> prefix =
        std::vector<int>(tokens.begin(), tokens.end() - 1);
    if (prefix.size() > 0 && QueryInternal(prefix) == nullptr) {
      return nullptr;
    }

    // insert the token vector to the radix tree
    int* insert_tokens_array = tokens.data();
    size_t insert_tokens_array_len = tokens.size();
    nodeData* dummy_data = new nodeData();
    nodeData* old_data;
    raxNode* dataNode = NULL;
    int retval = raxInsertAndReturnDataNode(
        this->tree, insert_tokens_array, insert_tokens_array_len, dummy_data,
        (void**) &dataNode, (void**) &old_data);
    if (dataNode == NULL) {
      throw std::runtime_error("Insert token list failed");
      return NULL;
    }
    if (retval == 1) {
      node_count++;
    }

    raxShow(this->tree);
    if (this->node_count > this->cache_capacity) {
      LOG(INFO) << "cache capacity is full, evict the last recent node";
      LOG(INFO) << "cache capacity:" << this->cache_capacity << " node count:" << this->node_count;
      // evict the last recent node (the node with the largest lru index-
      std::vector<int> evicted_tokens;
      raxFindLastRecentNode(this->tree->head, evicted_tokens);
      std::string evicted_str = "";
      for (size_t i = 0; i < evicted_tokens.size(); i++) {
        evicted_str += std::to_string(evicted_tokens[i]);
      }
      this->DeleteInternal(evicted_tokens, evicted_node);
    }

    raxNode *subTreeNode;
    std::vector<int> sub_tree_prefix;
    dataNode = raxFindAndReturnDataNode(this->tree, insert_tokens_array,
                                        insert_tokens_array_len, sub_tree_prefix, &subTreeNode, false);
    std::string sub_tree_prefix_str = "";
    for (int i = 0; i < sub_tree_prefix.size(); i++) {
      sub_tree_prefix_str += std::to_string(sub_tree_prefix[i]);
    }
    LOG(INFO) << "sub tree prefix:" << sub_tree_prefix_str;
    /**
     * if the data node is null, it means the evicted node is the same node as
     * the inserted node.
     */
    if (dataNode == NULL) {
      LOG(INFO) << "get failed";
      return NULL;
    }

    // TBD
    // cache subtree on root radixTree
    // recycle the memory of the rax(e.g. add delete function for make_shared)
    // rax* sub_tree = raxNew();
    // sub_tree->head = subTreeNode;

    // No need to fill prefix length because the prefix length is not used in
    // this case.
    // TBD
    // refactor this code.
    std::shared_ptr<RadixTree> tree = std::make_shared<RadixTree>(this->tree, this->cache_capacity, sub_tree_prefix);
    return std::make_shared<NodeWithTreeAttri>(std::make_shared<Node>(dataNode),
                                               tree);
  }

  void DeleteInternal(std::vector<int> tokens,
              std::shared_ptr<NodeWithTreeAttri>& evicted_node) {
    // remove the token vector from the radix tree
    // TBD
    // If the evicted node is the root node of sub tree, recycle the tree.
    int* delete_tokens_array = tokens.data();
    size_t delete_tokens_array_len = tokens.size();

    nodeData* old_data;
    raxNode* subTreeNode;
    std::vector<int> pre;
    raxFindAndReturnDataNode(this->tree, delete_tokens_array, delete_tokens_array_len, pre, &subTreeNode, false);
    int retval = raxRemove(this->tree, delete_tokens_array,
                           delete_tokens_array_len, (void**) &old_data, &subTreeNode);
    if (retval == 1) {
      std::shared_ptr<Node> node = std::make_shared<Node>(old_data);
      // rax* sub_tree = raxNew();
      // sub_tree->head = subTreeNode;
      // No need to fill prefix length because the prefix length is not used in
      // this case.
      // TBD
      // refactor this code.
      std::shared_ptr<RadixTree> tree = std::make_shared<RadixTree>(this->tree, this->cache_capacity, pre);
      evicted_node =
          std::make_shared<NodeWithTreeAttri>(node, tree);
      node_count--;
    } else {
      LOG(INFO) << "remove failed";
    }
  
  }

  std::shared_ptr<NodeWithTreeAttri> QueryInternal(std::vector<int> key) {
    LOG(INFO) << "Query";
    int* tokens = key.data();
    size_t tokens_len = key.size();

    if (this->tree == nullptr) {
      return NULL;
    }

    raxNode* subTreeNode;
    std::vector<int> prefix;
    raxNode* dataNode =
        raxFindAndReturnDataNode(this->tree, tokens, tokens_len, prefix, &subTreeNode);
    LOG(INFO) << "query subtree node:" << subTreeNode;
    if (dataNode == NULL) {
      LOG(INFO) << "get failed";
      return NULL;
    }

    std::shared_ptr<Node> node = std::make_shared<Node>(dataNode);
    // TBD
    // exists memory leak here.
    // rax* sub_tree = raxNew();
    // sub_tree->head = subTreeNode;
    // No need to fill prefix length because the prefix length is not used in
    // this case.
    // TBD
    // refactor this code.
    std::shared_ptr<RadixTree> tree = std::make_shared<RadixTree>(this->tree, this->cache_capacity, prefix);
    return std::make_shared<NodeWithTreeAttri>(node, tree);
  }

  std::string Serialize() {
    LOG(INFO) << "Serialize......";
    raxShow(this->tree);
    std::vector<std::vector<int>> token_list;
    std::vector<void*> data_list;
    std::vector<uint64_t> timestamp_list;
    std::vector<std::vector<int>> sub_tree_token_list;
    std::vector<void*> sub_tree_data_list;
    raxSerialize(this->tree, token_list, data_list, timestamp_list, &sub_tree_token_list,
                 &sub_tree_data_list);

    raxShow(this->tree);
    std::string serialized_str;

    if (token_list.size() != data_list.size()) {
      throw std::runtime_error("The size of token list and data list is not equal");
    }
    for (size_t index = 0; index < token_list.size(); index++) {
      for (size_t j = 0; j < token_list[index].size(); j++) {
        serialized_str += std::to_string(token_list[index][j]);
        if (j < token_list[index].size() - 1) {
          serialized_str += ",";
        }
      }
      serialized_str += "|";

      // convert timestamp(uint64) to hex string
      uint64_t timestamp = timestamp_list[index];
      std::ostringstream timestamp_oss;
      timestamp_oss << std::hex << timestamp;

      serialized_str += timestamp_oss.str() + "|";

      // convert data to hex string
      char* bytes = (char*) ((nodeData*) data_list[index])->data;
      std::ostringstream data_oss;

      for (int i = 0; i < ((nodeData*)data_list[index])->data_length; i++) {
          data_oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(static_cast<unsigned char>(bytes[i]));
      }
      serialized_str += data_oss.str() + "\n";
    }

    serialized_str += "\t\n";

    LOG(INFO) << "sub tree token list size:" << sub_tree_token_list.size();
    for (size_t index = 0; index < sub_tree_token_list.size(); index++) {
      for (size_t j = 0; j < sub_tree_token_list[index].size(); j++) {
        serialized_str += std::to_string(sub_tree_token_list[index][j]);
        if (j < sub_tree_token_list[index].size() - 1) {
          serialized_str += ",";
        }
      }
      serialized_str += "|";

      // for test
      // TreeData1 *data = (TreeData1*)((customData*) sub_tree_data_list[index])->data;
      // LOG(INFO) << "Does the data is object id?" << ((data->is_ptr == true) ? "true" : "false");
      // LOG(INFO) << "object id is:" << data->builder_object_id;

      // convert custom data to hex string
      char* bytes = (char*) ((customData*) sub_tree_data_list[index])->data;
      std::ostringstream data_oss;

      LOG(INFO) << "data lengtπh:" << ((customData*)sub_tree_data_list[index])->data_length;
      for (int i = 0; i < ((customData*)sub_tree_data_list[index])->data_length; ++i) {
          data_oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(static_cast<unsigned char>(bytes[i]));
      }
      LOG(INFO) << "data:" << ((customData*)sub_tree_data_list[index])->data;
      LOG(INFO) << "data oss:" << data_oss.str();
      serialized_str += data_oss.str() + "\n";
    }
    LOG(INFO) << "serialized_str:" << serialized_str;

    // use LZ4 to compress the serialized string
    const char* const src = serialized_str.c_str();
    const int src_size = serialized_str.size();
    const int max_dst_size = LZ4_compressBound(src_size);
    char* compressed_data = new char[max_dst_size];
    if (compressed_data == NULL) {
      LOG(INFO) << "Failed to allocate memory for *compressed_data.";
    }

    const int compressed_data_size =
        LZ4_compress_default(src, compressed_data, src_size, max_dst_size);
    if (compressed_data_size <= 0) {
      LOG(INFO) << "A 0 or negative result from LZ4_compress_default() "
                   "indicates a failure trying to compress the data. ";
    }

    if (compressed_data_size > 0) {
      LOG(INFO) << "We successfully compressed some data! Ratio: "
                << ((float) compressed_data_size / src_size);
    }

    if (compressed_data == NULL) {
      LOG(INFO) << "Failed to re-alloc memory for compressed_data.  Sad :(";
    }

    std::string compressed_str =
        std::string(compressed_data, compressed_data_size);
    std::string result =
        std::string((char*) &src_size, sizeof(int)) + compressed_str;
    delete[] compressed_data;
    return result;
  }


  static std::shared_ptr<RadixTree> Deserialize(std::string data) {
    LOG(INFO) << "Deserialize......";
    // use LZ4 to decompress the serialized string
    int src_size = *(int*) data.c_str();
    data.erase(0, sizeof(int));
    char* const decompress_buffer = new char[src_size];
    if (decompress_buffer == NULL) {
      LOG(INFO) << "Failed to allocate memory for *decompress_buffer.";
    }

    const int decompressed_size = LZ4_decompress_safe(
        data.c_str(), decompress_buffer, data.size(), src_size);
    if (decompressed_size < 0) {
      LOG(INFO) << "A negative result from LZ4_decompress_safe indicates a "
                   "failure trying to decompress the data.  See exit code "
                   "(echo $?) for value returned.";
    }
    if (decompressed_size >= 0) {
      LOG(INFO) << "We successfully decompressed some data!";
    }
    // if (decompressed_size != data.size()) {
    //     LOG(INFO) << "Decompressed data is different from original! \n";
    // }
    data = std::string(decompress_buffer, decompressed_size);
    delete[] decompress_buffer;

    std::vector<std::vector<int>> token_list;
    std::vector<void*> data_list;
    std::vector<int> data_size_list;
    std::vector<uint64_t> timestamp_list;
    std::vector<std::vector<int>> sub_tree_token_list;
    std::vector<void*> sub_tree_data_list;
    std::vector<int> sub_tree_data_size_list;
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

      size_t dataSize = dataPart.length() / 2; // Each byte is represented by two hex characters
      if (isMainTree) {
        data_size_list.push_back(dataSize);
      } else {
        sub_tree_data_size_list.push_back(dataSize);
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
          reinterpret_cast<unsigned char*>(data)[i] = static_cast<unsigned
          char>(byte);
        }
      }
      if (isMainTree) {
        token_list.push_back(keys);
        timestamp_list.push_back(timestamp);
        data_list.push_back(data);
      } else {
        sub_tree_token_list.push_back(keys);
        sub_tree_data_list.push_back(data);
      }
    }

    // This pointer will be freed by upper layer. Because this data
    // is created by upper layer. Here just recover it from serialized
    // string.
    // TBD
    // set the right cache capacity.
    std::shared_ptr<RadixTree> radix_tree = std::make_shared<RadixTree>(100);
    radix_tree->node_count = token_list.size();

    raxShow(radix_tree->tree);
    for (size_t i = 0; i < token_list.size(); i++) {
      std::string token_str = "";
      for (size_t j = 0; j < token_list[i].size(); j++) {
        token_str += std::to_string(token_list[i][j]);
      }
      LOG(INFO) << "token:" << token_str;
      int* insert_tokens_array = token_list[i].data();
      size_t insert_tokens_array_len = token_list[i].size();
      nodeData* data = new nodeData();
      data->data = data_list[i];
      data->data_length = data_size_list[i];
      raxNode* dataNode = NULL;
      int retval = raxInsertAndReturnDataNode(
          radix_tree->tree, insert_tokens_array, insert_tokens_array_len, data,
          (void**) &dataNode, NULL);
      // VINEYARD_ASSERT(retval == 1);
      if (dataNode == NULL) {
        throw std::runtime_error("Insert token list failed");
      }
      dataNode->timestamp = timestamp_list[i];
    }
    LOG(INFO) << "start to insert sub tree token list" << std::endl;
    for (size_t i = 0; i < sub_tree_token_list.size(); i++) {
      for (size_t j = 0; j < sub_tree_token_list[i].size(); j++) {
        LOG(INFO) << sub_tree_token_list[i][j];
      }

      raxNode* node = nullptr;
      LOG(INFO) << "stage 1";
      VINEYARD_ASSERT(radix_tree->tree != nullptr);

      // TBD refator this code.
      // raxFindNode(radix_tree->tree, sub_tree_token_list[i].data(),
      //                       sub_tree_token_list[i].size(), (void **)&node);
      // if (node->size > 1) {
      std::vector<int> prefix;
      node = raxFindAndReturnDataNode(radix_tree->tree, sub_tree_token_list[i].data(),
                            sub_tree_token_list[i].size(), prefix, NULL, false);
      // }
      VINEYARD_ASSERT(node != nullptr);
      LOG(INFO) << "stage 2";
      customData* data = new customData();
      data->data = sub_tree_data_list[i];
      LOG(INFO) << sub_tree_data_list[i];
      data->data_length = sub_tree_data_size_list[i];

      LOG(INFO) << "stage 3";
      node->issubtree = true;
      raxSetCustomData(node, data);

      // rax* rax_tree = raxNew();
      // rax_tree->head = node;
      std::shared_ptr<RadixTree> sub_tree = std::make_shared<RadixTree>(radix_tree->tree, radix_tree->cache_capacity, sub_tree_token_list[i]);
      // TBD
      // refactor this code.
      radix_tree->sub_tree.insert(sub_tree);
    }
    LOG(INFO) << "Deserialize success";
    LOG(INFO) << "set size:" << radix_tree->sub_tree.size();
    return radix_tree;
  }

  std::shared_ptr<RadixTree> SplitInternal(std::vector<int> tokens) {
    std::vector<int> prefix;
    nodeData* dummy_data = new nodeData();
    raxNode* sub_tree_root_node =
        raxSplit(this->tree, tokens.data(), tokens.size(), dummy_data, prefix);

    // TBD
    // if the sub_tree is null, delete this pointer.
    // rax* sub_rax = raxNew();
    // sub_rax->head = sub_tree_root_node;
    std::shared_ptr<RadixTree> sub_tree =
        std::make_shared<RadixTree>(this->tree, this->cache_capacity, prefix);
    LOG(INFO) << "split end";
    raxShow(this->tree);
    return sub_tree;
  }

  // Get child node list from this tree.
  static std::vector<std::shared_ptr<NodeWithTreeAttri>>
  TraverseTreeWithoutSubTree(std::shared_ptr<RadixTree> radix_tree) {
    std::vector<std::shared_ptr<NodeWithTreeAttri>> nodes;
    if (radix_tree == NULL) {
      LOG(INFO) << "traverse failed";
      return nodes;
    }


    LOG(INFO) << "travel:" << radix_tree << " headnode:" << radix_tree->tree->head;
    std::vector<raxNode*> dataNodeList;
    // raxNode* headNode = radix_tree->tree->head;
    raxShow(radix_tree->tree);
    std::string prefix_str;
    for (size_t i = 0; i < radix_tree->prefix.size(); i++) {
      prefix_str += std::to_string(radix_tree->prefix[i]) + " ";
    }
    LOG(INFO) << "prefix:" << prefix_str;
    std::vector<int> pre_tmp;
    raxNode* headNode = (raxNode*)raxFindAndReturnDataNode(radix_tree->tree, radix_tree->prefix.data(), radix_tree->prefix.size(), pre_tmp, NULL);
    LOG(INFO) << "head node:" << headNode;
    raxTraverseSubTree(headNode, dataNodeList);
    LOG(INFO) << "data node list:" << dataNodeList.size();
    for (size_t i = 0; i < dataNodeList.size(); i++) {
      nodes.push_back(std::make_shared<NodeWithTreeAttri>(
          std::make_shared<Node>(dataNodeList[i]), radix_tree));
    }
    return nodes;
  }

  rax* GetTree() {return this->tree;}

  void* GetCustomData() {
    // VINEYARD_ASSERT(tree->head->custom_data != nullptr);
    // LOG(INFO) << "get custom data with node:" << tree->head;
    // return ((customData *)tree->head->custom_data)->data;
    std::vector<int> pre;
    raxNode* node = raxFindAndReturnDataNode(this->tree, prefix.data(), prefix.size(), pre, NULL);
    LOG(INFO) << "node:" << node;
    std::string prefix_str = "";
    for (size_t i = 0; i < prefix.size(); i++) {
      prefix_str += std::to_string(prefix[i]) + " ";
    }
    LOG(INFO) << "prefix:" << prefix_str;
    raxShow(this->tree);
    VINEYARD_ASSERT(node->custom_data != nullptr);
    return ((customData *)node->custom_data)->data;
  }

  void SetCustomData(void* custom_data, int custom_data_length) {
    LOG(INFO) << "set custom data";
    customData* data = new customData();
    data->data = custom_data;
    data->data_length = custom_data_length;
    std::vector<int> pre;
    std::string prefix_str = "";
    for (size_t i = 0; i < prefix.size(); i++) {
      prefix_str += std::to_string(prefix[i]) + " ";
    }
    LOG(INFO) << "prefix:" << prefix_str;
    raxNode* node = raxFindAndReturnDataNode(this->tree, prefix.data(), prefix.size(), pre, NULL);
    LOG(INFO) << "node:" << node;
    node->issubtree = true;
    raxShow(this->tree);
    raxSetCustomData(node, data);

  }

  rax* GetRootTree() { return tree; }

  int GetCacheCapacity() { return cache_capacity; }

  std::set<std::shared_ptr<RadixTree>> GetSubTreeSet() { return sub_tree; }
};

#endif
