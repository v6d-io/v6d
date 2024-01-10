#ifndef RADIX_TREE_H
#define RADIX_TREE_H

extern "C" {
#include "radix.h"
}
#include "common/util/logging.h"

#include <memory>
#include <map>
#include <vector>

typedef struct nodeData {
  void* data;
  int data_length;
}nodeData;

class Node {
 private:
  raxNode *node;
  void* data;
  int data_length;

 public:
  Node(raxNode *node) {
    this->node = node;
    this->data = NULL;
    this->data_length = 0;
  }
  void set_data(void *data, int data_length) {
    nodeData *node_data = new nodeData();
    node_data->data = data;
    node_data->data_length = data_length;
    raxSetData(this->node, data);
  }

  void* get_data() {
    nodeData *nodedata = (nodeData *)raxGetData(this->node);
    return nodedata->data; 
  }
};

class RadixTree;

class NodeWithTreeAttri {
 private:
  Node *node;
  RadixTree *belong_to;

 public:
  NodeWithTreeAttri(Node *node, void *belong_to) {
    this->node = node;
    this->belong_to = (RadixTree *)belong_to;
  }

  Node *get_node() { return node; }

  RadixTree *get_tree() { return belong_to; }
};

class RadixTree {
 private:
   void *custom_data;
   int custom_data_length;
   rax *tree;
 public:
  RadixTree() {
    LOG(INFO) << "init radix tree";
    this->tree = raxNew();
    this->custom_data = NULL;
    this->custom_data_length = 0;
  }

  RadixTree(void *custom_data, int custom_data_length) {
    LOG(INFO) << "init radix tree with custom data";
    this->tree = raxNew();
    this->custom_data = custom_data;
    this->custom_data_length = custom_data_length;
  }

  void *GetCustomData() {
    return custom_data;
  }

  void insert(const std::vector<int> key, void* data, int data_length) {
    const int* tokens = key.data();
    size_t tokens_len = key.size();
    
    nodeData *insert_data = new nodeData();
    ((nodeData*)data)->data = data;
    ((nodeData*)data)->data_length = data_length;
    int retval = raxInsert(this->tree, tokens, tokens_len, insert_data, NULL);
    if (retval == 0) {
      if (errno == 0) {
        LOG(INFO) << "overwrite an existing token list";
      } else {
        LOG(INFO) << "insert failed with errno:" + std::to_string(errno);
      }
    } else {
        LOG(INFO) << "insert success";
    }
  }

  NodeWithTreeAttri *insert(const std::vector<int> tokens, int next_token) {
    // build the token vector with next_token that will be deleted
    const std::vector<int>& tokens_ref = tokens;
    std::vector<int> insert_tokens = const_cast<std::vector<int>&>(tokens_ref);
    insert_tokens.push_back(next_token);

    // insert the token vector to the radix tree
    int* insert_tokens_array = insert_tokens.data();
    size_t insert_tokens_array_len = insert_tokens.size();
    nodeData *dummy_data = new nodeData();
    raxNode *dataNode = raxInsertAndReturnDataNode(this->tree, insert_tokens_array, insert_tokens_array_len, dummy_data, NULL);
    if (dataNode == NULL) {
      LOG(INFO) << "insert failed";
      return NULL;
    }
    LOG(INFO) << "insert success";
    return new NodeWithTreeAttri(new Node(dataNode), this);
  }

  void Delete(const std::vector<int> tokens, int next_token) {
    // build the token vector with next_token that will be deleted
    const std::vector<int>& tokens_ref = tokens;
    std::vector<int> delete_tokens = const_cast<std::vector<int>&>(tokens_ref);
    delete_tokens.push_back(next_token);

    // remove the token vector from the radix tree
    int* delete_tokens_array = delete_tokens.data();
    size_t delete_tokens_array_len = delete_tokens.size();
    int retval = raxRemove(this->tree, delete_tokens_array, delete_tokens_array_len, NULL);
    if (retval==1) {
      LOG(INFO) << "remove success";
    } else {
      LOG(INFO) << "remove failed";
    }
  }

  void insert(const std::vector<int>& prefix, int key, void* data,
              int data_length) {
    std::vector<int> key_vec = prefix;
    key_vec.push_back(key);

    const int* tokens = key_vec.data();
    size_t tokens_len = key_vec.size();
    
    nodeData *insert_data = new nodeData();
    ((nodeData*)data)->data = data;
    ((nodeData*)data)->data_length = data_length;
    int retval = raxInsert(this->tree, tokens, tokens_len, insert_data, NULL);
    if (retval == 0) {
      if (errno == 0) {
        LOG(INFO) << "overwrite an existing token list";
      } else {
        LOG(INFO) << "insert failed with errno:" + std::to_string(errno);
      }
    } else {
        LOG(INFO) << "insert success";
    }
  }

  NodeWithTreeAttri* get(std::vector<int> key) {
    int* tokens = key.data();
    size_t tokens_len = key.size();

    raxNode *dataNode = raxFindAndReturnDataNode(this->tree, tokens, tokens_len);
    if (dataNode == NULL) {
      LOG(INFO) << "get failed";
      return NULL;
    }
    LOG(INFO) << "get success";
    return new NodeWithTreeAttri(new Node(dataNode), this);
  }

  NodeWithTreeAttri* get(std::vector<int> prefix, int key) {
    std::vector<int> key_vec = prefix;
    key_vec.push_back(key);
    return get(key_vec);
  }

  std::string serialize() { return std::string("this is a serialized string"); }

  static RadixTree *deserialize(std::string data) {
    LOG(INFO) << "deserialize with data:" + data;
    return new RadixTree();
  }

  RadixTree* split() {
    LOG(INFO) << "splits is not implemented";
    return this;
  }

  // Get child node list from this tree.
  std::vector<NodeWithTreeAttri *> traverse() {
    if (this->tree == NULL) {
      LOG(INFO) << "traverse failed";
      return std::vector<NodeWithTreeAttri *>();
    }
    std::vector<NodeWithTreeAttri *> nodes;

    int numele = this->tree->numele;
    raxNode **dataNodeList = (raxNode **)malloc(sizeof(raxNode*)*(numele));
    raxNode **current = dataNodeList;
    raxNode *headNode = this->tree->head;
    raxTraverse(headNode, &dataNodeList);
    for (int i = 0; i < numele; i++, current++) {
        nodes.push_back(new NodeWithTreeAttri(new Node(*current), this));
    }
    return nodes;
  }

  void *get_custom_data() {
    return custom_data;
  }

  void set_custom_data(void *custom_data, int custom_data_length) {
    this->custom_data = custom_data;
    this->custom_data_length = custom_data_length;
  }
};

#endif