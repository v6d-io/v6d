/* Rax -- A radix tree implementation.
 *
 * Copyright (c) 2017-2018, Salvatore Sanfilippo <antirez at gmail dot com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Redis nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef RADIX_H
#define RADIX_H

#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <vector>

/* Representation of a radix tree as implemented in this file, that contains
 * the token lists [1, 2, 3], [1, 2, 3, 4, 5, 6] and [1, 2, 3, 6, 7, 8] after
 * the insertion of each token list. When the node represents a key inside
 * the radix tree, we write it between [], otherwise it is written between ().
 *
 * This is the vanilla representation:
 *
 *              (1) []
 *                \
 *                (2) [1]
 *                  \
 *                  (3) [1,3]
 *                    \
 *                  [4   6] [1,2,3]
 *                  /     \
 *      [1,2,3,4] (5)     (7) [1,2,3,6]
 *                /         \
 *   [1,2,3,4,5] (6)         (8) [1,2,3,6,7]
 *              /              \
 *[1,2,3,4,5,6] []            [] [1,2,3,6,7,8]
 *
 * However, this implementation implements a very common optimization where
 * successive nodes having a single child are "compressed" into the node
 * itself as a list of tokens, each representing a next-level child,
 * and only the link to the node representing the last token node is
 * provided inside the representation. So the above representation is turned
 * into:
 *
 *                 ([1,2,3]) []
 *                     |
 *                  [4   6] [1,2,3]
 *                  /     \
 *     [1,2,3,4] ([5,6])    ([7,8]) [1,2,3,6]
 *                 /          \
 *  [1,2,3,4,5,6] []          [] [1,2,3,6,7,8]
 *
 * However this optimization makes the implementation a bit more complex.
 * For instance if a token list [1,1,2] is added in the above radix tree, a
 * "node splitting" operation is needed, since the [1,2,3] prefix is no longer
 * composed of nodes having a single child one after the other. This is the
 * above tree and the resulting node splitting after this event happens:
 *
 *
 *            (1) []
 *            /  \
 *   [1] ([1,2)  ([2,3]) [1]
 *                     \
 *                   [4   6] [1,2,3]
 *                   /     \
 *       [1,2,3,4] ([5,6])    ([7,8]) [1,2,3,6]
 *                 /          \
 *   [1,2,3,4,5,6] []          [] [1,2,3,6,7,8]
 *
 *
 * Similarly after deletion, if a new chain of nodes having a single child
 * is created (the chain must also not include nodes that represent keys),
 * it must be compressed back into a single node.
 *
 */

#define RAX_NODE_MAX_SIZE 1024
typedef struct raxNode {
  uint32_t iskey : 1;     /* Does this node contain a key? */
  uint32_t isnull : 1;    /* Associated value is NULL (don't store it). */
  uint32_t iscompr : 1;   /* Node is compressed. */
  uint32_t issubtree : 1; /* Node is the root node of a sub tree */
  uint32_t size : 26;     /* Number of children, or compressed string len. */
  uint32_t numnodes;      /* Number of the child nodes */
  uint32_t numele;        /* Number of elements inside this node. */
  uint64_t timestamp;     /* Timestamps of the node */
  uint32_t sub_tree_size; /* Number of nodes in the sub tree */
  void* custom_data;
  /* Data layout is as follows:
   *
   * If node is not compressed we have 'size' bytes, one for each children
   * token, and 'size' raxNode pointers, point to each child node.
   * Note how the character is not stored in the children but in the
   * edge of the parents:
   *
   * [header iscompr=0][1,2,3][1-ptr][2-ptr][3-ptr](value-ptr?)
   *
   * if node is compressed (iscompr bit is 1) the node has 1 children.
   * In that case the 'size' bytes of the string stored immediately at
   * the start of the data section, represent a sequence of successive
   * nodes linked one after the other, for which only the last one in
   * the sequence is actually represented as a node, and pointed to by
   * the current compressed node.
   *
   * [header iscompr=1][1,2,3][3-ptr](value-ptr?)
   *
   * Both compressed and not compressed nodes can represent a key
   * with associated data in the radix tree at any level (not just terminal
   * nodes).
   *
   * If the node has an associated key (iskey=1) and is not NULL
   * (isnull=0), then after the raxNode pointers pointing to the
   * children, an additional value pointer is present (as you can see
   * in the representation above as "value-ptr" field).
   */
  int data[];
} raxNode;

typedef struct rax {
  raxNode* head;
  raxNode* headDataNode;
  uint64_t numele;
  uint64_t numnodes;
} rax;

/* Stack data structure used by raxLowWalk() in order to, optionally, return
 * a list of parent nodes to the caller. The nodes do not have a "parent"
 * field for space concerns, so we use the auxiliary stack when needed. */
#define RAX_STACK_STATIC_ITEMS 32
typedef struct raxStack {
  void** stack; /* Points to static_items or an heap allocated array. */
  size_t items, maxitems; /* Number of items contained and total space. */
  /* Up to RAXSTACK_STACK_ITEMS items we avoid to allocate on the heap
   * and use this static array of pointers instead. */
  void* static_items[RAX_STACK_STATIC_ITEMS];
  int oom; /* True if pushing into this stack failed for OOM at some point. */
} raxStack;

/* Optional callback used for iterators and be notified on each rax node,
 * including nodes not representing keys. If the callback returns true
 * the callback changed the node pointer in the iterator structure, and the
 * iterator implementation will have to replace the pointer in the radix tree
 * internals. This allows the callback to reallocate the node to perform
 * very special operations, normally not needed by normal applications.
 *
 * This callback is used to perform very low level analysis of the radix tree
 * structure, scanning each possible node (but the root node), or in order to
 * reallocate the nodes to reduce the allocation fragmentation (this is the
 * Redis application for this callback).
 *
 * This is currently only supported in forward iterations (raxNext) */
typedef int (*raxNodeCallback)(raxNode** noderef);

/* Radix tree iterator state is encapsulated into this data structure. */
#define RAX_ITER_STATIC_LEN 128
#define RAX_ITER_JUST_SEEKED                                              \
  (1 << 0)                    /* Iterator was just seeked. Return current \
                                 element for the first iteration and      \
                                 clear the flag. */
#define RAX_ITER_EOF (1 << 1) /* End of iteration reached. */
#define RAX_ITER_SAFE                                \
  (1 << 2) /* Safe iterator, allows operations while \
              iterating. But it is slower. */
typedef struct raxIterator {
  int flags;
  rax* rt;        /* Radix tree we are iterating. */
  int* key;       /* The current string. */
  void* data;     /* Data associated to this key. */
  size_t key_len; /* Current key length. */
  size_t key_max; /* Max key len the current key buffer can hold. */
  int key_static_tokens[RAX_ITER_STATIC_LEN];
  bool add_to_subtree_list; /* Whether to add the current node to the subtree
                               list. */
  std::vector<std::vector<int>>* subtree_list; /* List of subtrees. */
  std::vector<void*>* subtree_data_list;       /* List of subtrees' data. */
  raxNode* node;           /* Current node. Only for unsafe iteration. */
  raxStack stack;          /* Stack used for unsafe iteration. */
  raxNodeCallback node_cb; /* Optional node callback. Normally set to NULL. */
} raxIterator;

// Wrapper for raxIterator to store the token list and its data node
// to reduce the memory usage.
typedef struct raxIteratorWrapper {
  std::vector<int> token_list; /* The current token list. */
  void* data;                  /* Data associated to this key. */
  raxNode* node;               /* Current node. Only for unsafe iteration. */
} raxIteratorWrapper;

/* A special pointer returned for not found items. */
extern void* raxNotFound;

/* Exported API. */
rax* raxNew(void);
int raxInsert(rax* rax, const std::vector<int>& token_list, void* data, void** old,
              bool set_timestamp = true);
int raxTryInsert(rax* rax, const std::vector<int>& token_list, void* data, void** old);
int raxInsertAndReturnDataNode(rax* rax, const std::vector<int>& token_list, void* data,
                               void** node, void** old);
int raxRemove(rax* rax, const std::vector<int>& token_list, void** old,
              bool set_timestamp = true);
void* raxFind(rax* rax, const std::vector<int>& token_list);
raxNode* raxFindAndReturnDataNode(rax* rax, const std::vector<int>& token_list,
                                  raxNode** sub_tree_node = NULL,
                                  bool set_timestamp = true);
void raxSetSubtree(raxNode* n);
void raxSetSubtreeAllocated(raxNode* node);
void raxSetSubtreeNotNull(raxNode* node);
int raxFindNodeWithParent(rax* rax, const std::vector<int>& token_list, void** node,
                          void** parent);
void raxFree(rax* rax);
void raxFreeWithCallback(rax* rax, void (*free_callback)(raxNode*));
void raxStart(raxIterator* it, rax* rt);
int raxSeek(raxIterator* it, const char* op, int* ele, size_t len);
int raxNext(raxIterator* it);
int raxPrev(raxIterator* it);
int raxRandomWalk(raxIterator* it, size_t steps);
int raxCompare(raxIterator* iter, const char* op, int* key, size_t key_len);
void raxStop(raxIterator* it);
int raxEOF(raxIterator* it);
std::string raxShow(rax* rax);
uint64_t raxSize(rax* rax);
void raxSetCustomData(raxNode* n, void* data);
void* raxGetCustomData(raxNode* n);
unsigned long raxTouch(raxNode* n);
void raxSetDebugMsg(int onoff);
void raxTraverse(raxNode* rax,
                 std::vector<std::shared_ptr<raxNode>>& dataNodeList);
void raxTraverseSubTree(raxNode* n, std::vector<raxNode*>& dataNodeList);
raxNode* raxSplit(rax* rax, const std::vector<int>& token_list, std::vector<int>& split_token_list);
void raxSerialize(rax* root, std::vector<std::vector<int>>& tokenList,
                  std::vector<void*>& dataList,
                  std::vector<uint64_t>& timestampsList,
                  std::vector<std::vector<int>>* subtreeList,
                  std::vector<void*>* subtreeNodeList);

/* Internal API. May be used by the node callback in order to access rax nodes
 * in a low level way, so this function is exported as well. */
void raxSetData(raxNode* n, void* data);
void* raxGetData(raxNode* n);
int raxFindNode(rax* rax, const std::vector<int>& token_list, void** node);
void raxFindLastRecentNode(raxNode* node, std::vector<int>& key);
void mergeTree(rax* first_tree, rax* second_tree,
               std::vector<std::vector<int>>& evicted_tokens,
               std::set<std::vector<int>>& insert_tokens, int max_node);
void testIteRax(rax* tree);
#endif
