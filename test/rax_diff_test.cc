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

#include <stdio.h>
#include <unistd.h>
#include <memory>
#include "kv-state-cache/radix-tree/radix.h"

int key_1[] = {1, 2};
int key_2[] = {1, 3};
int key_3[] = {1, 4};
int key_4[] = {1, 3, 1};
int key_5[] = {1, 3, 2};

void insert(rax* rt, int* key, int len) {
  for (int i = 1; i <= len; i++) {
    raxInsert(rt, key, i, NULL, NULL);
  }
}

int main(int argc, char** argv) {
  rax* rt_1 = raxNew();
  rax* rt_2 = raxNew();

  int max_node = argc > 1 ? atoi(argv[1]) : 3;

  // raxInsert(rt_2, key_1, 2, NULL, NULL);
  // raxInsert(rt_2, key_2, 2, NULL, NULL);

  // raxInsert(rt_1, key_3, 2, NULL, NULL);
  // raxInsert(rt_1, key_4, 3, NULL, NULL);
  // raxInsert(rt_1, key_5, 3, NULL, NULL);

  insert(rt_1, key_3, 2);
  insert(rt_1, key_4, 3);

  sleep(1);

  insert(rt_2, key_1, 2);
  insert(rt_2, key_2, 2);

  sleep(1);

  insert(rt_1, key_5, 3);

  raxShow(rt_1);
  printf("==============================\n");
  raxShow(rt_2);
  printf("==============================\n");

  testIteRax(rt_1);
  printf("==============================\n");
  testIteRax(rt_2);
  printf("==============================\n");

  std::vector<std::vector<int>> evicted_tokens;
  std::set<std::vector<int>> insert_tokens;
  mergeTree(rt_1, rt_2, evicted_tokens, insert_tokens, max_node);

  printf("evicted_tokens:\n");
  for (size_t i = 0; i < evicted_tokens.size(); i++) {
    for (size_t j = 0; j < evicted_tokens[i].size(); j++) {
      printf("%d ", evicted_tokens[i][j]);
    }
    printf("\n");
  }
  for (size_t i = 0; i < evicted_tokens.size(); i++) {
    raxRemove(rt_1, evicted_tokens[i].data(), evicted_tokens[i].size(), NULL, false);
  }

  for (auto it = insert_tokens.begin(); it != insert_tokens.end(); it++) {
    raxInsert(rt_1, const_cast<int*>(it->data()), it->size(), NULL,
              NULL, false);
  }

  raxShow(rt_1);
  printf("==============================\n");
  raxShow(rt_2);
  printf("==============================\n");

  testIteRax(rt_1);
  printf("==============================\n");
  testIteRax(rt_2);
  printf("==============================\n");

  return 0;
}