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

#ifndef MODULES_LLM_CACHE_HASH_HASHER_H_
#define MODULES_LLM_CACHE_HASH_HASHER_H_

#include <string>
#include <vector>

#include "common/util/status.h"
#include "llm-cache/hash/hash_algorithm.h"

namespace vineyard {

class Hasher {
 public:
  explicit Hasher(IHashAlgorithm* algorithm) { hashAlgorithm = algorithm; }

  /**
   * @brief Compute the path list for the token list
   *
   * @param tokenList The list of tokens
   * @param batchSize The size of the batch
   * @param splitNumber The number of splits
   * @param pathList The Relative path list of the token list
   *
   * @return Status
   *
   * @example
   *      Assume the token list is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   *      and the batch size is 3, the split number is 2.
   *
   *      The hash value will be computed for each batch as follows:
   *      1 2 3 -> hashValue1
   *      4 5 6 -> hashValue2
   *      7 8 9 -> hashValue3
   *      10 -> Abandoned(less than the batch size)
   *
   *      Then the hash value will be split into the path as follows:
   *      hashValue1(3e91df34) -> 3e/91/df/34
   *      hashValue2(3691d935) -> 36/91/d9/35
   *      hashValue3(4c90a490) -> 4c/90/a4/90
   *
   */
  Status computePathForTokens(const std::vector<int>& tokenList, int batchSize,
                              int splitNumber,
                              std::vector<std::string>& pathList) {
    char hashBuffer[9];
    int tokenSize = tokenList.size() - tokenList.size() % batchSize;
    // if the token list (upper_bound) is less than the batch size, then return
    // directly
    if (tokenSize < batchSize) {
      return Status::OK();
    }

    // split the token list into batches
    for (int i = 0; i < tokenSize; i += batchSize) {
      int hashValue =
          hashAlgorithm->hash(reinterpret_cast<const char*>(tokenList.data()),
                              (i + batchSize) * sizeof(int));
      // split the hash value into paths
      std::snprintf(hashBuffer, sizeof(hashBuffer), "%08x", hashValue);
      int index = 0;
      std::string path;
      while (index + splitNumber < 8) {
        path += std::string(hashBuffer + index, splitNumber) + "/";
        index += splitNumber;
      }
      path += std::string(hashBuffer + index, 8 - index);
      pathList.push_back(path);
    }
    return Status::OK();
  }

 private:
  IHashAlgorithm* hashAlgorithm;
};

}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_HASH_HASHER_H_
