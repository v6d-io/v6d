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

#include <iostream>
#include <random>
#include <vector>

#include "common/util/logging.h"

#include "llm-cache/hash/hasher.h"

using namespace vineyard;  //  NOLINT(build/namespaces)

constexpr int BATCHSIZE = 16;
constexpr int SPLITNUMBER = 2;
constexpr int TOKENLISTSIZE = 100000;

std::vector<int> generate_random_tokens(size_t max_length) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(1, 10000);

  size_t length = max_length;
  std::vector<int> tokens(length);
  for (size_t i = 0; i < length; ++i) {
    tokens[i] = dist(gen);
  }
  return tokens;
}

void test_with_tokens(IHashAlgorithm* hash_algorithm,
                      const std::string& hash_name) {
  LOG(INFO) << "start to test the " << hash_name << " with tokens";
  Hasher hasher(hash_algorithm);

  // test the hash with the tokens less than the batch size
  std::vector<int> tokens = generate_random_tokens(10);
  std::vector<std::string> paths;
  VINEYARD_CHECK_OK(
      hasher.computePathForTokens(tokens, BATCHSIZE, SPLITNUMBER, paths));
  VINEYARD_ASSERT(paths.size() == 0);

  // test the hash with the tokens more than the batch size
  std::vector<std::string> paths1, paths2;
  std::vector<int> tokens1 = generate_random_tokens(17);
  std::vector<int> tokens2 = generate_random_tokens(18);
  VINEYARD_CHECK_OK(
      hasher.computePathForTokens(tokens1, BATCHSIZE, SPLITNUMBER, paths1));
  VINEYARD_CHECK_OK(
      hasher.computePathForTokens(tokens2, BATCHSIZE, SPLITNUMBER, paths2));
  VINEYARD_ASSERT(paths1.size() == paths1.size());

  paths.clear();
  tokens = generate_random_tokens(100);
  VINEYARD_CHECK_OK(
      hasher.computePathForTokens(tokens, BATCHSIZE, SPLITNUMBER, paths));
  VINEYARD_ASSERT(paths.size() == size_t(100 / 16));
  LOG(INFO) << "Passed the " << hash_name << " test of tokens";
}

void test_accuracy(IHashAlgorithm* hash_algorithm,
                   const std::string& hash_name) {
  LOG(INFO) << "start to test the accuracy of the " << hash_name;
  MurmurHash3Algorithm hash;
  Hasher hasher(&hash);
  std::vector<std::string> paths1;
  std::vector<std::string> paths2;
  for (int i = 0; i < 100; i++) {
    std::vector<int> tokens = generate_random_tokens(100);
    VINEYARD_CHECK_OK(
        hasher.computePathForTokens(tokens, BATCHSIZE, SPLITNUMBER, paths1));
    VINEYARD_CHECK_OK(
        hasher.computePathForTokens(tokens, BATCHSIZE, SPLITNUMBER, paths2));
  }

  VINEYARD_ASSERT(paths1.size() == paths2.size());
  for (size_t i = 0; i < paths1.size(); i++) {
    VINEYARD_ASSERT(paths1[i] == paths2[i]);
  }

  LOG(INFO) << "Passed the accuracy test of the " << hash_name;
}

// test the hash conflict of the MurmurHash3Algorithm and CityHashAlgorithm
void test_hash_conflict() {
  LOG(INFO) << "start to test the hash conflict of the MurmurHash3Algorithm";

  MurmurHash3Algorithm murmur_hash;
  Hasher murmur_hasher(&murmur_hash);

  CityHashAlgorithm city_hash;
  Hasher city_hasher(&city_hash);

  std::map<std::string, int> murmur_hash_paths_map;
  std::map<std::string, int> city_hash_paths_map;

  std::vector<std::string> murmur_hash_paths;
  std::vector<std::string> city_hash_paths;

  int murmur_hash_conflict_count = 0;
  int city_hash_conflict_count = 0;
  int token_size = 0;

  std::map<std::vector<int>, int> tokens_map;
  for (int i = 0; i < TOKENLISTSIZE; i++) {
    std::vector<int> tokens = generate_random_tokens(16);
    tokens_map[tokens]++;
    token_size += tokens.size();
    VINEYARD_CHECK_OK(murmur_hasher.computePathForTokens(
        tokens, BATCHSIZE, SPLITNUMBER, murmur_hash_paths));
    VINEYARD_CHECK_OK(city_hasher.computePathForTokens(
        tokens, BATCHSIZE, SPLITNUMBER, city_hash_paths));
  }

  for (size_t i = 0; i < murmur_hash_paths.size(); i++) {
    murmur_hash_paths_map[murmur_hash_paths[i]]++;
  }
  for (size_t i = 0; i < city_hash_paths.size(); i++) {
    city_hash_paths_map[city_hash_paths[i]]++;
  }

  for (auto iter = murmur_hash_paths_map.begin();
       iter != murmur_hash_paths_map.end(); iter++) {
    if (iter->second > 1) {
      murmur_hash_conflict_count += (iter->second - 1);
    }
  }

  for (auto iter = city_hash_paths_map.begin();
       iter != city_hash_paths_map.end(); iter++) {
    if (iter->second > 1) {
      city_hash_conflict_count += (iter->second - 1);
    }
  }

  LOG(INFO) << "MurmurHash3Algorithm conflict count is "
            << murmur_hash_conflict_count << " / " << token_size
            << "CityHashAlgorithm conflict count is "
            << city_hash_conflict_count << " / " << token_size;
}

int main(int argc, char** argv) {
  MurmurHash3Algorithm murmur_hash;
  CityHashAlgorithm city_hash;
  test_with_tokens(&murmur_hash, "murmurhash");
  test_with_tokens(&city_hash, "cityhash");
  test_accuracy(&murmur_hash, "murmurhash");
  test_accuracy(&city_hash, "cityhash");
  test_hash_conflict();
  return 0;
}
