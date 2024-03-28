/** Copyright 2020-2024 Giulio Ermanno Pibiri and Roberto Trani
 *
 * The following sets forth attribution notices for third party software.
 *
 * PTHash:
 * The software includes components licensed by Giulio Ermanno Pibiri and
 * Roberto Trani, available at https://github.com/jermp/pthash
 *
 * Licensed under the MIT License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/MIT
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "pthash/essentials/essentials.hpp"

#include "pthash/encoders/compact_vector.hpp"
#include "pthash/encoders/ef_sequence.hpp"

#include <cassert>
#include <unordered_map>
#include <vector>

namespace pthash {

template <typename Iterator>
std::pair<std::vector<uint64_t>, std::vector<uint64_t>>
compute_ranks_and_dictionary(Iterator begin, uint64_t n) {
  // accumulate frequencies
  std::unordered_map<uint64_t, uint64_t> distinct;
  for (auto it = begin, end = begin + n; it != end; ++it) {
    auto find_it = distinct.find(*it);
    if (find_it != distinct.end()) {  // found
      (*find_it).second += 1;
    } else {
      distinct[*it] = 1;
    }
  }
  std::vector<std::pair<uint64_t, uint64_t>> vec;
  vec.reserve(distinct.size());
  for (auto p : distinct)
    vec.emplace_back(p.first, p.second);
  std::sort(vec.begin(), vec.end(),
            [](const std::pair<uint64_t, uint64_t>& x,
               const std::pair<uint64_t, uint64_t>& y) {
              return x.second > y.second;
            });
  distinct.clear();
  // assign codewords by non-increasing frequency
  std::vector<uint64_t> dict;
  dict.reserve(distinct.size());
  for (uint64_t i = 0; i != vec.size(); ++i) {
    auto p = vec[i];
    distinct.insert({p.first, i});
    dict.push_back(p.first);
  }

  std::vector<uint64_t> ranks;
  ranks.reserve(n);
  for (auto it = begin, end = begin + n; it != end; ++it)
    ranks.push_back(distinct[*it]);
  return {ranks, dict};
}

struct dictionary {
  template <typename Iterator>
  void encode(Iterator begin, uint64_t n) {
    auto [ranks, dict] = compute_ranks_and_dictionary(begin, n);
    m_ranks.build(ranks.begin(), ranks.size());
    m_dict.build(dict.begin(), dict.size());
  }

  static std::string name() { return "dictionary"; }

  size_t size() const { return m_ranks.size(); }

  size_t num_bits() const { return (m_ranks.bytes() + m_dict.bytes()) * 8; }

  uint64_t access(uint64_t i) const {
    uint64_t rank = m_ranks.access(i);
    return m_dict.access(rank);
  }

  template <typename Visitor>
  void visit(Visitor& visitor) {
    visitor.visit(m_ranks);
    visitor.visit(m_dict);
  }

  template <typename Loader>
  void load(Loader& loader) {
    m_ranks.load(loader);
    m_dict.load(loader);
  }

  template <typename Dumper>
  void dump(Dumper& dumper) const {
    m_ranks.dump(dumper);
    m_dict.dump(dumper);
  }

 private:
  compact_vector m_ranks;
  compact_vector m_dict;
};

template <typename Front, typename Back>
struct dual {
  template <typename Iterator>
  void encode(Iterator begin, uint64_t n) {
    size_t front_size = n * 0.3;
    m_front.encode(begin, front_size);
    m_back.encode(begin + front_size, n - front_size);
  }

  static std::string name() { return Front::name() + "-" + Back::name(); }

  size_t num_bits() const { return m_front.num_bits() + m_back.num_bits(); }

  uint64_t access(uint64_t i) const {
    if (i < m_front.size())
      return m_front.access(i);
    return m_back.access(i - m_front.size());
  }

  template <typename Visitor>
  void visit(Visitor& visitor) {
    visitor.visit(m_front);
    visitor.visit(m_back);
  }

  template <typename Loader>
  void load(Loader& loader) {
    m_front.load(loader);
    m_back.load(loader);
  }

  template <typename Dumper>
  void dump(Dumper& dumper) const {
    m_front.dump(dumper);
    m_back.dump(dumper);
  }

 private:
  Front m_front;
  Back m_back;
};

/* dual encoders */
typedef dual<dictionary, dictionary> dictionary_dictionary;

}  // namespace pthash
