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

#ifndef MODULES_GRAPH_UTILS_STRING_COLLECTION_H_
#define MODULES_GRAPH_UTILS_STRING_COLLECTION_H_

#include <string.h>

#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "boost/functional/hash.hpp"
#include "flat_hash_map/flat_hash_map.hpp"

namespace grape {

const uint64_t block_size = 64 * 1024 * 1024;
const uint64_t block_mask = block_size - 1;
const uint64_t block_offset = 26;
struct RefString {
  RefString() : str(NULL), len(0) {}
  RefString(const char* s, size_t l) : str(s), len(l) {}
  explicit RefString(const char* s) : str(s), len(strlen(s)) {}
  explicit RefString(const std::string& s) : str(s.data()), len(s.length()) {}
  std::string ToString() const {
    std::string res;
    res.resize(len);
    memcpy(&res[0], str, len);
    return res;
  }
  operator std::string() const {
    std::string ret;
    ret.resize(len);
    memcpy(&ret[0], str, len);
    return ret;
  }
  RefString& operator=(const std::string& other) {
    str = other.data();
    len = other.length();
    return *this;
  }
  const char* str;
  size_t len;
};
inline bool operator==(const RefString& lhs, const RefString& rhs) {
  return (lhs.len == rhs.len) && (0 == memcmp(lhs.str, rhs.str, lhs.len));
}
inline bool operator!=(const RefString& lhs, const RefString& rhs) {
  return !(lhs == rhs);
}

struct RSBlock {
  RSBlock() {
    data = static_cast<char*>(malloc(block_size));
    size = 0;
  }

  RSBlock(RSBlock&& rsb) noexcept {
    data = rsb.data;
    size = rsb.size;
    rsb.data = NULL;
    rsb.size = 0;
  }
  ~RSBlock() {
    if (data != NULL) {
      free(data);
    }
  }

  uint64_t remaining() const { return block_size - size; }
  bool enough(const std::string& str) { return remaining() > str.length(); }
  bool enough(const RefString& rs) { return remaining() > rs.len; }

  uint64_t append(const std::string& str) {
    uint64_t old_size = size;
    size += str.length();
    memcpy(&data[old_size], str.data(), str.length());
    return old_size;
  }

  uint64_t append(const RefString& rs) {
    uint64_t old_size = size;
    size += rs.len;
    memcpy(&data[old_size], rs.str, rs.len);
    return old_size;
  }

  template <typename IOADAPTOR_T>
  void Read(std::unique_ptr<IOADAPTOR_T>& io_adaptor) {
    CHECK(io_adaptor->Read(&size, sizeof(uint64_t)));
    if (data == NULL) {
      data = static_cast<char*>(malloc(block_size));
    }
    CHECK(io_adaptor->Read(data, size));
  }

  template <typename IOADAPTOR_T>
  void Write(std::unique_ptr<IOADAPTOR_T>& io_adaptor) {
    CHECK(io_adaptor->Write(&size, sizeof(uint64_t)));
    if (size) {
      CHECK(io_adaptor->Write(data, size));
    }
  }

  void SendTo(int worker_id, MPI_Comm comm, int tag = 0) const {
    MPI_Send(&size, sizeof(size_t), MPI_CHAR, worker_id, tag, comm);
    MPI_Send(data, static_cast<int>(size), MPI_CHAR, worker_id, tag, comm);
  }

  void RecvFrom(int worker_id, MPI_Comm comm, int tag = 0) {
    MPI_Recv(&size, sizeof(size_t), MPI_CHAR, worker_id, tag, comm,
             MPI_STATUS_IGNORE);
    MPI_Recv(data, static_cast<int>(size), MPI_CHAR, worker_id, tag, comm,
             MPI_STATUS_IGNORE);
  }

  char* data;
  uint64_t size;
};

class RSVector {
 public:
  RSVector() : count_(0) {}
  RSVector(const RSVector& rhs) : buffer_(rhs.buffer_), count_(rhs.count_) {}
  RSVector(RSVector&& rhs) {
    buffer_.swap(rhs.buffer_);
    count_ = rhs.count_;
    rhs.count_ = 0;
    rhs.buffer_.clear();
  }
  RSVector& operator=(RSVector&& other) {
    buffer_.clear();
    count_ = 0;
    swap(other);
    return *this;
  }
  void swap(RSVector& other) noexcept {
    buffer_.swap(other.buffer_);
    std::swap(count_, other.count_);
  }
  operator std::vector<std::string>() const {
    std::vector<std::string> ret;
    ret.reserve(size());
    for (auto& rs : *this) {
      ret.emplace_back(rs.ToString());
    }
    return ret;
  }
  void emplace(const std::string& str) {
    size_t old_size = buffer_.size();
    size_t new_size = old_size + str.length() + 1;
    buffer_.resize(new_size);
    memcpy(&buffer_[old_size], str.data(), str.length());
    buffer_[new_size - 1] = '\0';
    ++count_;
  }
  void emplace(const RefString& rs) {
    size_t old_size = buffer_.size();
    size_t new_size = old_size + rs.len + 1;
    buffer_.resize(new_size);
    memcpy(&buffer_[old_size], rs.str, rs.len);
    buffer_[new_size - 1] = '\0';
    ++count_;
  }
  void emplace_back(const std::string& str) {
    size_t old_size = buffer_.size();
    size_t new_size = old_size + str.length() + 1;
    buffer_.resize(new_size);
    memcpy(&buffer_[old_size], str.data(), str.length());
    buffer_[new_size - 1] = '\0';
    ++count_;
  }
  void emplace_back(const RefString& rs) {
    size_t old_size = buffer_.size();
    size_t new_size = old_size + rs.len + 1;
    buffer_.resize(new_size);
    memcpy(&buffer_[old_size], rs.str, rs.len);
    buffer_[new_size - 1] = '\0';
    ++count_;
  }
  void append(const RSVector& rsv) {
    buffer_.insert(buffer_.end(), rsv.buffer_.begin(), rsv.buffer_.end());
    count_ += rsv.count_;
  }
  struct const_iterator {
    const_iterator() = default;
    explicit const_iterator(const char* str, size_t len = 0)
        : current(str, len) {}
    ~const_iterator() = default;
    friend bool operator==(const const_iterator& lhs,
                           const const_iterator& rhs) {
      return lhs.current.str == rhs.current.str;
    }
    friend bool operator!=(const const_iterator& lhs,
                           const const_iterator& rhs) {
      return lhs.current.str != rhs.current.str;
    }
    const_iterator& operator++() {
      current.str += (strlen(current.str) + 1);
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator copy(current.str);
      ++*this;
      return copy;
    }
    const RefString& operator*() const {
      current.len = strlen(current.str);
      return current;
    }
    const RefString* operator->() const { return std::addressof(current); }
    mutable RefString current;
  };
  const_iterator begin() const {
    const char* str = buffer_.data();
    return const_iterator(str);
  }
  const_iterator end() const {
    const char* str = buffer_.data() + static_cast<ptrdiff_t>(buffer_.size());
    return const_iterator(str);
  }
  char* data() { return buffer_.data(); }
  const char* data() const { return buffer_.data(); }
  size_t size_in_bytes() const { return buffer_.size(); }
  size_t capacity() const { return buffer_.capacity(); }
  size_t size() const { return count_; }
  void clear() {
    buffer_.clear();
    count_ = 0;
  }
  void resize(size_t size, size_t count = 0) {
    buffer_.resize(size);
    count_ = count;
  }
  void reserve(size_t cap) { buffer_.reserve(cap); }

 private:
  std::vector<char> buffer_;
  size_t count_;
};

class StringCollection {
 public:
  StringCollection() {}
  ~StringCollection() {}

  StringCollection(StringCollection&& rhs) noexcept
      : blocks_(std::move(rhs.blocks_)),
        addr_lists_(std::move(rhs.addr_lists_)) {}

  size_t Put(const RefString& str) {
    if (blocks_.empty()) {
      blocks_.resize(1);
    }
    uint64_t high = blocks_.size() - 1;
    if (!blocks_[high].enough(str)) {
      blocks_.resize(blocks_.size() + 1);
      ++high;
    }
    uint64_t low = blocks_[high].append(str);
    uint64_t offset = generate_addr(high, low);
    size_t res = addr_lists_.size();
    addr_lists_.emplace_back(offset, str.len);
    return res;
  }

  size_t Put(const std::string& str) {
    RefString rs(str);
    return Put(rs);
  }

  RefString PutString(const RefString& str) {
    if (blocks_.empty()) {
      blocks_.resize(1);
    }
    uint64_t high = blocks_.size() - 1;
    if (!blocks_[high].enough(str)) {
      blocks_.resize(blocks_.size() + 1);
      ++high;
    }
    uint64_t low = blocks_[high].append(str);
    uint64_t offset = generate_addr(high, low);
    addr_lists_.emplace_back(offset, str.len);
    return RefString(blocks_[high].data + static_cast<ptrdiff_t>(low), str.len);
  }

  RefString PutString(const std::string& str) {
    RefString rs(str);
    return PutString(rs);
  }

  bool Get(size_t id, std::string& str) const {
    RefString rs;
    if (Get(id, rs)) {
      str.resize(rs.len);
      memcpy(&str[0], rs.str, rs.len);
      return true;
    }
    return false;
  }

  bool Get(size_t id, RefString& rs) const {
    uint64_t high, low;
    const auto& ssa = addr_lists_[id];
    parse_addr(ssa.addr, high, low);
    rs.str = blocks_[high].data + static_cast<ptrdiff_t>(low);
    rs.len = ssa.length;
    return true;
  }

  size_t Count() const { return addr_lists_.size(); }

  void Swap(StringCollection& other) {
    std::swap(blocks_, other.blocks_);
    std::swap(addr_lists_, other.addr_lists_);
  }

  std::vector<RSBlock>& GetBuffer() { return blocks_; }
  const std::vector<RSBlock>& GetBuffer() const { return blocks_; }

  template <typename IOADAPTOR_T>
  void Read(std::unique_ptr<IOADAPTOR_T>& io_adaptor) {
    grape::OutArchive oa;
    io_adaptor->ReadArchive(oa);
    size_t block_num, al_size;
    oa >> block_num >> al_size;
    blocks_.resize(block_num);
    addr_lists_.resize(al_size);
    for (auto& block : blocks_) {
      block.Read(io_adaptor);
    }
    io_adaptor->Read(addr_lists_.data(), al_size * sizeof(rs_addr));
  }

  template <typename IOADAPTOR_T>
  void Write(std::unique_ptr<IOADAPTOR_T>& io_adaptor) {
    grape::InArchive ia;
    size_t block_num = blocks_.size();
    size_t al_size = addr_lists_.size();
    ia << block_num << al_size;
    io_adaptor->WriteArchive(ia);
    ia.Clear();
    for (auto& block : blocks_) {
      block.Write(io_adaptor);
    }
    io_adaptor->Write(addr_lists_.data(), al_size * sizeof(rs_addr));
  }

  void SendTo(int worker_id, MPI_Comm comm, int tag = 0) const {
    size_t block_num = blocks_.size();
    MPI_Send(&block_num, sizeof(size_t), MPI_CHAR, worker_id, tag, comm);
    for (auto& block : blocks_) {
      block.SendTo(worker_id, comm, tag);
    }
    grape::sync_comm::Send(addr_lists_, worker_id, tag, comm);
  }

  void RecvFrom(int worker_id, MPI_Comm comm, int tag = 0) {
    size_t block_num;
    MPI_Recv(&block_num, sizeof(size_t), MPI_CHAR, worker_id, tag, comm,
             MPI_STATUS_IGNORE);
    blocks_.resize(block_num);
    for (auto& block : blocks_) {
      block.RecvFrom(worker_id, comm, tag);
    }
    grape::sync_comm::Recv(addr_lists_, worker_id, tag, comm);
  }

 private:
  struct rs_addr {
    rs_addr() = default;
    rs_addr(uint64_t a, size_t l) : addr(a), length(l) {}
    ~rs_addr() = default;
    uint64_t addr;
    size_t length;
  };

  void parse_addr(uint64_t addr, uint64_t& high, uint64_t& low) const {
    high = (addr >> block_offset);
    low = (addr & block_mask);
  }

  uint64_t generate_addr(uint64_t high, uint64_t low) const {
    uint64_t res = high;
    res <<= block_offset;
    res |= low;
    return res;
  }

  std::vector<RSBlock> blocks_;
  std::vector<rs_addr> addr_lists_;
};
}  // namespace grape
namespace std {
template <>
struct hash<grape::RefString> {
  size_t operator()(const grape::RefString& rs) const noexcept {
    return boost::hash_range(rs.str, rs.str + rs.len);
  }
  // using hash_policy = ska::prime_number_hash_policy;
};

}  // namespace std

#endif  // MODULES_GRAPH_UTILS_STRING_COLLECTION_H_
