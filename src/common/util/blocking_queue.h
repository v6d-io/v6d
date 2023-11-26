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

#ifndef SRC_COMMON_UTIL_BLOCKING_QUEUE_H_
#define SRC_COMMON_UTIL_BLOCKING_QUEUE_H_

#include <atomic>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <utility>

namespace vineyard {

/** A blocking queue for Job in the coordinator.*/

template <typename T>
class BlockingQueue {
 public:
  BlockingQueue() : size_limit_(std::numeric_limits<size_t>::max()) {}
  ~BlockingQueue() {}

  void SetLimit(size_t limit) { size_limit_ = limit; }

  void Push(const T& item) {
    {
      std::unique_lock<std::mutex> lk(lock_);
      while (queue_.size() >= size_limit_) {
        full_.wait(lk);
      }
      queue_.emplace_back(item);
    }
    empty_.notify_one();
  }

  void Push(T&& item) {
    {
      std::unique_lock<std::mutex> lk(lock_);
      while (queue_.size() >= size_limit_) {
        full_.wait(lk);
      }
      queue_.emplace_back(std::move(item));
    }
    empty_.notify_one();
  }

  T Pop() {
    {
      std::unique_lock<std::mutex> lk(lock_);
      while (queue_.empty()) {
        empty_.wait(lk);
      }
      T rc(std::move(queue_.front()));
      queue_.pop_front();
      full_.notify_one();
      return rc;
    }
  }

  void Get(T& item) {
    {
      std::unique_lock<std::mutex> lk(lock_);
      while (queue_.empty()) {
        empty_.wait(lk);
      }
      item = std::move(queue_.front());
      queue_.pop_front();
      full_.notify_one();
    }
  }

  size_t Size() const { return queue_.size(); }

  void Clear() { queue_.clear(); }

 private:
  std::deque<T> queue_;
  size_t size_limit_;
  std::mutex lock_;
  std::condition_variable empty_, full_;
};

template <typename T>
class PCBlockingQueue {
 public:
  PCBlockingQueue() : size_limit_(std::numeric_limits<size_t>::max()) {}
  ~PCBlockingQueue() {}

  void SetLimit(size_t limit) { size_limit_ = limit; }

  void DecProducerNum() {
    {
      std::unique_lock<std::mutex> lk(lock_);
      --producer_num_;
    }
    if (producer_num_ == 0) {
      empty_.notify_all();
    }
  }

  void SetProducerNum(int pn) { producer_num_ = pn; }

  void Put(const T& item) {
    {
      std::unique_lock<std::mutex> lk(lock_);
      while (queue_.size() >= size_limit_) {
        full_.wait(lk);
      }
      queue_.emplace_back(item);
    }
    empty_.notify_one();
  }

  void Put(T&& item) {
    {
      std::unique_lock<std::mutex> lk(lock_);
      while (queue_.size() >= size_limit_) {
        full_.wait(lk);
      }
      queue_.emplace_back(std::move(item));
    }
    empty_.notify_one();
  }

  bool Get(T& item) {
    {
      std::unique_lock<std::mutex> lk(lock_);
      while (queue_.empty() && (producer_num_ != 0)) {
        empty_.wait(lk);
      }
      if (queue_.empty() && (producer_num_ == 0)) {
        return false;
      } else {
        item = std::move(queue_.front());
        queue_.pop_front();
        full_.notify_one();
        return true;
      }
    }
  }

  size_t Size() const { return queue_.size(); }

  bool End() const { return queue_.empty() && (producer_num_ == 0); }

 private:
  std::deque<T> queue_;
  size_t size_limit_;
  std::mutex lock_;
  std::condition_variable empty_, full_;

  std::atomic<int> producer_num_;
};

class SpinLock {
  std::atomic_flag locked = ATOMIC_FLAG_INIT;

 public:
  void lock() {
    while (locked.test_and_set(std::memory_order_acquire)) {
      {}
    }
  }

  void unlock() { locked.clear(std::memory_order_relaxed); }
};

template <typename T>
class SpinBlockingQueue {
 public:
  SpinBlockingQueue() {}
  ~SpinBlockingQueue() {}

  void Push(const T& item) {
    lock.lock();
    queue_.emplace_back(item);
    lock.unlock();
  }

  void Push(T&& item) {
    lock.lock();
    queue_.emplace_back(std::move(item));
    lock.unlock();
  }

  void Get(T& item) {
    lock.lock();
    item = std::move(queue_.front());
    queue_.pop_front();
    lock.unlock();
  }

  size_t Size() const { return queue_.size(); }

  void Clear() { queue_.clear(); }

  bool Empty() const { return queue_.empty(); }

 private:
  std::deque<T> queue_;
  SpinLock lock;
};
}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_BLOCKING_QUEUE_H_
