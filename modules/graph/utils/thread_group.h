/** Copyright 2020-2022 Alibaba Group Holding Limited.

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

#ifndef MODULES_GRAPH_UTILS_THREAD_GROUP_H_
#define MODULES_GRAPH_UTILS_THREAD_GROUP_H_

#include <future>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/util/status.h"
#include "graph/utils/error.h"

namespace vineyard {
class ThreadGroup {
  using tid_t = uint32_t;
  using return_t = Status;

 public:
  explicit ThreadGroup(tid_t parallelism = std::thread::hardware_concurrency())
      : parallelism_(parallelism), tid_(0), stopped_(false) {}

  template <class F_T, class... ARGS_T>
  tid_t AddTask(F_T&& f, ARGS_T&&... args) {
    if (stopped_) {
      throw std::runtime_error("ThreadGroup is stopped");
    }
    while (getRunningThreadNum() >= parallelism_) {
      std::lock_guard<std::mutex> lg(mutex_);

      while (!finished_threads_.empty()) {
        finished_threads_.front().join();
        finished_threads_.pop();
      }
      std::this_thread::yield();
    }

    auto task_wrapper = [this](tid_t tid, F_T&& _f,
                               auto&&... _args) -> return_t {
      return_t v;

      try {
        v = std::move(_f(std::forward<ARGS_T>(_args)...));
      } catch (std::runtime_error& e) {
        v = Status(StatusCode::kUnknownError, e.what());
      }

      std::lock_guard<std::mutex> lg(mutex_);

      finished_threads_.push(std::move(threads_.at(tid)));
      threads_.erase(tid);
      return v;
    };

    auto task = std::make_shared<std::packaged_task<return_t()>>(
        std::bind(std::move(task_wrapper), tid_, std::forward<F_T>(f),
                  std::forward<ARGS_T>(args)...));

    std::lock_guard<std::mutex> lg(mutex_);

    threads_.emplace(tid_, std::thread([task]() { (*task)(); }));
    tasks_[tid_] = task->get_future();
    return tid_++;
  }

  ~ThreadGroup() {
    stopped_ = true;
    while (getRunningThreadNum() > 0) {
      std::this_thread::yield();
    }

    std::lock_guard<std::mutex> lg(mutex_);

    while (!finished_threads_.empty()) {
      finished_threads_.front().join();
      finished_threads_.pop();
    }
  }

  return_t TaskResult(tid_t tid) {
    auto fu_it = tasks_.find(tid);
    return fu_it->second.get();
  }

  std::vector<return_t> TakeResults() {
    std::vector<return_t> results;
    auto it = tasks_.begin();

    while (it != tasks_.end()) {
      auto& fu = it->second;

      results.push_back(fu.get());
      it = tasks_.erase(it);
    }
    return results;
  }

 private:
  size_t getRunningThreadNum() {
    std::unique_lock<std::mutex> lk(mutex_);
    return threads_.size();
  }

  tid_t parallelism_;
  tid_t tid_;
  bool stopped_;
  std::unordered_map<tid_t, std::thread> threads_;
  std::unordered_map<tid_t, std::future<return_t>> tasks_;
  std::queue<std::thread> finished_threads_;
  std::mutex mutex_;
};
}  // namespace vineyard
#endif  // MODULES_GRAPH_UTILS_THREAD_GROUP_H_
