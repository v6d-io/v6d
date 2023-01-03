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

#include <future>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "grape/worker/comm_spec.h"

#include "common/util/status.h"
#include "graph/utils/thread_group.h"

namespace vineyard {

ThreadGroup::ThreadGroup(tid_t parallelism) : parallelism_(parallelism) {
  tid_.store(0);
  stopped_.store(false);
  running_task_num_.store(0);

  for (size_t i = 0; i < parallelism_; ++i) {
    workers_.emplace_back([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->mutex_);
          this->condition_.wait(lock, [this] {
            return this->stopped_.load() || !this->pending_tasks_.empty();
          });
          if (this->stopped_.load() && this->pending_tasks_.empty()) {
            return;
          }
          task = std::move(this->pending_tasks_.front());
          this->pending_tasks_.pop();
        }
        // go
        running_task_num_.fetch_add(1);
        task();
        running_task_num_.fetch_sub(1);
      }
    });
  }
}

ThreadGroup::ThreadGroup(const grape::CommSpec& comm_spec)
    : ThreadGroup(
          (std::thread::hardware_concurrency() + comm_spec.local_num() - 1) /
          comm_spec.local_num()) {}

ThreadGroup::~ThreadGroup() {
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    stopped_.store(true);
  }

  while (getRunningThreadNum() > 0) {
    std::this_thread::yield();
  }

  condition_.notify_all();
  for (std::thread& worker : workers_) {
    worker.join();
  }
}

ThreadGroup::return_t ThreadGroup::TaskResult(tid_t tid) {
  auto fu_it = tasks_.find(tid);
  return fu_it->second.get();
}

std::vector<ThreadGroup::return_t> ThreadGroup::TakeResults() {
  std::vector<return_t> results;
  auto it = tasks_.begin();

  while (it != tasks_.end()) {
    auto& fu = it->second;

    results.push_back(fu.get());
    it = tasks_.erase(it);
  }
  return results;
}

size_t ThreadGroup::getRunningThreadNum() { return running_task_num_.load(); }

DynamicThreadGroup::DynamicThreadGroup(tid_t parallelism)
    : parallelism_(parallelism), tid_(0), stopped_(false) {}

DynamicThreadGroup::DynamicThreadGroup(const grape::CommSpec& comm_spec)
    : tid_(0), stopped_(false) {
  parallelism_ =
      (std::thread::hardware_concurrency() + comm_spec.local_num() - 1) /
      comm_spec.local_num();
}

DynamicThreadGroup::~DynamicThreadGroup() {
  stopped_ = true;
  while (getRunningThreadNum() > 0) {
    std::this_thread::yield();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  while (!finished_threads_.empty()) {
    finished_threads_.front().join();
    finished_threads_.pop();
  }
}

DynamicThreadGroup::return_t DynamicThreadGroup::TaskResult(tid_t tid) {
  auto fu_it = tasks_.find(tid);
  return fu_it->second.get();
}

std::vector<DynamicThreadGroup::return_t> DynamicThreadGroup::TakeResults() {
  std::vector<return_t> results;
  auto it = tasks_.begin();

  while (it != tasks_.end()) {
    auto& fu = it->second;

    results.push_back(fu.get());
    it = tasks_.erase(it);
  }
  return results;
}

size_t DynamicThreadGroup::getRunningThreadNum() {
  std::lock_guard<std::mutex> lock(mutex_);
  return threads_.size();
}

}  // namespace vineyard
