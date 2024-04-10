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

#ifndef MODULES_LLM_CACHE_THREAD_GROUP_H_
#define MODULES_LLM_CACHE_THREAD_GROUP_H_

#include <future>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__has_include) && __has_include(<version>)
#include <version>
#endif

#include "common/util/status.h"

#ifdef __cpp_lib_is_invocable
template <class T, typename... Args>
using result_of_t = std::invoke_result_t<T, Args...>;
#else
template <class T, typename... Args>
using result_of_t = typename std::result_of<T(Args...)>::type;
#endif

namespace vineyard {
namespace parallel {

class ThreadGroup {
  using tid_t = uint32_t;
  // Returns the path index and task status for parallel execution.
  // The path index is used to identify and delete results of unsuccessful
  // tasks.
  using return_t = std::pair<int, Status>;

 public:
  explicit ThreadGroup(
      uint32_t parallelism = std::thread::hardware_concurrency());

  ThreadGroup(const ThreadGroup&) = delete;
  ThreadGroup(ThreadGroup&&) = delete;

  template <class F, class... Args>
  tid_t AddTask(F&& f, Args&&... args) {
    static_assert(std::is_same<return_t, result_of_t<F, Args...>>::value,
                  "The return type of the task must be `Status`");
    if (stopped_.load()) {
      throw std::runtime_error("ThreadGroup is stopped");
    }

    auto task_wrapper = [](F&& _f, auto&&... _args) -> return_t {
      try {
        return std::move(_f(std::forward<Args>(_args)...));
      } catch (std::exception& e) {
        return std::pair(-1, Status(StatusCode::kUnknownError, e.what()));
      }
    };

    auto task = std::make_shared<std::packaged_task<return_t()>>(
        std::bind(std::move(task_wrapper), std::forward<F>(f),
                  std::forward<Args>(args)...));

    tid_t current_task_id = tid_.fetch_add(1);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopped_.load()) {
        throw std::runtime_error("ThreadGroup is stopped");
      }
      pending_tasks_.emplace([task]() { (*task)(); });
      tasks_[current_task_id] = task->get_future();
    }
    condition_.notify_one();
    return current_task_id;
  }

  ~ThreadGroup();

  return_t TaskResult(tid_t tid);

  std::vector<return_t> TakeResults();

 private:
  size_t getRunningThreadNum();

  tid_t parallelism_;
  std::atomic<tid_t> tid_;
  std::atomic_bool stopped_;
  std::atomic<size_t> running_task_num_;
  std::unordered_map<tid_t, std::future<return_t>> tasks_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::queue<std::function<void()>> pending_tasks_;
};

/**
 * @brief A thread group that dynamically allocate a new thread for each tasks.
 *
 * @AddTask@ will be blocked until there are spare thread resources.
 */
class DynamicThreadGroup {
  using tid_t = uint32_t;
  using return_t = Status;

 public:
  explicit DynamicThreadGroup(
      tid_t parallelism = std::thread::hardware_concurrency());

  DynamicThreadGroup(const DynamicThreadGroup&) = delete;
  DynamicThreadGroup(DynamicThreadGroup&&) = delete;

  template <class F, class... Args>
  tid_t AddTask(F&& f, Args&&... args) {
    static_assert(std::is_same<return_t, result_of_t<F, Args...>>::value,
                  "The return type of the task must be `Status`");
    if (stopped_) {
      throw std::runtime_error("DynamicThreadGroup is stopped");
    }
    while (getRunningThreadNum() >= parallelism_) {
      std::lock_guard<std::mutex> lock(mutex_);

      while (!finished_threads_.empty()) {
        finished_threads_.front().join();
        finished_threads_.pop();
      }
      std::this_thread::yield();
    }

    auto task_wrapper = [this](tid_t tid, F&& _f, auto&&... _args) -> return_t {
      return_t v;

      try {
        v = std::move(_f(std::forward<Args>(_args)...));
      } catch (std::exception& e) {
        v = Status(StatusCode::kUnknownError, e.what());
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_threads_.push(std::move(threads_.at(tid)));
        threads_.erase(tid);
      }
      return v;
    };

    tid_t current_task_id = tid_.fetch_add(1);
    auto task = std::make_shared<std::packaged_task<return_t()>>(
        std::bind(std::move(task_wrapper), current_task_id, std::forward<F>(f),
                  std::forward<Args>(args)...));

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopped_.load()) {
        throw std::runtime_error("ThreadGroup is stopped");
      }

      threads_.emplace(current_task_id, std::thread([task]() { (*task)(); }));
      tasks_[current_task_id] = task->get_future();
    }
    return current_task_id;
  }

  ~DynamicThreadGroup();

  return_t TaskResult(tid_t tid);

  std::vector<return_t> TakeResults();

 private:
  size_t getRunningThreadNum();

  tid_t parallelism_;
  std::atomic<tid_t> tid_;
  std::atomic_bool stopped_;
  std::unordered_map<tid_t, std::thread> threads_;
  std::unordered_map<tid_t, std::future<return_t>> tasks_;
  std::queue<std::thread> finished_threads_;
  std::mutex mutex_;
};

}  // namespace parallel
}  // namespace vineyard

#endif  // MODULES_LLM_CACHE_THREAD_GROUP_H_
