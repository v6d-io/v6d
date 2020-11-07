/** Copyright 2020 Alibaba Group Holding Limited.

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
#include <map>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "graph/utils/error.h"
#include "common/util/status.h"

namespace vineyard {
class ThreadGroup {
  using tid_t = uint32_t;
  using return_t = Status;

 public:
  template <class F_T, class... ARGS_T>
  tid_t AddTask(F_T&& f, ARGS_T&&... args) {
    auto task_wrapper = [](F_T&& _f, auto&&... _args) -> return_t {
      try {
        return _f(std::forward<ARGS_T>(_args)...);
      } catch (std::runtime_error& e) {
        return Status(StatusCode::kUnknownError, e.what());
      }
    };

    auto task = std::make_shared<std::packaged_task<return_t()>>(
        std::bind(std::move(task_wrapper), std::forward<F_T>(f),
                  std::forward<ARGS_T>(args)...));
    std::thread([task]() { (*task)(); }).detach();
    tasks_[tid_] = task->get_future();
    return tid_++;
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
  tid_t tid_;
  std::map<tid_t, std::future<return_t>> tasks_;
};
}  // namespace vineyard
#endif  // MODULES_GRAPH_UTILS_THREAD_GROUP_H_
