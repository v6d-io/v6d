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

#ifndef SRC_COMMON_UTIL_MONITOR_H_
#define SRC_COMMON_UTIL_MONITOR_H_

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifdef ENABLE_VINEYARD_MONITOR

#define MONITOR_START(__monitor) __monitor.StartTick();
#define MONITOR_END(__monitor) __monitor.EndTick()
#define DUMP_MONITOR(__monitors) __monitors.Dump()
#define DUMP_MONITOR_HEADER() vineyard::monitor::Monitor::DumpHeader()
#define MONITOR_AUTO(__monitor) \
  vineyard::monitor::MonitorGuard monitor##guard(__monitor)

#define MONITOR_CLEAR(__monitor, __name, __unit) \
  do {                                           \
    __monitor.Clear();                           \
    __monitor.SetName(__name);                   \
    __monitor.SetUnit(__unit);                   \
  } while (0)

#else

#define MONITOR_START(__monitor)
#define MONITOR_END(__monitor)
#define DUMP_MONITOR(__monitors)
#define DUMP_MONITOR_HEADER()
#define MONITOR_AUTO(__monitor)
#define MONITOR_CLEAR(__monitor, __name, __unit)

#endif

namespace vineyard {

namespace monitor {

enum UNIT { MICROSECONDS, MILLISECONDS, NANOSECONDS, SECONDS };

class Monitor {
 public:
  Monitor(std::string name, UNIT unit) : name_(std::move(name)), unit_(unit) {}

  Monitor() : name_("default_monitor"), unit_(MICROSECONDS) {}

  void StartTick() {
    if (started_) {
      return;
    }
    mutex_.lock();

    started_ = true;
    last_time_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch());
  }

  void EndTick() {
    if (!started_) {
      return;
    }
    mutex_.unlock();

    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    timestamp_.push_back(now - last_time_);
    last_time_ = now;
    started_ = false;
  }

  static void DumpHeader() {
    std::cout << "Dump Monitor Info:" << std::endl;
    size_t name_width = 20;
    size_t count_width = 10;
    size_t time_width = 18;
    std::cout << std::left << std::setw(name_width) << "Name"
              << std::setw(count_width) << "Count" << std::setw(time_width)
              << "Avg" << std::setw(time_width) << "Min"
              << std::setw(time_width) << "Max" << std::setw(time_width)
              << "P50" << std::setw(time_width) << "P99"
              << std::setw(time_width) << "Total" << std::endl;
  }

  void Dump() {
    if (timestamp_.empty()) {
      return;
    }
    std::sort(timestamp_.begin(), timestamp_.end());
    size_t count = timestamp_.size();
    std::chrono::nanoseconds total = std::chrono::nanoseconds::zero();
    for (const auto& t : timestamp_) {
      total += t;
    }
    std::string unit;
    uint64_t factor = 1;
    if (unit_ == NANOSECONDS) {
      unit = "ns";
    } else if (unit_ == MICROSECONDS) {
      unit = "us";
      factor = 1000;
    } else if (unit_ == MILLISECONDS) {
      unit = "ms";
      factor = 1000 * 1000;
    } else if (unit_ == SECONDS) {
      unit = "s";
      factor = 1000 * 1000 * 1000;
    } else {
      unit = "us";
    }
    auto avg = total / count;
    auto min = timestamp_.front();
    auto max = timestamp_.back();
    auto p50 = timestamp_[count / 2];
    auto p99 =
        timestamp_[std::min(count - 1, static_cast<size_t>(count * 0.99))];

    const int name_width = 20;
    const int count_width = 10;
    const int time_width = 18;

    std::cout << std::left << std::setw(name_width)
              << name_.substr(0, 17) + (name_.length() > 17 ? "..." : "")
              << std::setw(count_width) << count << std::setw(time_width)
              << format(static_cast<double>(avg.count()) / factor, unit)
              << std::setw(time_width)
              << format(static_cast<double>(min.count()) / factor, unit)
              << std::setw(time_width)
              << format(static_cast<double>(max.count()) / factor, unit)
              << std::setw(time_width)
              << format(static_cast<double>(p50.count()) / factor, unit)
              << std::setw(time_width)
              << format(static_cast<double>(p99.count()) / factor, unit)
              << std::setw(time_width)
              << format(static_cast<double>(total.count()) / factor, unit)
              << std::endl;
  }

  void Clear() {
    timestamp_.clear();
    last_time_ = std::chrono::nanoseconds::zero();
    started_ = false;
  }

  void SetUnit(UNIT unit) { unit_ = unit; }

  void SetName(const std::string& name) { name_ = name; }

  ~Monitor() = default;

 private:
  std::string format(double value, std::string unit) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3) << value << unit;
    return ss.str();
  }

  std::vector<std::chrono::nanoseconds> timestamp_;
  std::chrono::nanoseconds last_time_;
  bool started_ = false;
  std::string name_;
  UNIT unit_;
  std::mutex mutex_;
};

class MonitorGuard {
 public:
  explicit MonitorGuard(Monitor& monitor) : monitor_(monitor) {
    MONITOR_START(monitor_);
  }

  ~MonitorGuard() { MONITOR_END(monitor_); }

 private:
  Monitor& monitor_;
};

}  // namespace monitor

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_MONITOR_H_
