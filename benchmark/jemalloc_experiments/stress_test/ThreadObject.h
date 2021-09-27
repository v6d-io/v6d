#pragma once

#include <exception>
#include <mutex>
#include <queue>
#include <thread>

#include "Allocation.h"

class ThreadObject {
public:
  // frees all allocations whose lifetime has elapsed
  void free();
  // free all allocations, even if the lifetime hasn't expired
  void freeIgnoreLifetime();
  // Add an allocation to be freed after a particular time
  void addToFree(Allocation a);
  // calls malloc, or return [nullptr] if the simulation should be done
  void *allocate(size_t sz);

  // get the current time for this threads logical clock
  int currentPhase() const;

  // the time that the simulation will stop (according to this thread's logical
  // clock)
  int maxPhase() const;

  ThreadObject();

private:
  std::mutex lock_;
  std::priority_queue<Allocation, std::vector<Allocation>,
                      std::greater<Allocation>>
      q_;
  size_t allocSoFar_;
};
