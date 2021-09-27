#pragma once

#include <vector>

#include "ThreadObject.h"

enum class ProducerStatus {
  // the producer tried to allocate memory from the thread object, but it failed
  AllocationFailed,
  // the producer did everything it was supposed to do and can be unregistered
  Done,
  // the producer ran for some time, and deferred it's execution to the mixer
  Yield
};

class Producer {
public:
  /* Run the producer using [myThread] to allocate memory. The producer should
   * use approximately [memUsageHint] bytes of memory, but the accuracy to which
   * this is done is up to individual producers. Set [retStatus] to tell the
   * mixer what to do next. */
  virtual Allocation run(ThreadObject &myThread, size_t memUsageHint,
                         ProducerStatus &retStatus) = 0;
  /* Called on each Producer when this thread stops simulating.
   * Frees any memory stored in the producer. */
  virtual void cleanup() = 0;
};

// allocates a vector of size [sz]
class VectorProducer : public Producer {
public:
  Allocation run(ThreadObject &myThread, size_t memUsageHint,
                 ProducerStatus &retStatus);
  // allocate, and then free after [lifetime] has elapsed
  VectorProducer(size_t maxSize, size_t initialSize, int lifetime);
  void cleanup();

private:
  size_t maxSize_;
  size_t initialSize_;
  int lifetime_;
  size_t currentSize_;
  void *ptr_;
};

/* allocates a block of size [alloc_sz], and then immediately frees it. Repeats
 * this [n_allocs] times. */
class SimpleProducer : public Producer {
public:
  Allocation run(ThreadObject &myThread, size_t memUsageHint,
                 ProducerStatus &retStatus);
  SimpleProducer(int allocSize, int numAllocs);
  void cleanup();

private:
  int allocSize_;
  int allocsLeft_;
};

// Allocates many similarly sized blocks, and then frees them all at once later.
class LinkedListProducer : public Producer {
public:
  Allocation run(ThreadObject &myThread, size_t memUsageHint,
                 ProducerStatus &retStatus);
  // allocate [numNodes] blocks of size [nodeSize] with lifetime [lifetime]
  LinkedListProducer(size_t nodeSize, int numNodes, int lifetime);
  void cleanup();

private:
  size_t nodeSize_;
  int nodesRemaining_;
  int lifetime_;
  std::vector<void *> toFree_;
};
