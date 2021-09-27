#include "Producers.h"

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "jemalloc/include/jemalloc/jemalloc.h"

void *allocateAndUse(ThreadObject &myThread, size_t &memUsed, size_t sz) {
  void *ptr = myThread.allocate(sz);
  memUsed += sz;
  if (ptr != nullptr) {
    memset(ptr, 0, sz);
  }
  return ptr;
}

// Simple Producer

SimpleProducer::SimpleProducer(int allocSize, int numAllocs)
    : allocSize_(allocSize), allocsLeft_(numAllocs) {}
Allocation SimpleProducer::run(ThreadObject &myThread, size_t memUsageHint,
                               ProducerStatus &retStatus) {
  size_t memUsed = 0;
  while (true) {
    if (this->allocsLeft_ <= 0) {
      retStatus = ProducerStatus::Done;
      return Allocation();
    }
    if (memUsed >= memUsageHint) {
      retStatus = ProducerStatus::Yield;
      return Allocation();
    }
    void *ptr = allocateAndUse(myThread, memUsed, this->allocSize_);
    if (ptr == nullptr) {
      retStatus = ProducerStatus::AllocationFailed;
      return Allocation();
    }
    this->allocsLeft_ -= 1;
#if defined(BENCH_SYSTEM)
    free(ptr);
#elif defined(BENCH_JEMALLOC)
    vineyard_je_free(ptr);
#elif defined(BENCH_VINEYARD)
    vineyard_free(ptr);
#elif defined(BENCH_ARENA)
    vineyard_arena_free(ptr);
#else
    free(ptr);
#endif
  }
}

void SimpleProducer::cleanup() {}

// Vector Producer

VectorProducer::VectorProducer(size_t maxSize, size_t initialSize, int lifetime)
    : maxSize_(maxSize), initialSize_(initialSize), currentSize_(0),
      ptr_(nullptr), lifetime_(lifetime) {}

Allocation VectorProducer::run(ThreadObject &myThread, size_t memUsageHint,
                               ProducerStatus &retStatus) {
  size_t memUsed = 0;

  if (this->currentSize_ == 0) {
    this->ptr_ = allocateAndUse(myThread, memUsed, this->initialSize_);
    if (this->ptr_ == nullptr) {
      retStatus = ProducerStatus::AllocationFailed;
      return Allocation();
    }
    this->currentSize_ = this->initialSize_;
  }

  while (true) {
    if (this->currentSize_ >= this->maxSize_) {
      retStatus = ProducerStatus::Done;
      return Allocation({this->ptr_}, this->lifetime_);
    }
    if (memUsed >= memUsageHint) {
      retStatus = ProducerStatus::Yield;
      return Allocation();
    }

#if defined(BENCH_SYSTEM)
    free(this->ptr_);
#elif defined(BENCH_JEMALLOC)
    vineyard_je_free(this->ptr_);
#elif defined(BENCH_VINEYARD)
    vineyard_free(this->ptr_);
#elif defined(BENCH_ARENA)
    vineyard_arena_free(this->ptr_);
#else
    free(this->ptr_);
#endif
    this->currentSize_ *= 2;
    this->ptr_ = allocateAndUse(myThread, memUsed, this->currentSize_);
    if (ptr_ == nullptr) {
      retStatus = ProducerStatus::AllocationFailed;
      return Allocation();
    }
  }
}

void VectorProducer::cleanup() {
  if (this->ptr_ != nullptr) {
#if defined(BENCH_SYSTEM)
    free(this->ptr_);
#elif defined(BENCH_JEMALLOC)
    vineyard_je_free(this->ptr_);
#elif defined(BENCH_VINEYARD)
    vineyard_free(this->ptr_);
#elif defined(BENCH_ARENA)
    vineyard_arena_free(this->ptr_);
#else
    free(this->ptr_);
#endif
  }
}

// LinkedList Producer

Allocation LinkedListProducer::run(ThreadObject &myThread, size_t memUsageHint,
                                   ProducerStatus &retStatus) {
  size_t memUsed = 0;

  while (true) {
    if (this->nodesRemaining_ <= 0) {
      retStatus = ProducerStatus::Done;
      return Allocation(std::move(this->toFree_), this->lifetime_);
    }
    if (memUsed >= memUsageHint) {
      retStatus = ProducerStatus::Yield;
      return Allocation();
    }
    void *newNode = allocateAndUse(myThread, memUsed, this->nodeSize_);
    if (newNode == nullptr) {
      retStatus = ProducerStatus::AllocationFailed;
      return Allocation();
    }
    nodesRemaining_ -= 1;
    this->toFree_.push_back(newNode);
  }
}

void LinkedListProducer::cleanup() {
  for (auto &ptr : this->toFree_) {
#if defined(BENCH_SYSTEM)
    free(ptr);
#elif defined(BENCH_JEMALLOC)
    vineyard_je_free(ptr);
#elif defined(BENCH_VINEYARD)
    vineyard_free(ptr);
#elif defined(BENCH_ARENA)
    vineyard_arena_free(ptr);
#else
    free(ptr);
#endif
  }
}

// allocate [numNodes] blocks of size [nodeSize] with lifetime [lifetime]
LinkedListProducer::LinkedListProducer(size_t nodeSize, int numNodes,
                                       int lifetime)
    : nodeSize_(nodeSize), nodesRemaining_(numNodes), lifetime_(lifetime),
      toFree_() {
  this->toFree_.reserve(numNodes);
}
