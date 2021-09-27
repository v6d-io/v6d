#include "ThreadObject.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <mutex>
#include <thread>

#include <assert.h>
#include <gflags/gflags.h>

#include "SizeConstants.h"
#include "jemalloc/include/jemalloc/jemalloc.h"

// #define TCACHE_NONE
// #define VINEYARD_MALLOC

DEFINE_int64(alloc_per_thread, k1GB,
             "stop each thread after allocating this amount of memory");

DEFINE_int64(bytes_per_phase, k1MB, "bytes allocated per clock 'tick'");

void ThreadObject::free() {
  std::lock_guard<std::mutex> guard(this->lock_);

  while (!this->q_.empty() &&
         this->q_.top().freeAfterAbsolute <= this->currentPhase()) {
    this->q_.top().clear();
    this->q_.pop();
  }
}

void ThreadObject::freeIgnoreLifetime() {
  std::lock_guard<std::mutex> guard(this->lock_);

  while (!this->q_.empty()) {
    this->q_.top().clear();
    this->q_.pop();
  }
}

void ThreadObject::addToFree(Allocation a) {
  int absolutePhase = this->currentPhase() + a.freeAfterRelative;
  a.freeAfterAbsolute = absolutePhase;
  std::lock_guard<std::mutex> guard(this->lock_);
  this->q_.push(a);
}

void *ThreadObject::allocate(size_t sz) {
  if (FLAGS_alloc_per_thread <= this->allocSoFar_) {
    return nullptr;
  } else {
    this->allocSoFar_ += sz;
    assert(sz > 0);
#if defined(BENCH_SYSTEM)
    void *r = malloc(sz);
#elif defined(BENCH_JEMALLOC)
    void *r = vineyard_je_mallocx(sz, MALLOCX_TCACHE_NONE);;
#elif defined(BENCH_VINEYARD)
    void *r = vineyard_malloc(sz);
#elif defined(BENCH_ARENA)
    void *r = vineyard_arena_malloc(sz);
#else
    void *r = malloc(sz);
#endif
    // void *r = vineyard_je_mallocx(sz, MALLOCX_TCACHE_NONE);
    // void *r = vineyard::_ArenaAllocator().Allocate(sz);
    // void *r = vineyard_je_malloc(sz);
    if (r == nullptr) {
      std::cout << "malloc failed." << std::endl;
      exit(1);
    }
    return r;
  }
}

int ThreadObject::currentPhase() const {
  return this->allocSoFar_ / FLAGS_bytes_per_phase;
}

int ThreadObject::maxPhase() const {
  return FLAGS_alloc_per_thread / FLAGS_bytes_per_phase;
}

ThreadObject::ThreadObject() : allocSoFar_(0) {}
