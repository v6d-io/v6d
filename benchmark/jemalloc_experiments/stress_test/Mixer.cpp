#include "Mixer.h"

#include <atomic>
#include <cstdint>
#include <iostream>

#include <assert.h>
#include <gflags/gflags.h>
#include "jemalloc/include/jemalloc/jemalloc.h"
#include <stdlib.h>
#include <sys/mman.h>

#include "SizeConstants.h"

using std::shared_ptr;
using std::unique_ptr;
using std::vector;

DEFINE_int32(max_producers, 5, "max producers per thread at one time");
DEFINE_int32(producer_scale_param, 100,
             "Vaguely scales the amount of stuff a "
             "single producer does, in a producer-defined way.");
DEFINE_double(peak_priority, 100.0, "Priority for bursty producers");
DEFINE_double(ramp_priority, 1.0, "Priority for background producers");

Mixer::Mixer(const Distribution *distr, int me,
             vector<shared_ptr<ThreadObject>> threadObjects)
    : distr_(distr), threadObjects_(threadObjects), me_(me),
      consumerIdPicker_(0, threadObjects.size() - 1) {
  std::vector<double> distributionWeights;
  for (auto &sizeClass : *distr) {
    distributionWeights.push_back(sizeClass.freq);
  }
  sizeClassPicker_ = std::discrete_distribution<int>(begin(distributionWeights),
                                                     end(distributionWeights));
  addProducers();
}

ThreadObject &Mixer::myThread() { return *this->threadObjects_[this->me_]; }

void Mixer::registerProducer(double weight, unique_ptr<Producer> p) {
  this->producers_.push_back(std::move(p));
  this->weightArray_.push_back(weight);
  this->producerPicker_ = std::discrete_distribution<int>(
      begin(this->weightArray_), end(this->weightArray_));
}

void Mixer::unregisterProducer(int index) {
  double weight = this->weightArray_[index];
  this->weightArray_.erase(begin(this->weightArray_) + index);
  this->producers_.erase(begin(this->producers_) + index);
  this->producerPicker_ = std::discrete_distribution<int>(
      begin(this->weightArray_), end(this->weightArray_));
}

void Mixer::addProducer() {
  int sizeClassIndex = this->sizeClassPicker_(this->generator_);
  SizeClass sizeClass = (*this->distr_)[sizeClassIndex];
  std::uniform_int_distribution<int> initialSizeFuzz(1, sizeClass.size / 2);

  std::uniform_int_distribution<int> strategyPicker(1, 8);
  int strategy = strategyPicker(this->generator_);

  double weight;
  Producer *p;
  int maxLifetime = this->myThread().maxPhase();
  std::uniform_int_distribution<int> longLifetime(maxLifetime / 10,
                                                  maxLifetime);
  std::uniform_int_distribution<int> shortLifetime(1, maxLifetime / 10);
  if (1 <= strategy && strategy <= 3) {
    /* allocate a ramp
     * - long lifetime
     * - low priority; slowly accumulates in the background */
    weight = FLAGS_ramp_priority;
    int lifetime = longLifetime(this->generator_);
    // VectorProducer
    if (strategy == 1) {
      p = new VectorProducer(sizeClass.size, initialSizeFuzz(this->generator_),
                             lifetime);
    }
    // LinkedListProducer
    if (strategy == 2) {
      p = new LinkedListProducer(sizeClass.size, FLAGS_producer_scale_param,
                                 lifetime);
    }
    // SimpleProducer
    if (strategy == 3) {
      p = new SimpleProducer(sizeClass.size, FLAGS_producer_scale_param);
    }
  } else if (4 <= strategy && strategy <= 5) {
    /* allocate a plateau
     * - finishes quickly
     * - long lifetime; stays for duration of program */
    weight = FLAGS_peak_priority;
    int lifetime = shortLifetime(this->generator_);
    // VectorProducer
    if (strategy == 4) {
      p = new VectorProducer(sizeClass.size, initialSizeFuzz(this->generator_),
                             lifetime);
    }
    // LinkedListProducer
    if (strategy == 5) {
      p = new LinkedListProducer(sizeClass.size, FLAGS_producer_scale_param,
                                 lifetime);
    }
  } else {
    weight = FLAGS_peak_priority;
    int lifetime = longLifetime(this->generator_);
    /* allocate a peak
     * - high priority
     * - finishes quickly
     * - short lifetime */
    // VectorProducer
    if (strategy == 6) {
      p = new VectorProducer(sizeClass.size, initialSizeFuzz(this->generator_),
                             lifetime);
    }
    // LinkedListProducer
    if (strategy == 7) {
      p = new LinkedListProducer(sizeClass.size, FLAGS_producer_scale_param,
                                 lifetime);
    }
    // SimpleProducer
    if (strategy == 8) {
      p = new SimpleProducer(sizeClass.size, FLAGS_producer_scale_param);
    }
  }

  assert(p != nullptr);

  this->registerProducer(weight, std::move(std::unique_ptr<Producer>(p)));
}

void Mixer::addProducers() {
  while (this->producers_.size() < FLAGS_max_producers) {
    this->addProducer();
  }
}

int Mixer::pickProducer() { return this->producerPicker_(this->generator_); }

// Picks next producer for the mixer to run. Currently uniform random choice
ThreadObject &Mixer::pickConsumer() {
  int consumerIndex = this->consumerIdPicker_(this->generator_);
  return *(this->threadObjects_[consumerIndex]);
}

constexpr size_t kMaxDataCacheSize = 8000000;
constexpr size_t kMaxInstCacheSize = 32000;

static char dataBurner[kMaxDataCacheSize] = {0};
static char *instBurner = nullptr;

constexpr unsigned char instRet = {0xC3};
constexpr unsigned char instNop = {0x90};

void burnDataCache(size_t n) {
  // Do something slightly non-trivial so this doesn't get optimized away
  size_t nClipped = (n > kMaxDataCacheSize) ? kMaxDataCacheSize : n;
  char c = dataBurner[0];
  for (int i = 0; i < nClipped; i++) {
    dataBurner[i] = c + 1;
  }
}

void initInstBurner() {
  size_t sz = kMaxInstCacheSize + 1;

  instBurner = (char *)mmap(NULL, sz, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  char *p = instBurner;
  for (int i = 0; i < sz - 1; ++i) {
    *(p++) = instNop;
  }
  *(p++) = instRet;
  if (mprotect(instBurner, sz, PROT_NONE) == -1) {
    std::cout << "mprotect failed" << std::endl;
    exit(1);
  }
  if (mprotect(instBurner, sz, PROT_EXEC | PROT_READ) == -1) {
    std::cout << "mprotect failed" << std::endl;
    exit(1);
  }
}

void burnInstCache(size_t n) {
  int nClipped = (n > kMaxInstCacheSize) ? kMaxInstCacheSize : n;
  int offset = kMaxInstCacheSize - nClipped;

  void (*f)() = (void (*)())(instBurner + offset);
  (*f)();
}

void Mixer::run() {
  while (true) {
    this->myThread().free();
    // otherwise run a random producer
    if (this->producers_.size() == 0) {
      std::cout << "ran out of producers" << std::endl;
      exit(0);
    }
    int producerIndex = this->pickProducer();
    ProducerStatus st;
    Allocation a =
        this->producers_[producerIndex]->run(this->myThread(), 100000, st);
    if (st == ProducerStatus::AllocationFailed) {
      for (auto &producer : this->producers_) {
        producer->cleanup();
      }
      break;
    } else if (st == ProducerStatus::Done) {
      this->unregisterProducer(producerIndex);
    }
    if (!a.isEmpty()) {
      this->pickConsumer().addToFree(std::move(a));
    }

    addProducers();

    burnInstCache(kMaxInstCacheSize);
  }
  if (vineyard_je_mallctl("thread.tcache.flush", NULL, NULL, NULL, 0)) {
    std::cout << "je_mallctl failed. Exiting..." << std::endl;
  }
  // Main loop will cleanup memory after all threads are done
}
