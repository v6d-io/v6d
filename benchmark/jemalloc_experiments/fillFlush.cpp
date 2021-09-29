#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "gflags/gflags.h"

// Flags that control single-threaded behavior
DEFINE_int32(batch_size, 1000, "Number of pointers owned by a thread at a time");
DEFINE_int32(batch_thread_migrations, 200, "Number of pointers to free in a run");
DEFINE_int32(batch_frees, 200, "Number of pointers in a batch to free");
DEFINE_int32(batch_sleep_ms, 1, "Number of milliseconds to sleep between batch free and alloc");

// Flags that control cross-thread behavior
DEFINE_int32(num_threads, 1, "Number of threads to run the test");
DEFINE_int32(shared_buffer_size, 10 * 1000, "Shared buffer size");

// Control parameters
DEFINE_int32(num_runs, -1, "Number of runs to perform (or -1 to loop forever)");
DEFINE_int32(malloc_size, 32, "Size of the allocations");
DEFINE_int32(randseed, 12345, "Random seed, for gesture in the direction of reproducibility");

typedef std::minstd_rand URNG;

std::vector<std::atomic<void*>> createSharedBuffer(URNG& urng) {
  std::vector<void*> resultNonAtomic(FLAGS_shared_buffer_size);
  for (int i = 0; i < FLAGS_shared_buffer_size; ++i) {
    resultNonAtomic[i] = std::malloc(FLAGS_malloc_size);
  }
  std::shuffle(resultNonAtomic.begin(), resultNonAtomic.end(), urng);

  std::vector<std::atomic<void*>> result(resultNonAtomic.begin(), resultNonAtomic.end());
  return result;
}

void doThreadMigrations(
    URNG& urng,
    std::vector<void*>& batch,
    std::vector<std::atomic<void*>>& sharedBuffer) {
  std::uniform_int_distribution<int> sharedDist(0, sharedBuffer.size());
  std::uniform_int_distribution<int> localDist(0, batch.size());

  for (int i = 0; i < FLAGS_batch_thread_migrations; ++i) {
    int localIndex = localDist(urng);
    int sharedIndex = sharedDist(urng);
    void* oldLocal = batch[localIndex];
    void* newLocal = sharedBuffer[sharedIndex].exchange(oldLocal);
    batch[localIndex] = newLocal;
  }
}

void doFrees(URNG& urng, std::vector<void*>& batch) {
  for (int i = 0; i < FLAGS_batch_frees; ++i) {
    std::free(batch[i]);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(FLAGS_batch_sleep_ms));
  for (int i = 0; i < FLAGS_batch_frees; ++i) {
    batch[i] = std::malloc(FLAGS_malloc_size);
  }
  std::shuffle(batch.begin(), batch.end(), urng);
}

void doThread(unsigned initSeed, std::vector<std::atomic<void*>>& sharedBuffer) {
  std::vector<void*> batch(FLAGS_batch_size);
  for (int i = 0; i < FLAGS_batch_size; ++i) {
    batch[i] = std::malloc(FLAGS_malloc_size);
  }
  URNG urng(initSeed);
  for (unsigned i = 0; i < (unsigned) FLAGS_num_runs || FLAGS_num_runs == -1; ++i) {
    doThreadMigrations(urng, batch, sharedBuffer);
    doFrees(urng, batch);
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  URNG urng(FLAGS_randseed);
  auto sharedBuffer = createSharedBuffer(urng);
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < FLAGS_num_threads; ++i) {
    unsigned seed = (unsigned)urng() + i;
    threads.emplace_back([&, seed]() {
      doThread(seed, sharedBuffer);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}
