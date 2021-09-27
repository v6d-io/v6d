#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <gflags/gflags.h>
#include "jemalloc/include/jemalloc/jemalloc.h"
#include "Distribution.h"
#include "Mixer.h"

DEFINE_int32(num_threads, 1, "number of threads to run");
DEFINE_bool(print_malloc_stats, false, "print out malloc stats after running");
DEFINE_string(distribution_file, "", "path to distribution file");
static bool validateDistributionFile(const char *flagName,
                                     const std::string &val) {
  return val.length() != 0;
}
DEFINE_validator(distribution_file, &validateDistributionFile);

using std::shared_ptr;
using std::vector;

void createAndRunMixer(const Distribution *distr, int me,
                       vector<shared_ptr<ThreadObject>> threadObjects) {
  Mixer m(distr, me, threadObjects);
  m.run();
}

double run() {
  initInstBurner();
  Distribution distr = parseDistribution(FLAGS_distribution_file.c_str());

  // Set up a work queue for each thread
  vector<std::thread> threads;
  vector<shared_ptr<ThreadObject>> threadObjects;
  for (int i = 0; i < FLAGS_num_threads; i++) {
    auto threadObject = shared_ptr<ThreadObject>(new ThreadObject());
    threadObjects.push_back(threadObject);
  }

  for (int i = 0; i < FLAGS_num_threads; i++) {
    // each thread gets an arbitrary id given by [i]
    threads.push_back(std::thread(createAndRunMixer, &distr, i, threadObjects));
  }

  using namespace std::chrono;

  high_resolution_clock::time_point beginTime = high_resolution_clock::now();
  for (auto &t : threads) {
    t.join();
  }

  // Cleanup any remaining memory
  for (auto &t : threadObjects) {
    t->freeIgnoreLifetime();
  }
  high_resolution_clock::time_point endTime = high_resolution_clock::now();
  duration<double> span = duration_cast<duration<double>>(endTime - beginTime);

  return span.count();
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  double time = run();

  if (FLAGS_print_malloc_stats) {
    if (vineyard_je_mallctl("thread.tcache.flush", NULL, NULL, NULL, 0)) {
      std::cout << "je_mallctl failed. Exiting..." << std::endl;
    }
      vineyard_je_malloc_stats_print(NULL, NULL, NULL);
  }

  std::cout << "Elapsed time: " << time << std::endl;
}
