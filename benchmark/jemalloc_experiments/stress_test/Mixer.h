#pragma once

#include <random>
#include <vector>

#include "Distribution.h"
#include "Producers.h"
#include "ThreadObject.h"

void initInstBurner();

class Mixer {
public:
  void run();
  Mixer(const Distribution *distr, int me,
        std::vector<std::shared_ptr<ThreadObject>> threadObjects);

private:
  // the thread id that this mixer is running on
  int me_;
  // work queues for each thread indexed by thread number
  std::vector<std::shared_ptr<ThreadObject>> threadObjects_;
  /* Picks a consumer to free memory allocated by a producer. Currently uniform
   * random choice */
  ThreadObject &pickConsumer();

  std::uniform_int_distribution<int> consumerIdPicker_;
  std::default_random_engine generator_;

  // for picking producer with weighted random choice
  std::vector<double> weightArray_;
  std::vector<std::unique_ptr<Producer>> producers_;
  std::discrete_distribution<int> producerPicker_;
  // Picks the index of the next producer for the mixer to run. Uses
  // [producerPicker_].
  int pickProducer();

  // [pickProducer] constructs producers using [distr_] as a guideline
  const Distribution *distr_;
  /* Generated from [distr_]; generates indexes into [distr_] randomly, weighted
   * by the frequency of the size classes in [distr_]. */
  std::discrete_distribution<int> sizeClassPicker_;

  // add producers until [FLAGS_max_producers]
  void addProducers();
  // randomly choose a producer and add it to the mixer
  void addProducer();

  // get the thread object that this mixer is running on
  ThreadObject &myThread();

  // register [p] to get scheduled by the mixer with priority [weight]
  void registerProducer(double weight, std::unique_ptr<Producer> p);
  // unregister the producer indexed by [index] in [_producers]
  void unregisterProducer(int index);
};
