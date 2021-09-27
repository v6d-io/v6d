#include <vector>

#include "malloc/allocator.h"

// simple wrapper that pairs a group of allocated blocks with a lifetime

#define BENCH_VINEYARD
// #define BENCH_JEMALLOC
// #define BENCH_SYSTEM
// #define BENCH_ARENA

class Allocation {
public:
  // sorts based on [freeAfterAbsolute] field
  bool operator<(const Allocation &that) const;
  bool operator>(const Allocation &that) const;
  // true iff [this->toFree_] is empty
  bool isEmpty() const;
  // free the memory stored within
  void clear() const;

  Allocation() = default;

  /* [freeAfter] is a number of phases, according to the thread that is
   * responsible for this allocation */
  Allocation(std::vector<void *> toFree, int freeAfter);

  // number of phases to live for relative to allocation
  int freeAfterRelative;
  // absolute phase number at which to free, based on a particular threads clock
  int freeAfterAbsolute;

private:
  std::vector<void *> toFree_;
  // absolute time after which this should be freed
};
