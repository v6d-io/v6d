#include "Allocation.h"

#include "jemalloc/include/jemalloc/jemalloc.h"

bool Allocation::operator<(const Allocation &that) const {
  return this->freeAfterAbsolute < that.freeAfterAbsolute;
}

bool Allocation::operator>(const Allocation &that) const {
  return this->freeAfterAbsolute > that.freeAfterAbsolute;
}

bool Allocation::isEmpty() const { return this->toFree_.size() == 0; }

Allocation::Allocation(std::vector<void *> toFree, int freeAfterArg)
    : toFree_(toFree), freeAfterRelative(freeAfterArg), freeAfterAbsolute(0) {}

void Allocation::clear() const {
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


