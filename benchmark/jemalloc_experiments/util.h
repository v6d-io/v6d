#include <cstdint>

namespace util {

inline void compilerBarrier() {
  asm volatile("" : : : "memory");
}

inline std::uint64_t rdtsc() noexcept {
  std::uint32_t lo, hi;
  asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)lo) | ((uint64_t)hi << 32);
}

// begin and end must be a multiple of 64.
inline void flushCache(void* beginv, std::size_t size) {
  char* begin = static_cast<char*>(beginv);
  char* end = begin + size;

  for (char* ptr = begin; ptr != end; ptr += 64) {
    __builtin_ia32_clflush(static_cast<void*>(ptr));
  }
}

// Returns time to execute func, in cycles.
template <typename Func>
std::uint64_t runTimed(Func func) {
  std::uint64_t begin = rdtsc();
  compilerBarrier();
  func();
  compilerBarrier();
  std::uint64_t end = rdtsc();
  return end - begin;
}

}
