#pragma once
#include <vector>
#include <utils.h>

namespace vineyard{

#define UPPER_OF_RANGE_1 185
#define UPPER_OF_RANGE_2 249

inline uint64_t unaligned_load_u64(const uint8_t* p)
{
  uint64_t x;
  std::memcpy(&x, p, 8);
  return x;
}

template<typename T>
size_t varint_encode(const T &input, std::vector<uint8_t> &output)
{
  if (x < UPPER_OF_RANGE_1) {
    output.push_back(static_cast<uint8_t>(x));
  } else if (x <= UPPER_OF_RANGE_1 + 255 + 256 * (UPPER_OF_RANGE_2 - 1 - UPPER_OF_RANGE_1)) {
    x -= UPPER_OF_RANGE_1;
    output.push_back(UPPER_OF_RANGE_1 + (x >> 8));
    output.push_back(x & 0xff);
  } else {
    unsigned bits = 64 - __builtin_clzll(x);
    unsigned bytes = (bits + 7) / 8;
    output.push_back(UPPER_OF_RANGE_2 + (bytes - 2));
    for (unsigned n = 0; n < bytes; n++) {
      output.push_back(x & 0xff);
      x >>= 8;
    }
  }
}

template <typename T>
size_t varint_decode(const uint8_t *input, std::vector<T> &output, int64_t start, int64_t end)
{
  LOG(INFO) << __func__;
  int64_t count = end - start;
  uint8_t *o_input;
  std::vector<T> temp;
  LOG(INFO) << "count:" << count;
  for(int i = 0; i < end; i++) {
    uint8_t b0 = *input++;
    if (LIKELY(b0 < UPPER_OF_RANGE_1)) {
      temp.push_back(b0);
    } else if (b0 < UPPER_OF_RANGE_2) {
      uint8_t b1 = *input++;
      temp.push_back(UPPER_OF_RANGE_1 + b1 + ((b0 - UPPER_OF_RANGE_1) << 8));
    } else {
      size_t sh = b0 - UPPER_OF_RANGE_2;
      temp.push_back(unaligned_load_u64(input) & ((uint64_t(1) << 8 * sh << 16) - 1));
      input += 2 + sh;
    }
  }
  while(start < end) {
    output.push_back(temp[start]);
    start++;
  }
  return input - o_input;
}

template <typename T>
size_t varint_decode(const uint8_t *input, T &output)
{
  size_t ret;
  std::vector<T> out;
  ret = varint_decode(input, out, 0, 1);
  output = out[0];
  return ret; 
}

}