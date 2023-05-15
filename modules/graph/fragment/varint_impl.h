/** Copyright 2020-2023 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef MODULES_GRAPH_FRAGMENT_VARINT_IMPL_H_
#define MODULES_GRAPH_FRAGMENT_VARINT_IMPL_H_
#include <bits/types.h>
#include <vector>

namespace vineyard {

#define UPPER_OF_RANGE_1 185
#define UPPER_OF_RANGE_2 249

inline uint64_t unaligned_load_u64(const uint8_t* p) {
  uint64_t x;
  std::memcpy(&x, p, 8);
  return x;
}

/* header:
|----------------------------------|
| pre_size(4 bits) | size (4 bits) |
|----------------------------------|
*/

// static inline uint8_t construct_header(uint8_t pre_size, uint8_t size) {
//   return (pre_size << 4) | (size & 0x0F);
// }

// static inline unsigned int get_varint_pre_size(const uint8_t header) {
//   return (unsigned int) (header >> 4);
// }

// static inline unsigned int get_varint_size(const uint8_t header) {
//   return (unsigned int) (header & 0x0F);
// }

// inline const uint8_t* get_pointer(const uint8_t* start, const size_t& index)
// {
//   for (size_t i = 0; i < index; i++) {
//     start += (get_varint_size(*start) + 1);
//   }
//   return start;
// }

template <typename T>
void varint_encode(T input, std::vector<uint8_t>& output) {
  if (input < UPPER_OF_RANGE_1) {
    output.push_back(static_cast<uint8_t>(input));
  } else if (input <= UPPER_OF_RANGE_1 + 255 +
                          256 * (UPPER_OF_RANGE_2 - 1 - UPPER_OF_RANGE_1)) {
    input -= UPPER_OF_RANGE_1;
    output.push_back(UPPER_OF_RANGE_1 + (input >> 8));
    output.push_back(input & 0xff);
  } else {
    unsigned bits = 64 - __builtin_clzll(input);
    unsigned bytes = (bits + 7) / 8;
    output.push_back(UPPER_OF_RANGE_2 + (bytes - 2));
    for (unsigned n = 0; n < bytes; n++) {
      output.push_back(input & 0xff);
      input >>= 8;
    }
  }
}

template <typename T>
size_t varint_decode(const uint8_t* input, T& output) {
  const uint8_t* origin_input = input;
  uint8_t b0 = *input++;
  if (LIKELY(b0 < UPPER_OF_RANGE_1)) {
    output = b0;
  } else if (b0 < UPPER_OF_RANGE_2) {
    uint8_t b1 = *input++;
    output = UPPER_OF_RANGE_1 + b1 + ((b0 - UPPER_OF_RANGE_1) << 8);
  } else {
    size_t sh = b0 - UPPER_OF_RANGE_2;
    output = unaligned_load_u64(input) & ((uint64_t(1) << 8 * sh << 16) - 1);
    input += 2 + sh;
  }
  return static_cast<size_t>(input - origin_input);
}

}  // namespace vineyard
#endif  // MODULES_GRAPH_FRAGMENT_VARINT_IMPL_H_