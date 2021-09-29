#pragma once

#include <vector>

#include "SizeConstants.h"

struct SizeClass {
  size_t size;
  double freq;
  bool operator<(const SizeClass &that) const { return this->freq < that.freq; }
};

typedef std::vector<SizeClass> Distribution;

Distribution parseDistribution(const char *fileName);
