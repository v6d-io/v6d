#include "Distribution.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <gflags/gflags.h>

DEFINE_int64(max_size_class, 10 * k1KB, "max size class to allocate");

SizeClass parseSizeClass(const std::string &ln) {
  std::istringstream strStream(ln);
  size_t sizeClass;
  double freq;
  if (!(strStream >> sizeClass >> freq)) {
    std::cout << "File format invalid. Failed to following line:\n\e[0;31m"
              << ln << "\e[0m" << std::endl;
    exit(1);
  }
  if (freq > 1.0) {
    std::cout << "Warning: this looks off; frequency greater than 1.0"
              << std::endl;
    freq = 1.0;
  }
  return {sizeClass, freq};
}

Distribution parseDistribution(const char *fileName) {
  std::string line;
  std::ifstream f(fileName);

  if (!f) {
    std::cout << "Specified file '" << fileName << "' not found." << std::endl;
    exit(1);
  }

  Distribution d;

  while (std::getline(f, line)) {
    SizeClass sz = parseSizeClass(line);
    if (sz.size <= FLAGS_max_size_class) {
      d.push_back(sz);
    }
  }

  std::sort(begin(d), end(d));
  return d;
}
