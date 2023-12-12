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

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#include <cinttypes>  // IWYU pragma: keep
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "common/util/env.h"

#if defined(__unix__) || defined(__unix) || defined(unix) || \
    (defined(__APPLE__) && defined(__MACH__))
#include <sys/resource.h>
#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#elif defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
#include <fcntl.h>
#include <malloc.h>
#include <sys/statvfs.h>
#endif
#endif

#ifdef __linux__
#ifndef SHMMAX_SYS_FILE
#define SHMMAX_SYS_FILE "/proc/sys/kernel/shmmax"
#endif
#else
#include <sys/sysctl.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>

namespace vineyard {

namespace detail {

#if 0
#if defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
static void parse_proc_smaps(int64_t& private_rss, int64_t& shared_rss) {
  FILE* fp = nullptr;
  if ((fp = fopen("/proc/self/smaps", "r")) == nullptr) {
    return;
  }
  char line[1024];
  while (!feof(fp)) {
    if (fgets(line, 1024, fp) == nullptr) {
      break;
    }
    if (strncmp(line, "Pss:", 4) == 0) {
      private_rss += atoll(line + 4) * 1024;
    } else if (strncmp(line, "Rss:", 4) == 0) {
      shared_rss += atoll(line + 4) * 1024;
    }
  }
}
#endif
#endif

}  // namespace detail

void create_dirs(const char* path) {
  if (path == nullptr) {
    return;
  }
  size_t length = strlen(path);
  if (length == 0) {
    return;
  }
  char* temp = static_cast<char*>(malloc(length + 1));
  memset(temp, 0x00, length + 1);
  for (size_t i = 0; i < length; i++) {
    temp[i] = path[i];
    if (temp[i] == '/') {
      if (access(temp, 0) != 0) {
        mkdir(temp, 0755);
      }
    }
  }
  if (access(temp, 0) != 0) {
    mkdir(temp, 0755);
  }
  free(temp);
}

/**
 * @brief Returns the current resident set size (physical memory use) measured
 * in bytes.
 *
 * c.f.: https://stackoverflow.com/a/14927379/5080177
 */
size_t get_rss(bool include_shared_memory) {
  // why "trim_rss" first?
  //
  //  - for more accurate statistics
  //  - as a hint for allocator to release pages in places where `get_rss()`
  //    is called (where memory information is in cencern) in programs.
  trim_rss();

#if defined(__APPLE__) && defined(__MACH__)
  /* OSX ------------------------------------------------------ */
  struct mach_task_basic_info info;
  mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &info,
                &infoCount) != KERN_SUCCESS)
    return (size_t) 0L; /* Can't access? */
  return (size_t) info.resident_size;
#elif defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
  // /* Linux ---------------------------------------------------- */
  int64_t rss = 0L, shared_rss = 0L;
  FILE* fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t) 0L; /* Can't open? */
  //
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return (size_t) 0L; /* Can't read? */
  }
  // read the second number
  if (fscanf(fp, "%ld", &shared_rss) != 1) {
    fclose(fp);
    return (size_t) 0L; /* Can't read? */
  }
  fclose(fp);
  if (include_shared_memory) {
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);
  } else {
    return (size_t)(rss - shared_rss) * (size_t) sysconf(_SC_PAGESIZE);
  }
#else
  /* Unknown OS ----------------------------------------------- */
  return 0;
#endif
}

std::string get_rss_pretty(const bool include_shared_memory) {
  return prettyprint_memory_size(get_rss(include_shared_memory));
}

void trim_rss() {
#if defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
  malloc_trim(1024 * 1024 /* 1MB */);
#endif
}

/**
 * @brief Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes.
 *
 * c.f.: https://stackoverflow.com/a/14927379/5080177
 */
size_t get_shared_rss() {
#if defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
  /* Linux ---------------------------------------------------- */
  int64_t shared_rss = 0L;
  FILE* fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t) 0L; /* Can't open? */
  if (fscanf(fp, "%*s%*s%ld", &shared_rss) != 1) {
    fclose(fp);
    return (size_t) 0L; /* Can't read? */
  }
  fclose(fp);
  return (size_t) shared_rss * (size_t) sysconf(_SC_PAGESIZE);
#else
  /* Unknown OS ----------------------------------------------- */
  return 0;
#endif
}

std::string get_shared_rss_pretty() {
  return prettyprint_memory_size(get_shared_rss());
}

/**
 * @brief Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes.
 *
 * c.f.: https://stackoverflow.com/a/14927379/5080177
 */
size_t get_peak_rss() {
#if defined(__unix__) || defined(__unix) || defined(unix) || \
    (defined(__APPLE__) && defined(__MACH__))
  /* BSD, Linux, and OSX -------------------------------------- */
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
  return (size_t) rusage.ru_maxrss;
#else
  return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
  /* Unknown OS ----------------------------------------------- */
  return 0;
#endif
}

std::string get_peak_rss_pretty() {
  return prettyprint_memory_size(get_peak_rss());
}

/**
 * @brief Return the size limitation of available shared memory.
 *
 * c.f.: https://github.com/nicobou/shmdata/blob/develop/shmdata/sysv-shm.cpp
 */
int64_t get_maximum_shared_memory() {
  int64_t shmmax = 0;
#ifdef __linux__
  int shm_fd = open("/dev/shm", O_RDONLY);
  struct statvfs shm_vfs_stats;
  fstatvfs(shm_fd, &shm_vfs_stats);
  // The value shm_vfs_stats.f_bsize is the block size, and the value
  // shm_vfs_stats.f_bavail is the number of available blocks.
  shmmax = shm_vfs_stats.f_bsize * shm_vfs_stats.f_bavail;
  close(shm_fd);
#else
  size_t len = sizeof(shmmax);
  if (sysctlbyname("kern.sysv.shmmax", &shmmax, &len, NULL, 0) == -1) {
    std::clog << "[warn] Failed to read shmmax from 'kern.sysv.shmmax'!"
              << std::endl;
  }
#endif
  return shmmax;
}

/**
 * @brief Return the memory size in human readable way.
 */
std::string prettyprint_memory_size(size_t nbytes) {
  if (nbytes >= (1LL << 40)) {
    return std::to_string(nbytes * 1.0 / (1LL << 40)) + " TB";
  } else if (nbytes >= (1LL << 30)) {
    return std::to_string(nbytes * 1.0 / (1LL << 30)) + " GB";
  } else if (nbytes >= (1LL << 20)) {
    return std::to_string(nbytes * 1.0 / (1LL << 20)) + " MB";
  } else if (nbytes >= (1LL << 10)) {
    return std::to_string(nbytes * 1.0 / (1LL << 10)) + " KB";
  } else {
    return std::to_string(nbytes) + " B";
  }
}

/**
 * @brief Parse human-readable size. Note that any extra character that follows
 * a valid sequence will be ignored.
 */
int64_t parse_memory_size(std::string const& nbytes) {
  const char *start = nbytes.c_str(), *end = nbytes.c_str() + nbytes.size();
  char* parsed_end = nullptr;
  double parse_size = std::strtod(start, &parsed_end);
  if (end == parsed_end || *parsed_end == '\0') {
    return static_cast<int64_t>(parse_size);
  }
  switch (*parsed_end) {
  case 'k':
  case 'K':
    return static_cast<int64_t>(parse_size * (1LL << 10));
  case 'm':
  case 'M':
    return static_cast<int64_t>(parse_size * (1LL << 20));
  case 'g':
  case 'G':
    return static_cast<int64_t>(parse_size * (1LL << 30));
  case 't':
  case 'T':
    return static_cast<int64_t>(parse_size * (1LL << 40));
  case 'P':
  case 'p':
    return static_cast<int64_t>(parse_size * (1LL << 50));
  case 'e':
  case 'E':
    return static_cast<int64_t>(parse_size * (1LL << 60));
  default:
    return static_cast<int64_t>(parse_size);
  }
}

int64_t read_physical_memory_limit() {
  // see also: https://stackoverflow.com/a/71392704/5080177
  constexpr const int64_t unlimited = 0x7f00000000000000;

  int64_t limit_in_bytes = 0;
  FILE* fp = nullptr;
  if ((fp = fopen("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r")) !=
      nullptr) {
#if defined(__APPLE__) && defined(__MACH__)
    if (fscanf(fp, "%lld", &limit_in_bytes) != 1 ||
#else
    if (fscanf(fp, "%ld", &limit_in_bytes) != 1 ||
#endif
        limit_in_bytes >= unlimited) {
      limit_in_bytes = 0;
    }
    fclose(fp);
  }
  if (limit_in_bytes != 0) {
    return limit_in_bytes;
  }

  if ((fp = fopen("/sys/fs/cgroup/memory.max", "r")) != nullptr) {
#if defined(__APPLE__) && defined(__MACH__)
    if (fscanf(fp, "%lld", &limit_in_bytes) != 1 ||
#else
    if (fscanf(fp, "%ld", &limit_in_bytes) != 1 ||
#endif
        limit_in_bytes >= unlimited) {
      limit_in_bytes = 0;
    }
    fclose(fp);
  }
  if (limit_in_bytes != 0) {
    return limit_in_bytes;
  }

#ifdef __linux__
  int64_t physical_pages = sysconf(_SC_PHYS_PAGES);
  if (physical_pages == -1) {
    return -1;
  }
  int64_t page_size = sysconf(_SC_PAGE_SIZE);
  if (page_size == -1) {
    return -1;
  }
  limit_in_bytes = physical_pages * page_size;
#else
  size_t len = sizeof(limit_in_bytes);
  if (sysctlbyname("hw.memsize", &limit_in_bytes, &len, NULL, 0) == -1) {
    return -1;
  }
#endif

  return limit_in_bytes;
}

}  // namespace vineyard
