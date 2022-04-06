/** Copyright 2020-2021 Alibaba Group Holding Limited.

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
#include <cinttypes>
#endif

#include "common/util/env.h"

#if defined(__unix__) || defined(__unix) || defined(unix) || \
    (defined(__APPLE__) && defined(__MACH__))
#include <sys/resource.h>
#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#elif defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
#include <fcntl.h>
#include <sys/statvfs.h>
#endif
#endif

#ifdef __linux__
#ifndef SHMMAX_SYS_FILE
#define SHMMAX_SYS_FILE "/proc/sys/kernel/shmmax"
#endif
#else
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#include <iostream>

namespace vineyard {

/**
 * @brief Returns the current resident set size (physical memory use) measured
 * in bytes.
 *
 * c.f.: https://stackoverflow.com/a/14927379/5080177
 */
size_t get_rss() {
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
  /* Linux ---------------------------------------------------- */
  int64_t rss = 0L;
  FILE* fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t) 0L; /* Can't open? */
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return (size_t) 0L; /* Can't read? */
  }
  fclose(fp);
  return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);
#else
  /* Unknown OS ----------------------------------------------- */
  return 0;
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

/**
 * @brief Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes.
 *
 * c.f.: https://stackoverflow.com/a/14927379/5080177
 */
size_t get_peek_rss() {
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
  if (-1 == sysctlbyname("kern.sysv.shmmax", &shmmax, &len, NULL, 0)) {
    std::clog << "[warn] Failed to read shmmax from 'kern.sysv.shmmax'!"
              << std::endl;
  }
#endif
  return shmmax;
}

}  // namespace vineyard
