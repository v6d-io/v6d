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

The implementation of `get_rss`, `get_shared_rss` and `get_peek_rss`
are referred from https://stackoverflow.com/a/14927379/5080177, which
has the following license header originally:

  * Author:  David Robert Nadeau
  * Site:    http://NadeauSoftware.com/
  * License: Creative Commons Attribution 3.0 Unported License
  *          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#ifndef SRC_COMMON_UTIL_ENV_H_
#define SRC_COMMON_UTIL_ENV_H_

#include <sys/param.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__unix__) || defined(__unix) || defined(unix) || \
    (defined(__APPLE__) && defined(__MACH__))
#include <sys/resource.h>
#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#elif defined(__linux__) || defined(__linux) || defined(linux) || \
    defined(__gnu_linux__)
#include <stdio.h>
#endif
#endif

#include <string>

namespace vineyard {

inline std::string read_env(const char* name) {
  if (const char* envp = std::getenv(name)) {
    return std::string(envp);
  }
  return std::string{};
}

inline std::string read_env(std::string const& name) {
  return read_env(name.c_str());
}

inline std::string get_hostname() {
  if (const char* envp = std::getenv("MY_HOST_NAME")) {
    return std::string(envp);
  } else {
    char hostname_value[MAXHOSTNAMELEN];
    gethostname(&hostname_value[0], MAXHOSTNAMELEN);
    return std::string(hostname_value);
  }
}

inline std::string get_nodename() {
  if (const char* envp = std::getenv("MY_NODE_NAME")) {
    return std::string(envp);
  } else {
    return get_hostname();
  }
}

inline int get_pid() { return static_cast<int>(getpid()); }

/**
 * @brief Returns the current resident set size (physical memory use) measured
 * in bytes.
 *
 * c.f.: https://stackoverflow.com/a/14927379/5080177
 */
inline size_t get_rss() {
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
inline size_t get_shared_rss() {
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
inline size_t get_peek_rss() {
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

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_ENV_H_
