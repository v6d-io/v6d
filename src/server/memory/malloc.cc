/**
 * NOLINT(legal/copyright)
 *
 * The file src/server/memory/malloc.cc is referred and derived from project
 * apache-arrow,
 *
 *    https://github.com/apache/arrow/blob/master/cpp/src/plasma/malloc.cc
 *
 * which has the following license:
 *
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
 */

#include "server/memory/malloc.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <stddef.h>
#include <string>
#include <vector>

#include "gflags/gflags.h"

#include "common/util/logging.h"  // IWYU pragma: keep

namespace vineyard {

namespace memory {

/// Gap between two consecutive mmap regions allocated by fake_mmap.
/// This ensures that the segments of memory returned by
/// fake_mmap are never contiguous and dlmalloc does not coalesce it
/// (in the client we cannot guarantee that these mmaps are contiguous).
constexpr int64_t kMmapRegionsGap = sizeof(size_t);

// Fine-grained control for whether we need pre-populate the shared memory.
//
// Usually it causes a long wait time at the start up, but it could improved
// the performance of visiting shared memory.
//
// In cases that the startup time doesn't much matter, e.g., in kubernetes
// environment, pre-populate will archive a win.
DEFINE_bool(reserve_memory, false,
            "Reserving enough physical memory pages for vineyardd");

std::unordered_map<void*, MmapRecord> mmap_records;

static void* pointer_advance(void* p, ptrdiff_t n) {
  return (unsigned char*) p + n;
}

static void* pointer_retreat(void* p, ptrdiff_t n) {
  return (unsigned char*) p - n;
}

static ptrdiff_t pointer_distance(void const* pfrom, void const* pto) {
  return (unsigned char const*) pto - (unsigned char const*) pfrom;
}

namespace detail {

#ifdef __linux__
struct memfd_create_compat {
  memfd_create_compat() {
    handle_ = dlopen(NULL, RTLD_NOW);
    if (handle_) {
      memfd_create_fn =
          (int (*)(const char*, unsigned int)) dlsym(handle_, "memfd_create");
    }
    if (memfd_create_fn) {
      LOG(INFO) << "Use memfd_create(2) for shared memory";
    } else {
      LOG(INFO) << "Use mkstemp(3) for shared memory";
    }
  }

  inline int operator()() {
    if (memfd_create_fn) {
      std::string file_template = "vineyard-bulk-XXXXXX";
      std::vector<char> file_name(file_template.begin(), file_template.end());
      file_name.push_back('\0');
      int fd = memfd_create_fn(&file_name[0], 0);
      if (fd < 0 && errno == ENOSYS) {
        LOG(WARNING) << "Looks like that we are working inside WSL 1 ("
                     << strerror(errno)
                     << "), fallback to mkstemp(3) for shared memory";
        // disable memfd_create, and fall through
        memfd_create_fn = nullptr;
      } else {
        return fd;  // propogate the error
      }
    }
    std::string file_template = "/dev/shm/vineyard-bulk-XXXXXX";
    std::vector<char> file_name(file_template.begin(), file_template.end());
    file_name.push_back('\0');
    int fd = mkstemp(&file_name[0]);
    if (fd < 0) {
      return fd;
    }
    if (unlink(&file_name[0]) != 0) {
      LOG(ERROR) << "failed to unlink file " << &file_name[0];
      close(fd);
      return -1;
    }
    return fd;
  }

  ~memfd_create_compat() {
    if (handle_) {
      dlclose(handle_);
    }
  }

 private:
  void* handle_ = nullptr;
  int (*memfd_create_fn)(const char* name, unsigned int flags) = nullptr;
};
#endif

}  // namespace detail

// Create a buffer. This is creating a temporary file and then
// immediately unlinking it so we do not leave traces in the system.
int create_buffer(int64_t size, bool memory) {
  int fd = -1;
#ifdef _WIN32
  if (!CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                         (DWORD)((uint64_t) size >> (CHAR_BIT * sizeof(DWORD))),
                         (DWORD)(uint64_t) size, NULL)) {
    fd = -1;
  }

#else             // _WIN32

  /**
   * Notes [memfd_create vs. mkstemp]
   *
   * Use memfd_create(2) on Linux if available. Note that with `memfd_create`,
   * the shared memory usage is accounted to the process that owns the file
   * descriptor, i.e., to the process that creates the blob (which triggers the
   * **anonymous** physical memory allocation), which is a bit counterintuitive
   * (as the shared memory usage cannot be observed on the vineyardd process),
   * but a natural fit for the use cases of vineyard, including on Kubernetes
   * where we subject the memory limitation to the pod level and want to limits
   * the memory usage of the compute process.
   *
   * The `memfd_create` syscall resolves the SIGBUS errors as well.
   *
   * [1]: https://dvdhrm.wordpress.com/2014/06/10/memfd_create2/
   */

  // directory where to create the memory-backed file:
  // on macos or disk-backed on linux
  std::string file_template = "/tmp/vineyard-bulk-XXXXXX";

  std::vector<char> file_name(file_template.begin(), file_template.end());
  file_name.push_back('\0');
#ifdef __linux__  // see also: Notes [memfd_create vs. mkstemp]
  if (memory) {
    static detail::memfd_create_compat memfd_create_compat;
    fd = memfd_create_compat();
  } else {
    fd = mkstemp(&file_name[0]);
  }
#else
  fd = mkstemp(&file_name[0]);
#endif            // __linux__

  if (fd < 0) {
    LOG(ERROR) << "create_buffer failed to open file " << &file_name[0];
    return -1;
  }

  // Immediately unlink the file so we do not leave traces in the system.
#ifdef __linux__  // see also: Notes [memfd_create vs. mkstemp]
  if (!memory && unlink(&file_name[0]) != 0) {
#else
  if (unlink(&file_name[0]) != 0) {
#endif
    LOG(ERROR) << "failed to unlink file " << &file_name[0];
    return -1;
  }

  if (true) {
    // Increase the size of the file to the desired size. This seems not to be
    // needed for files that are backed by the huge page fs, see also
    // http://www.mail-archive.com/kvm-devel@lists.sourceforge.net/msg14737.html
    if (ftruncate(fd, (off_t) size) != 0) {
      LOG(ERROR) << "failed to ftruncate file " << &file_name[0];
      return -1;
    }
  }
#endif  // _WIN32
  // return the created fd
  return fd;
}

int create_buffer(int64_t size, std::string const& path) {
  int fd = open(path.c_str(), O_RDWR | O_CREAT | O_NONBLOCK, 0666);
  if (fd < 0) {
    LOG(ERROR) << "failed to open file '" << path << "', " << strerror(errno);
    return fd;
  }
  off_t fsize = lseek(fd, 0, SEEK_END);
  if (fsize < size) {
    if (ftruncate(fd, (off_t) size) != 0) {
      LOG(ERROR) << "failed to ftruncate file '" << path << "', "
                 << strerror(errno);
      return -1;
    }
  }
  return fd;
}

void* mmap_buffer(int64_t size, bool* is_committed, bool* is_zero) {
  // Add kMmapRegionsGap so that the returned pointer is deliberately not
  // page-aligned. This ensures that the segments of memory returned by
  // fake_mmap are never contiguous.
  size += kMmapRegionsGap;

  int fd = create_buffer(size);
  return mmap_buffer(fd, size, true, is_committed, is_zero);
}

void* mmap_buffer(int fd, int64_t size, bool gap, bool* is_committed,
                  bool* is_zero) {
  if (fd < 0) {
    LOG(ERROR) << "failed to create buffer during mmap: " << strerror(errno);
    return nullptr;
  }
  // MAP_POPULATE can be used to pre-populate the page tables for this memory
  // region
  // which avoids work when accessing the pages later. However it causes long
  // pauses
  // when mmapping the files. Only supported on Linux.

  int mmap_flag = MAP_SHARED;
  if (FLAGS_reserve_memory) {
#ifdef __linux__
    mmap_flag |= MAP_POPULATE;
    *is_committed = true;
    LOG(INFO) << "mmap with MAP_POPULATE, reserving memory ...";
#endif
  }

  void* pointer = mmap(NULL, size, PROT_READ | PROT_WRITE, mmap_flag, fd, 0);
  if (pointer == MAP_FAILED) {
    LOG(ERROR) << "mmap failed with error: " << strerror(errno);
    return pointer;
  }

  MmapRecord& record = mmap_records[pointer];
  record.fd = fd;
  record.size = size;
  record.kind = MmapRecord::Kind::kMalloc;

  if (gap) {
    // We lie to dlmalloc/mimalloc about where mapped memory actually lives.
    pointer = pointer_advance(pointer, kMmapRegionsGap);
  }
  return pointer;
}

int munmap_buffer(void* addr, int64_t size) {
  addr = pointer_retreat(addr, kMmapRegionsGap);
  size += kMmapRegionsGap;

  auto entry = mmap_records.find(addr);

  if (entry == mmap_records.end() || entry->second.size != size) {
    // Reject requests to munmap that don't directly match previous
    // calls to mmap, to prevent dlmalloc from trimming.
    return -1;
  }

  int r = munmap(addr, size);
  if (r == 0) {
    close(entry->second.fd);
  }

  mmap_records.erase(entry);
  return r;
}

void GetMallocMapInfo(void* addr, int* fd, int64_t* map_size,
                      ptrdiff_t* offset) {
  // About the efficiency: the records size usually small, thus linear search
  // is enough.
  for (const auto& entry : mmap_records) {
    if (addr >= entry.first &&
        addr < pointer_advance(entry.first, entry.second.size)) {
      *fd = entry.second.fd;
      *map_size = entry.second.size;
      *offset = pointer_distance(entry.first, addr);
      return;
    }
  }
  LOG(INFO) << "fd not found for " << addr
            << ", mmap records size = " << mmap_records.size();
  *fd = -1;
  *map_size = 0;
  *offset = 0;
}

}  // namespace memory

}  // namespace vineyard
