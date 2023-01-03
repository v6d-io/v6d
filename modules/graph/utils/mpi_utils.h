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

#ifndef MODULES_GRAPH_UTILS_MPI_UTILS_H_
#define MODULES_GRAPH_UTILS_MPI_UTILS_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"

#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "grape/worker/comm_spec.h"

#include "common/util/arrow.h"

namespace vineyard {

template <class T>
static void GlobalAllGatherv(T& object, std::vector<T>& to_exchange,
                             const grape::CommSpec& comm_spec) {
  grape::InArchive ia;
  ia << object;
  size_t send_count = ia.GetSize();
  auto worker_num = comm_spec.worker_num();
  auto* recv_counts = static_cast<int*>(malloc(sizeof(int) * worker_num));
  MPI_Allgather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT,
                comm_spec.comm());

  size_t recv_buf_len = 0;
  for (int i = 0; i < worker_num; i++) {
    recv_buf_len += recv_counts[i];
  }

  grape::OutArchive oa(recv_buf_len);

  int* displs = static_cast<int*>(malloc(sizeof(size_t) * worker_num));
  displs[0] = 0;
  for (int i = 1; i < worker_num; i++) {
    displs[i] = displs[i - 1] + recv_counts[i - 1];
  }
  MPI_Allgatherv(ia.GetBuffer(), send_count, MPI_CHAR, oa.GetBuffer(),
                 recv_counts, displs, MPI_CHAR, comm_spec.comm());
  to_exchange.resize(worker_num);
  for (int i = 0; i < worker_num; i++) {
    oa >> to_exchange[i];
  }
  free(recv_counts);
  free(displs);
}

template <class T>
static void _GatherR(std::vector<T>& object, MPI_Comm comm) {
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  size_t send_count = 0;
  size_t* recv_counts = static_cast<size_t*>(malloc(sizeof(size_t) * size));
  MPI_Gather(&send_count, sizeof(size_t), MPI_CHAR, recv_counts, sizeof(size_t),
             MPI_CHAR, rank, comm);

  int* recv_counts_int = static_cast<int*>(malloc(sizeof(int) * size));
  int* recv_displs = static_cast<int*>(malloc(sizeof(int) * size));
  size_t recv_size = 0;
  for (int i = 0; i < size; ++i) {
    recv_counts_int[i] = recv_counts[i];
    recv_displs[i] = recv_size;
    recv_size += recv_counts[i];
  }

  CHECK_LT(recv_size, std::numeric_limits<int>::max());

  grape::OutArchive oa(recv_size);
  MPI_Gatherv(NULL, 0, MPI_CHAR, oa.GetBuffer(), recv_counts_int, recv_displs,
              MPI_CHAR, rank, comm);

  for (int i = 0; i < size; i++) {
    if (i == rank) {
      continue;
    }
    oa >> object[i];
  }
  free(recv_counts);
  free(recv_counts_int);
  free(recv_displs);
}

template <class T>
static void _GatherL(const T& object, int root, MPI_Comm comm) {
  grape::InArchive ia;
  ia << object;
  size_t send_count = ia.GetSize();
  MPI_Gather(&send_count, sizeof(size_t), MPI_CHAR, NULL, sizeof(size_t),
             MPI_CHAR, root, comm);
  MPI_Gatherv(ia.GetBuffer(), send_count, MPI_CHAR, NULL, NULL, NULL, MPI_CHAR,
              root, comm);
}

}  // namespace vineyard

namespace grape {
inline InArchive& operator<<(InArchive& in_archive,
                             std::shared_ptr<arrow::Schema>& schema) {
  if (schema != nullptr) {
    std::shared_ptr<arrow::Buffer> out;
#if defined(ARROW_VERSION) && ARROW_VERSION < 2000000
    CHECK_ARROW_ERROR_AND_ASSIGN(
        out, arrow::ipc::SerializeSchema(*schema, nullptr,
                                         arrow::default_memory_pool()));
#else
    CHECK_ARROW_ERROR_AND_ASSIGN(
        out,
        arrow::ipc::SerializeSchema(*schema, arrow::default_memory_pool()));
#endif
    in_archive.AddBytes(out->data(), out->size());
  }
  return in_archive;
}

inline OutArchive& operator>>(OutArchive& out_archive,
                              std::shared_ptr<arrow::Schema>& schema) {
  if (!out_archive.Empty()) {
    auto buffer = std::make_shared<arrow::Buffer>(
        reinterpret_cast<const uint8_t*>(out_archive.GetBuffer()),
        out_archive.GetSize());
    arrow::io::BufferReader reader(buffer);
    CHECK_ARROW_ERROR_AND_ASSIGN(schema,
                                 arrow::ipc::ReadSchema(&reader, nullptr));
  }
  return out_archive;
}

#if ARROW_VERSION_MAJOR >= 10
inline InArchive& operator<<(InArchive& archive, const std::string_view& str) {
  archive << str.length();
  archive.AddBytes(str.data(), str.length());
  return archive;
}

inline OutArchive& operator>>(OutArchive& archive, std::string_view& str) {
  size_t length;
  archive >> length;
  str = std::string_view(reinterpret_cast<char*>(archive.GetBytes(length)),
                         length);
  return archive;
}
#endif

}  // namespace grape

#endif  // MODULES_GRAPH_UTILS_MPI_UTILS_H_
