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

#ifndef MODULES_RDMA_UTIL_H_
#define MODULES_RDMA_UTIL_H_

#include <rdma/fabric.h>
#include <string>

#include "common/util/logging.h"
#include "common/util/status.h"

namespace vineyard {

#define ALIGNMENT 64

#define CHECK_ERROR(condition, message) \
  if (condition) {                     \
    return Status::Invalid(message);     \
  }

#define POST(post_fn, op_str, ...)		\
	do {									\
		int ret, rc;							\
		while (1) { \		
      LOG(INFO) << "call post fn\n";					\
			ret = post_fn(__VA_ARGS__);				\
			LOG(INFO) << "call post fn end\n";								\
			if (!ret) {						\
				return Status::OK();						\
			}							\
										\
			if (ret != -FI_EAGAIN) {				\
				LOG(INFO) << op_str << " " << ret;			\
        std::string msg = "Failed to post " + std::string(op_str);	\
				return Status::Invalid(msg);	\
			}							\
			usleep(1000);						\
		}								\
	} while (0)

static inline size_t GetAlignedSize(size_t size, size_t alignment)
{
	return ((size % alignment) == 0) ?
		size : ((size / alignment) + 1) * alignment;
}

struct VineyardBufferContext {
	fid_ep *ep;
	uint64_t rkey;
};

struct VineyardMsg {
	int type;
}

#define VINEYARD_FIVERSION FI_VERSION(1,20)

}

#endif
