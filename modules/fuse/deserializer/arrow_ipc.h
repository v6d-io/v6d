#ifndef MODULES_FUSE_ADAPTORS_ARROW_IPC_H_
#define MODULES_FUSE_ADAPTORS_ARROW_IPC_H_

#if defined(WITH_ARROW_IPC)

#include <memory>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/key_value_metadata.h"
#include "arrow/util/macros.h"
#include "basic/ds/array.h"
#include "basic/ds/dataframe.h"
#include "client/client.h"
#include "client/ds/blob.h"
#include "client/ds/core_types.h"
#include "client/ds/i_object.h"
#include "common/util/logging.h"
#include "common/util/uuid.h"
#include "basic/ds/array.h"
namespace vineyard {
namespace fuse {

std::shared_ptr<arrow::Buffer> arrow_ipc_view(
    std::shared_ptr<vineyard::DataFrame>& df);

// template <typename T>
std::shared_ptr<arrow::Buffer> arrow_ipc_view(
    std::shared_ptr<vineyard::NumericArray<int64_t>>& arr);
std::shared_ptr<arrow::Buffer> arrow_ipc_view(
    std::shared_ptr<vineyard::NumericArray<bool>>& arr);
}  // namespace fuse
}  // namespace vineyard

#endif

#endif