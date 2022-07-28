#ifndef MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_
#define MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_
#include <stdio.h>

#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/util/env.h"

#include <glog/logging.h>
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
#include "common/util/typename.h"

#include "common/util/uuid.h"

namespace vineyard {
namespace fuse {
using vineyard_deserializer_nt = std::shared_ptr<arrow::Buffer> (*)(
    const std::shared_ptr<vineyard::Object>&);

template <typename T>
std::shared_ptr<arrow::Buffer> numeric_array_arrow_ipc_view(
    const std::shared_ptr<vineyard::Object>& p);
std::shared_ptr<arrow::Buffer> string_array_arrow_ipc_view(
    const std::shared_ptr<vineyard::Object>& p);
std::shared_ptr<arrow::Buffer> bool_array_arrow_ipc_view(
    const std::shared_ptr<vineyard::Object>& p);
std::shared_ptr<arrow::Buffer> dataframe_arrow_ipc_view(
    const std::shared_ptr<vineyard::Object>& p);
std::unordered_map<std::string, vineyard::fuse::vineyard_deserializer_nt>
arrow_ipc_register_once();
}  // namespace fuse
}  // namespace vineyard
#endif  // MODULES_FUSE_ADAPTORS_ARROW_IPC_DESERIALIZER_REGISTRY_H_
