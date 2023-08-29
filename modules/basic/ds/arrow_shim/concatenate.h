/**
 * NOLINT(legal/copyright)
 *
 * The file modules/basic/ds/arrow_shim/concatenate.h is referred and derived
 * from project apache-arrow,
 *
 * https://github.com/apache/arrow/blob/master/cpp/src/arrow/array/concatenate.h
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

#ifndef MODULES_BASIC_DS_ARROW_SHIM_CONCATENATE_H_
#define MODULES_BASIC_DS_ARROW_SHIM_CONCATENATE_H_

#include <memory>

#include "arrow/type_fwd.h"
#include "arrow/util/macros.h"

namespace vineyard {

namespace arrow_shim {

using namespace arrow;  // NOLINT(build/namespaces)

Result<std::shared_ptr<Array>> Concatenate(
    ArrayVector&& arrays, MemoryPool* pool = default_memory_pool());

Result<std::shared_ptr<Buffer>> ConcatenateBuffers(BufferVector&& buffers,
                                                   MemoryPool* pool = NULLPTR);

}  // namespace arrow_shim

}  // namespace vineyard

#endif  // MODULES_BASIC_DS_ARROW_SHIM_CONCATENATE_H_
