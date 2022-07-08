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

#ifndef SRC_SERVER_MEMORY_SPILLABLEPAYLOAD_H_
#define SRC_SERVER_MEMORY_SPILLABLEPAYLOAD_H_

#include <cstddef>
#include <cstdint>
#include "common/memory/payload.h"
#include "common/util/uuid.h"
#include "server/memory/memory.h"

namespace vineyard {
    struct SpillablePayload : public Payload{
        SpillablePayload() = default;
        SpillablePayload(ObjectID object_id, int64_t size, uint8_t* ptr, int fd, int64_t msize, ptrdiff_t offset):Payload(object_id, size, ptr, fd, msize, offset), is_spilled(0) {}

        bool operator==(const SpillablePayload& other) const{
            return ((object_id == other.object_id) && (store_fd == other.store_fd) &&
            (data_offset == other.data_offset) &&
            (data_size == other.data_size));
        }

        bool IsSpilled() override { return is_spilled; }

        Status Spill() override;

        Status ReloadFromSpill(std::shared_ptr<BulkStore> bulk_store_ptr) override;

        bool is_spilled;
    };
}


#endif