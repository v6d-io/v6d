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

#include "server/memory/spillable_payload.h"
#include "client/ds/i_object.h"
#include "common/util/status.h"
#include "common/util/uuid.h"
#include "server/util/file_io.h"

namespace vineyard{

  // spilled file name format: tmp_spill_<object_id>
  // we need to dump 1. object_id 2. data_size
  Status SpillablePayload::Spill(){
    assert(this->is_sealed);
    std::string file_name = "tmp_spill" + std::to_string(object_id);
    util::SpillWriteFile spill_file(file_name);
    // spill object_id + data_size to disk
    // write object_id
    RETURN_ON_ERROR(spill_file.Open());
    RETURN_ON_ERROR(spill_file.Write(reinterpret_cast<const char*>(pointer),
                                    data_size, object_id));
    RETURN_ON_ERROR(spill_file.Sync());
    // TODO: free this pointer
    pointer = nullptr;
    is_spilled = true;
    return Status::OK();
  }

  // reload object_id data_size
  Status SpillablePayload::ReloadFromSpill(std::shared_ptr<BulkStore> bulk_store_ptr){
    assert(is_spilled == true);
    // if(is_spilled == false){
    // return Status::OK();
    // }

    // reload 1. object_id 2. data_size back to memory
    std::string file_name = "tmp_spill" + std::to_string(object_id);
    util::SpillReadFile spill_file(file_name);
    RETURN_ON_ERROR(spill_file.Open());
    {
      char buf[sizeof(object_id)];
      RETURN_ON_ERROR(spill_file.Read(sizeof(object_id), buf));
      if (object_id != util::DecodeFixed64(buf)) {
        return Status::IOError("Opened wrong file: " + file_name);
      }
    }
    {
      char buf[sizeof(data_size)];
      RETURN_ON_ERROR(spill_file.Read(sizeof(uint64_t), buf));
      if(data_size != util::DecodeFixed64(buf)){
        return Status::IOError("Opened wrong file: " + file_name);
      }
      pointer = nullptr;
          // bulk_store_ptr->AllocateMemoryWithSpill(data_size, &store_fd, &map_size, &data_offset);
    }
    if(pointer == nullptr){
      return Status::NotEnoughMemory("Failed to allocate memory of size " + std::to_string(data_size) + " while reload spilling file");
    }
    return Status::OK();
    }
}