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

#ifndef SRC_COMMON_UTIL_SIDECAR_H_
#define SRC_COMMON_UTIL_SIDECAR_H_

#include <string>

#include "common/util/json.h"
#include "common/util/status.h"

namespace vineyard {

#ifndef GET_BLOB_RECV_MEM_SIZE
#define GET_BLOB_RECV_MEM_SIZE (4096)
#endif  // GET_BLOB_RECV_MEM_SIZE

#ifndef ERROR_MSG_LENGTH
#define ERROR_MSG_LENGTH (256)
#endif  // ERROR_MSG_LENGTH

#ifndef MAX_METAS_FROM_NAME
#define MAX_METAS_FROM_NAME (1000)
#endif  // MAX_METAS_FROM_NAME

struct ClientAttributes {
  std::string req_name;

  static ClientAttributes Default() { return ClientAttributes{.req_name = ""}; }

  ClientAttributes SetReqName(std::string name) {
    this->req_name = name;
    return *this;
  }

  json ToJson() {
    json j;
    j["req_name"] = req_name;
    return j;
  }

  static ClientAttributes FromJson(const json& j) {
    ClientAttributes attr;
    if (j.contains("req_name")) {
      attr.req_name = j["req_name"].get<std::string>();
    }
    return attr;
  }

  void ToBinary(void* data, size_t& size) {
    std::string str = json_to_string(ToJson());
    memcpy(data, str.c_str(), str.size());
    size = str.length();
  }

  void FromBinary(void* data, size_t size, ClientAttributes& attr) {
    std::string str(static_cast<char*>(data), size);
    attr = FromJson(json_from_buf(str.c_str(), str.length()));
  }

  std::string ToJsonString() { return json_to_string(ToJson()); }

  static ClientAttributes FromJsonString(std::string s) {
    return FromJson(json_from_buf(s.c_str(), s.length()));
  }
};

Status CreateMmapMemory(int& fd, size_t size, void*& base);

Status CreateMmapMemory(std::string file_name, int& fd, size_t size,
                        void*& base);

Status WriteErrorMsg(Status status, void* base, size_t size);

Status ReleaseMmapMemory(int fd, void* base, size_t size);

Status CheckBlobReceived(void* base, size_t size, int index, bool& finished);

Status SetBlobReceived(void* base, int index);

}  // namespace vineyard

#endif  // SRC_COMMON_UTIL_SIDECAR_H_
