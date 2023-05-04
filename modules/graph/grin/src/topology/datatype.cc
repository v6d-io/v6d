/** Copyright 2020 Alibaba Group Holding Limited.

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

#include "graph/grin/src/predefine.h"
extern "C" {
#include "graph/grin/include/topology/datatype.h"
}

int grin_get_int32(const void* value) {
    return *(static_cast<const int*>(value));
}

unsigned int grin_get_uint32(const void* value) {
    return *(static_cast<const unsigned int*>(value));
}

long long int grin_get_int64(const void* value) {
    return *(static_cast<const long int*>(value));
}

unsigned long long int grin_get_uint64(const void* value) {
    return *(static_cast<const unsigned long int*>(value));
}

float grin_get_float(const void* value) {
    return *(static_cast<const float*>(value));
}

double grin_get_double(const void* value) {
    return *(static_cast<const double*>(value));
}

const char* grin_get_string(const void* value) {
    return static_cast<const char*>(value);
}