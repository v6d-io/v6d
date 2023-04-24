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

int grin_get_int32(void* value) {
    return *(static_cast<int*>(value));
}

unsigned int grin_get_uint32(void* value) {
    return *(static_cast<unsigned int*>(value));
}

long int grin_get_int64(void* value) {
    return *(static_cast<long int*>(value));
}

unsigned long int grin_get_uint64(void* value) {
    return *(static_cast<unsigned long int*>(value));
}

float grin_get_float(void* value) {
    return *(static_cast<float*>(value));
}

double grin_get_double(void* value) {
    return *(static_cast<double*>(value));
}

char* grin_get_string(void* value) {
    return static_cast<char*>(value);
}