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

#ifndef MODULES_MALLOC_MALLOC_WRAPPER_H_
#define MODULES_MALLOC_MALLOC_WRAPPER_H_

#include <stddef.h>

#if !defined(VINEYARD_MALLOC_PREFIX)
#define VINEYARD_MALLOC_PREFIX
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void* vineyard_malloc(size_t size);
void* vineyard_realloc(void* pointer, size_t size);
void* vineyard_calloc(size_t num, size_t size);
void vineyard_free(void* pointer);
void vineyard_freeze(void* pointer);
void vineyard_allocator_finalize(int renew);

void* vineyard_arena_malloc(size_t size);
void vineyard_arena_free(void* ptr);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // MODULES_MALLOC_MALLOC_WRAPPER_H_
