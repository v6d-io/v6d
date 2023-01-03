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
#include "io_v6d_core_client_ds_ffi_ObjectMeta.h"

#include <cstring>
#include <string>

#include "vineyard/client/ds/i_object.h"
#include "vineyard/client/ds/object_factory.h"
#include "vineyard/client/ds/object_meta.h"
#include "vineyard/common/util/json.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     io_v6d_core_client_ds_ffi_ObjectMeta
 * Method:    constructNative
 * Signature: (Ljava/lang/String;[J[J)J
 */
JNIEXPORT jlong JNICALL
Java_io_v6d_io_v6d_core_client_ds_ffi_ObjectMeta_constructNative(
    JNIEnv* env, jobject, jstring meta, jlongArray objects, jlongArray pointers,
    jlongArray sizes) {
  jboolean isCopy = JNI_FALSE;

  jsize length = env->GetStringUTFLength(meta);
  const char* chars = env->GetStringUTFChars(meta, &isCopy);
  jsize nobjects = env->GetArrayLength(objects);
  jsize nbuffers = env->GetArrayLength(pointers);
  jsize nsizes = env->GetArrayLength(sizes);
  assert(nobjects == nbuffers);
  assert(nobjects == nsizes);
  auto object_elements = env->GetLongArrayElements(objects, &isCopy);
  auto pointer_elements = env->GetLongArrayElements(pointers, &isCopy);
  auto size_elements = env->GetLongArrayElements(sizes, &isCopy);

  auto metadata = vineyard::ObjectMeta::Unsafe(
      std::string(chars, length), nobjects,
      reinterpret_cast<vineyard::ObjectID*>(object_elements),
      reinterpret_cast<uintptr_t*>(pointer_elements),
      reinterpret_cast<size_t*>(size_elements));
  fprintf(stderr, "unsafe object metadata: %p\n", metadata.get());
  if (metadata == nullptr) {
    return reinterpret_cast<jlong>(nullptr);
  }

  // make sure the "PostConstruct" been invoked
  metadata->ForceLocal();

  auto object =
      vineyard::ObjectFactory::Create(metadata->GetTypeName(), *metadata);
  fprintf(stderr, "resolved FFI objects: %p\n", object.get());

  env->ReleaseStringUTFChars(meta, chars);
  env->ReleaseLongArrayElements(objects, object_elements, JNI_ABORT);
  env->ReleaseLongArrayElements(pointers, pointer_elements, JNI_ABORT);
  env->ReleaseLongArrayElements(sizes, size_elements, JNI_ABORT);

  // release the metadata that not be used anymore.
  metadata.reset();

  return reinterpret_cast<jlong>(object.release());
}

#ifdef __cplusplus
}
#endif
