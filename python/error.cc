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

#include <functional>
#include <stdexcept>

#include "common/util/status.h"

#include "pybind11/pybind11.h"
#include "pybind11_utils.h"  // NOLINT(build/include_subdir)

namespace py = pybind11;

namespace vineyard {

#ifndef DEFINE_PYBIND_EXCEPTION
#define DEFINE_PYBIND_EXCEPTION(name)                  \
  struct name##Exception : public std::runtime_error { \
    name##Exception(std::string const& message)        \
        : std::runtime_error(message) {}               \
  }
#endif

#ifndef REGISTER_PYBIND_EXCEPTION
#define REGISTER_PYBIND_EXCEPTION(mod, name)     \
  py::register_exception<name##Exception>(       \
      mod, VINEYARD_STRINGIFY(name) "Exception", \
      pybind11::detail::get_exception_object<VineyardException>())
#endif

#ifndef THROW_ON_ERROR_OF
#define THROW_ON_ERROR_OF(name) \
  case StatusCode::k##name:     \
    throw name##Exception(status.ToString())
#endif

DEFINE_PYBIND_EXCEPTION(Vineyard);
DEFINE_PYBIND_EXCEPTION(Invalid);
DEFINE_PYBIND_EXCEPTION(KeyError);
DEFINE_PYBIND_EXCEPTION(TypeError);
DEFINE_PYBIND_EXCEPTION(IOError);
DEFINE_PYBIND_EXCEPTION(EndOfFile);
DEFINE_PYBIND_EXCEPTION(NotImplemented);
DEFINE_PYBIND_EXCEPTION(AssertionFailed);
DEFINE_PYBIND_EXCEPTION(UserInputError);
DEFINE_PYBIND_EXCEPTION(ObjectExists);
DEFINE_PYBIND_EXCEPTION(ObjectNotExists);
DEFINE_PYBIND_EXCEPTION(ObjectSealed);
DEFINE_PYBIND_EXCEPTION(ObjectNotSealed);
DEFINE_PYBIND_EXCEPTION(ObjectIsBlob);
DEFINE_PYBIND_EXCEPTION(ObjectTypeError);
DEFINE_PYBIND_EXCEPTION(MetaTreeInvalid);
DEFINE_PYBIND_EXCEPTION(MetaTreeTypeInvalid);
DEFINE_PYBIND_EXCEPTION(MetaTreeTypeNotExists);
DEFINE_PYBIND_EXCEPTION(MetaTreeNameInvalid);
DEFINE_PYBIND_EXCEPTION(MetaTreeNameNotExists);
DEFINE_PYBIND_EXCEPTION(MetaTreeLinkInvalid);
DEFINE_PYBIND_EXCEPTION(MetaTreeSubtreeNotExists);
DEFINE_PYBIND_EXCEPTION(VineyardServerNotReady);
DEFINE_PYBIND_EXCEPTION(ArrowError);
DEFINE_PYBIND_EXCEPTION(ConnectionFailed);
DEFINE_PYBIND_EXCEPTION(ConnectionError);
DEFINE_PYBIND_EXCEPTION(EtcdError);
DEFINE_PYBIND_EXCEPTION(AlreadyStopped);
DEFINE_PYBIND_EXCEPTION(RedisError);
DEFINE_PYBIND_EXCEPTION(NotEnoughMemory);
DEFINE_PYBIND_EXCEPTION(StreamDrained);
DEFINE_PYBIND_EXCEPTION(StreamFailed);
DEFINE_PYBIND_EXCEPTION(InvalidStreamState);
DEFINE_PYBIND_EXCEPTION(StreamOpened);
DEFINE_PYBIND_EXCEPTION(GlobalObjectInvalid);
DEFINE_PYBIND_EXCEPTION(UnknownError);

void bind_error(py::module& mod) {
  py::register_exception<VineyardException>(mod, "VineyardException",
                                            PyExc_RuntimeError);
  REGISTER_PYBIND_EXCEPTION(mod, Invalid);
  REGISTER_PYBIND_EXCEPTION(mod, KeyError);
  REGISTER_PYBIND_EXCEPTION(mod, TypeError);
  REGISTER_PYBIND_EXCEPTION(mod, IOError);
  REGISTER_PYBIND_EXCEPTION(mod, EndOfFile);
  REGISTER_PYBIND_EXCEPTION(mod, NotImplemented);
  REGISTER_PYBIND_EXCEPTION(mod, AssertionFailed);
  REGISTER_PYBIND_EXCEPTION(mod, UserInputError);
  REGISTER_PYBIND_EXCEPTION(mod, ObjectExists);
  REGISTER_PYBIND_EXCEPTION(mod, ObjectNotExists);
  REGISTER_PYBIND_EXCEPTION(mod, ObjectSealed);
  REGISTER_PYBIND_EXCEPTION(mod, ObjectNotSealed);
  REGISTER_PYBIND_EXCEPTION(mod, ObjectIsBlob);
  REGISTER_PYBIND_EXCEPTION(mod, ObjectTypeError);
  REGISTER_PYBIND_EXCEPTION(mod, MetaTreeInvalid);
  REGISTER_PYBIND_EXCEPTION(mod, MetaTreeTypeInvalid);
  REGISTER_PYBIND_EXCEPTION(mod, MetaTreeTypeNotExists);
  REGISTER_PYBIND_EXCEPTION(mod, MetaTreeNameInvalid);
  REGISTER_PYBIND_EXCEPTION(mod, MetaTreeNameNotExists);
  REGISTER_PYBIND_EXCEPTION(mod, MetaTreeLinkInvalid);
  REGISTER_PYBIND_EXCEPTION(mod, MetaTreeSubtreeNotExists);
  REGISTER_PYBIND_EXCEPTION(mod, VineyardServerNotReady);
  REGISTER_PYBIND_EXCEPTION(mod, ArrowError);
  REGISTER_PYBIND_EXCEPTION(mod, ConnectionFailed);
  REGISTER_PYBIND_EXCEPTION(mod, ConnectionError);
  REGISTER_PYBIND_EXCEPTION(mod, EtcdError);
  REGISTER_PYBIND_EXCEPTION(mod, AlreadyStopped);
  REGISTER_PYBIND_EXCEPTION(mod, RedisError);
  REGISTER_PYBIND_EXCEPTION(mod, NotEnoughMemory);
  REGISTER_PYBIND_EXCEPTION(mod, StreamDrained);
  REGISTER_PYBIND_EXCEPTION(mod, StreamFailed);
  REGISTER_PYBIND_EXCEPTION(mod, InvalidStreamState);
  REGISTER_PYBIND_EXCEPTION(mod, StreamOpened);
  REGISTER_PYBIND_EXCEPTION(mod, GlobalObjectInvalid);
  REGISTER_PYBIND_EXCEPTION(mod, UnknownError);
}

void throw_on_error(Status const& status) {
  switch (status.code()) {
  case StatusCode::kOK:
    return;
    THROW_ON_ERROR_OF(Invalid);
    THROW_ON_ERROR_OF(KeyError);
    THROW_ON_ERROR_OF(TypeError);
    THROW_ON_ERROR_OF(IOError);
    THROW_ON_ERROR_OF(EndOfFile);
    THROW_ON_ERROR_OF(NotImplemented);
    THROW_ON_ERROR_OF(AssertionFailed);
    THROW_ON_ERROR_OF(UserInputError);
    THROW_ON_ERROR_OF(ObjectExists);
    THROW_ON_ERROR_OF(ObjectNotExists);
    THROW_ON_ERROR_OF(ObjectSealed);
    THROW_ON_ERROR_OF(ObjectNotSealed);
    THROW_ON_ERROR_OF(ObjectIsBlob);
    THROW_ON_ERROR_OF(ObjectTypeError);
    THROW_ON_ERROR_OF(MetaTreeInvalid);
    THROW_ON_ERROR_OF(MetaTreeTypeInvalid);
    THROW_ON_ERROR_OF(MetaTreeTypeNotExists);
    THROW_ON_ERROR_OF(MetaTreeNameInvalid);
    THROW_ON_ERROR_OF(MetaTreeNameNotExists);
    THROW_ON_ERROR_OF(MetaTreeLinkInvalid);
    THROW_ON_ERROR_OF(MetaTreeSubtreeNotExists);
    THROW_ON_ERROR_OF(VineyardServerNotReady);
    THROW_ON_ERROR_OF(ArrowError);
    THROW_ON_ERROR_OF(ConnectionFailed);
    THROW_ON_ERROR_OF(ConnectionError);
    THROW_ON_ERROR_OF(EtcdError);
    THROW_ON_ERROR_OF(AlreadyStopped);
    THROW_ON_ERROR_OF(RedisError);
    THROW_ON_ERROR_OF(NotEnoughMemory);
    THROW_ON_ERROR_OF(StreamDrained);
    THROW_ON_ERROR_OF(StreamFailed);
    THROW_ON_ERROR_OF(InvalidStreamState);
    THROW_ON_ERROR_OF(StreamOpened);
    THROW_ON_ERROR_OF(GlobalObjectInvalid);
    THROW_ON_ERROR_OF(UnknownError);
  default:
    throw std::runtime_error("Unknow status: " + status.ToString());
  }
}

}  // namespace vineyard
