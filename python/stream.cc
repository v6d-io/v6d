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

#include <functional>
#include <map>
#include <memory>
#include <unordered_map>

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

#pragma GCC visibility push(default)
#include "basic/stream/byte_stream.h"
#include "basic/stream/dataframe_stream.h"
#include "client/client.h"
#pragma GCC visibility pop

#include "pybind11_utils.h"  // NOLINT(build/include)

namespace py = pybind11;
using namespace py::literals;  // NOLINT(build/namespaces)

namespace vineyard {

void bind_stream(py::module& mod) {
  // ByteStreamWriter
  py::class_<ByteStreamWriter, std::unique_ptr<ByteStreamWriter>>(
      mod, "ByteStreamWriter")
      .def(
          "next",
          [](ByteStreamWriter* self, size_t const size) -> py::object {
            std::unique_ptr<arrow::MutableBuffer> chunk = nullptr;
            throw_on_error(self->GetNext(size, chunk));
            auto chunk_ptr = chunk.release();
            auto pa = py::module::import("pyarrow");
            return pa.attr("py_buffer")(py::memoryview::from_memory(
                chunk_ptr->mutable_data(), chunk_ptr->size(), false));
          },
          "size"_a)
      .def("finish",
           [](ByteStreamWriter* self) { throw_on_error(self->Finish()); })
      .def("abort",
           [](ByteStreamWriter* self) { throw_on_error(self->Abort()); });

  // ByteStreamReader
  py::class_<ByteStreamReader, std::unique_ptr<ByteStreamReader>>(
      mod, "ByteStreamReader")
      .def("next", [](ByteStreamReader* self) -> py::object {
        std::unique_ptr<arrow::Buffer> chunk = nullptr;
        throw_on_error(self->GetNext(chunk));
        auto chunk_ptr = chunk.release();
        auto pa = py::module::import("pyarrow");
        return pa.attr("py_buffer")(py::memoryview::from_memory(
            const_cast<uint8_t *>(chunk_ptr->data()), chunk_ptr->size(), true));
      });

  // ByteStream
  py::class_<ByteStream, std::shared_ptr<ByteStream>, Object>(mod, "ByteStream")
      .def(
          "open_reader",
          [](ByteStream* self,
             Client& client) -> std::unique_ptr<ByteStreamReader> {
            std::unique_ptr<ByteStreamReader> reader = nullptr;
            throw_on_error(self->OpenReader(client, reader));
            return reader;
          },
          "client"_a)
      .def(
          "open_writer",
          [](ByteStream* self,
             Client& client) -> std::unique_ptr<ByteStreamWriter> {
            std::unique_ptr<ByteStreamWriter> writer = nullptr;
            throw_on_error(self->OpenWriter(client, writer));
            return writer;
          },
          "client"_a)
      .def("__getitem__",
           [](ByteStream* self, std::string const& key) {
             return self->GetParams().at(key);
           })
      .def_property_readonly(
          "params", [](ByteStream* self) { return self->GetParams(); });

  // ByteStreamBuilder
  py::class_<ByteStreamBuilder, std::shared_ptr<ByteStreamBuilder>,
             ObjectBuilder>(mod, "ByteStreamBuilder")
      .def(py::init<Client&>())
      .def("__setitem__",
           [](ByteStreamBuilder* self, std::string const& key,
              std::string const& value) { self->SetParam(key, value); })
      .def("set_params",
           [](ByteStreamBuilder* self,
              std::map<std::string, std::string> const& params) {
             for (auto const& kv : params) {
               self->SetParam(kv.first, kv.second);
             }
           })
      .def("set_params",
           [](ByteStreamBuilder* self,
              std::unordered_map<std::string, std::string> const& params) {
             for (auto const& kv : params) {
               self->SetParam(kv.first, kv.second);
             }
           });

  // DataframeStreamWriter
  py::class_<DataframeStreamWriter, std::unique_ptr<DataframeStreamWriter>>(
      mod, "DataframeStreamWriter")
      .def(
          "next",
          [](DataframeStreamWriter* self, size_t const size) -> py::object {
            std::unique_ptr<arrow::MutableBuffer> chunk = nullptr;
            throw_on_error(self->GetNext(size, chunk));
            auto chunk_ptr = chunk.release();
            auto pa = py::module::import("pyarrow");
            return pa.attr("py_buffer")(py::memoryview::from_memory(
                chunk_ptr->mutable_data(), chunk_ptr->size(), false));
          },
          "size"_a)
      .def("finish",
           [](DataframeStreamWriter* self) { throw_on_error(self->Finish()); })
      .def("abort",
           [](DataframeStreamWriter* self) { throw_on_error(self->Abort()); });

  // DataframeStreamReader
  py::class_<DataframeStreamReader, std::unique_ptr<DataframeStreamReader>>(
      mod, "DataframeStreamReader")
      .def("next", [](DataframeStreamReader* self) -> py::object {
        std::unique_ptr<arrow::Buffer> chunk = nullptr;
        throw_on_error(self->GetNext(chunk));
        auto chunk_ptr = chunk.release();
        auto pa = py::module::import("pyarrow");
        return pa.attr("py_buffer")(py::memoryview::from_memory(
            const_cast<uint8_t *>(chunk_ptr->data()), chunk_ptr->size(), true));
      });

  // DataFrameStream
  py::class_<DataframeStream, std::shared_ptr<DataframeStream>, Object>(
      mod, "DataframeStream")
      .def(
          "open_reader",
          [](DataframeStream* self,
             Client& client) -> std::unique_ptr<DataframeStreamReader> {
            std::unique_ptr<DataframeStreamReader> reader = nullptr;
            throw_on_error(self->OpenReader(client, reader));
            return reader;
          },
          "client"_a)
      .def(
          "open_writer",
          [](DataframeStream* self,
             Client& client) -> std::unique_ptr<DataframeStreamWriter> {
            std::unique_ptr<DataframeStreamWriter> writer = nullptr;
            throw_on_error(self->OpenWriter(client, writer));
            return writer;
          },
          "client"_a)
      .def("__getitem__",
           [](DataframeStream* self, std::string const& key) {
             return self->GetParams().at(key);
           })
      .def_property_readonly(
          "params", [](DataframeStream* self) { return self->GetParams(); });

  // DataframeStreamBuilder
  py::class_<DataframeStreamBuilder, std::shared_ptr<DataframeStreamBuilder>,
             ObjectBuilder>(mod, "DataframeStreamBuilder")
      .def(py::init<Client&>())
      .def("__setitem__",
           [](DataframeStreamBuilder* self, std::string const& key,
              std::string const& value) { self->SetParam(key, value); })
      .def("set_params",
           [](DataframeStreamBuilder* self,
              std::map<std::string, std::string> const& params) {
             for (auto const& kv : params) {
               self->SetParam(kv.first, kv.second);
             }
           })
      .def("set_params",
           [](DataframeStreamBuilder* self,
              std::unordered_map<std::string, std::string> const& params) {
             for (auto const& kv : params) {
               self->SetParam(kv.first, kv.second);
             }
           });
}

}  // namespace vineyard
