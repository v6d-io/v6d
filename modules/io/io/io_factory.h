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

#ifndef MODULES_IO_IO_IO_FACTORY_H_
#define MODULES_IO_IO_IO_FACTORY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "client/client.h"
#include "io/io/i_io_adaptor.h"

namespace vineyard {
/** I/O factory.
 *
 * I/O factory, encapsulating I/O adaptor. You can use <CreateIOAdaptor> to
 * obtain a kind of I/O adaptor, such as S3, HDFS, HTTP, and Local I/O adaptor.
 */
class IOFactory {
 public:
  /** Create an I/O adaptor.
   * @param location the file location.
   *
   * @return a kind of I/O adaptor.
   */
  static void Init();

  static void Finalize();

  /** Create an I/O adaptor.
   * @param location the file location.
   *
   * @return a kind of I/O adaptor.
   */
  static std::unique_ptr<IIOAdaptor> CreateIOAdaptor(
      const std::string& location, Client* client = nullptr);

  using io_initializer_t = std::unique_ptr<IIOAdaptor> (*)(const std::string&,
                                                           Client* client);

  static bool Register(std::string const& kind, io_initializer_t initializer);

  static bool Register(std::vector<std::string> const& kinds,
                       io_initializer_t initializer);

 private:
  static std::unordered_map<std::string, io_initializer_t>& getKnownAdaptors();
};

}  // namespace vineyard

#endif  // MODULES_IO_IO_IO_FACTORY_H_
