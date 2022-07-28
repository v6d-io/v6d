#include<unordered_map>
extern "C" __attribute__((visibility("default"))) void*
__GetGlobalVineyardRegistry() {
  static std::unordered_map<
      std::string, vineyard::fuse::DeserializerFactory::object_initializer_t>* known_types =
      new std::unordered_map<std::string,
                             vineyard::fuse::DeserializerFactory::object_initializer_t>();
  return reinterpret_cast<void*>(known_types);
}