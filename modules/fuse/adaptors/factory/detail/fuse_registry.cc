#include "../adaptor_factory.h"

#include<unordered_map>
extern "C" __attribute__((visibility("default"))) void*
__CreateGlobalVineyardFuseAdaptorsRegistry() {
  #ifndef NDEBUG
      // See: Note [std::clog instead of DVLOG()]
      std::clog << "__CreateGlobalVineyardFuseAdaptorsRegistry is called " << std::endl;
#endif
  static std::unordered_map<
      std::string, vineyard::fuse::vineyard_deserializer_t>* known_adaptors =
      new std::unordered_map<std::string,
                             vineyard::fuse::vineyard_deserializer_t>();
      
  return reinterpret_cast<void*>(known_adaptors);
}