#include "common/util/typename.h"

#include <unordered_map>
#include <memory>
#include <iostream>
#ifndef MODULES_FUSE_ADAPTORS_FACTORY_ADAPTOR_FACTORY
#define MODULES_FUSE_ADAPTORS_FACTORY_ADAPTOR_FACTORY
namespace vineyard {
namespace fuse {
using vineyard_deserializers_getter_t = void* (*) ();
using vineyard_deserializer_t = void* (*) (void*);


/**
 * @brief FORCE_INSTANTIATE is a tool to guarantee the argument not be optimized
 * by the compiler, even when it is unused. This trick is useful when we want
 * hold a reference of static member in constructors.
 */
template <typename T>
inline void FORCE_INSTANTIATE(T) {}


class AdaptorFactory  {


   public:
    template <typename T>
    static bool Register() {
      const std::string name = type_name<T>();
#ifndef NDEBUG
/*       static bool __trace = !read_env("VINEYARD_TRACE_REGISTRY").empty();
      if (__trace) {
        // See: Note [std::cerr instead of DVLOG()]
        std::clog << "vineyard: register data type: " << name << std::endl;
      } */
      std::clog<<"regsiter the "<< name<<std::endl;
#endif
      auto& known_deserializers = getDeserializers();
      // the explicit `static_cast` is used to help overloading resolution.
      known_deserializers[name] = static_cast<vineyard_deserializer_t>(&T::deserialize);
      return true;
    }

    static vineyard_deserializer_t getDeserializer(std::string& type){
      auto& m=getDeserializers();
      if(m.find(type)==m.end()){
        #ifndef NDEBUG
    std::clog << "[debug] create an instance with the unknown typename: "
              << type << std::endl;
#endif
    return nullptr; 
      }else{
        return m[type];
      }
    }

   private:
    static std::unordered_map<std::string, vineyard_deserializer_t>&
    getDeserializers();
    static vineyard_deserializers_getter_t __GetGlobalVineyardFuseAdaptorsRegistry;

  };
 
} // namespace fuse
} // namespace vineyard
#endif