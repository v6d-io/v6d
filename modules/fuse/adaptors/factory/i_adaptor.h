#include "adaptor_factory.h"
#include "common/util/typename.h"
namespace vineyard
{
    namespace fuse
    {
      


        template <typename T>
        class FuseAdaptorRegistered {
        protected:
        __attribute__((visibility("default"))) FuseAdaptorRegistered() {
                #ifndef NDEBUG
                    // See: Note [std::clog instead of DVLOG()]
                    std::clog << "vineyard::fuse::detail::register" <<type_name<T>()<< std::endl;
                #endif
            FORCE_INSTANTIATE(registered);
        }        
        private:
        __attribute__((visibility("default"))) static const bool registered;
        };
        template <typename T>
        const bool FuseAdaptorRegistered<T>::registered = AdaptorFactory::Register<T>();
       
    } // namespace fuse
    
} // namespace vineyard

