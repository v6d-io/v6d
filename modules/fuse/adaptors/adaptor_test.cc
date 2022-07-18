#include<memory>
#include<string>
#include"factory/adaptor_factory.h"
#include"factory/mock.h"
vineyard::fuse::AdaptorFactory adt;
vineyard::fuse::Mock m;
int main(){

    std::string t = "vineyard::fuse::Mock";
    auto d = vineyard::fuse::AdaptorFactory::getDeserializer(t);
    d(nullptr);
    return 0;
}