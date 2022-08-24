#ifndef MODULES_FUSE_CACHE_MANAGER_MANAGER_HPP_
#define MODULES_FUSE_CACHE_MANAGER_MANAGER_HPP_

#include "fuse/cache_manager/manager.h"
#include "arrow/buffer.h"

namespace vineyard {
namespace fuse {

namespace cache_manager {
template <typename KV>
void CacheManager<KV>::popToNBytes(size_t n) {
  while (this->curBytes > n) {
    auto keyToBeDel = myList.back().key;
    auto dataToBeDel = myList.back().value;
    this->curBytes -= dataToBeDel->capacity();
    myList.pop_back();
    myMap.erase(keyToBeDel);
    DLOG(INFO) << "remove key: " << keyToBeDel << " value: " << dataToBeDel->ToString()<< " remaining bytes: "<<this->curBytes;
  }
}
template <class KV>
bool CacheManager<KV>::WithInCapacity(size_t data) {
  return data <= capacityBytes;
}
template<class KV>
CacheManager<KV>::CacheManager(size_t capacityBytes):capacityBytes(capacityBytes),curBytes(0){
}
template<class KV>
CacheManager<KV>::CacheManager():capacityBytes(0),curBytes(0){
}
template<class KV>
void CacheManager<KV>::resize(size_t targetCapacityBytes){
  capacityBytes =  targetCapacityBytes;
}
template<class KV>
void CacheManager<KV>::destroy(){
  this->~CacheManager();
}
template<class KV>
bool CacheManager<KV>::has(const typename KV::KeyType& key){
  return myMap.find(key)!= myMap.end();
}
template<class KV>
typename KV::ValType CacheManager<KV>::operator[](const typename KV::KeyType& key) {
  return get(key);
}

template<class KV>
size_t CacheManager<KV>::getCapacityBytes(){
        return this->capacityBytes;
}
template<class KV>
size_t CacheManager<KV>::getCurBytes(){
        return this->curBytes;
}
template <class KV>
Status CacheManager<KV>::put(const typename KV::KeyType& key, typename KV::ValType v) {

  if (WithInCapacity(v->capacity())) {

    auto found_map_iter = myMap.find(key);

    if (found_map_iter != myMap.end()) {
      DLOG(INFO) << "update key: " << key << " value: " << v->ToString()<<std::endl;

      auto found_key = found_map_iter->first;
      auto& found_kv = found_map_iter->second;

      curBytes -= found_kv->value->capacity();
      popToNBytes(capacityBytes - v->capacity());
      myList.splice(myList.begin(), this->myList, found_kv);
      found_kv->value = v;
      return Status::OK();
    } else {
      DLOG(INFO) << "put key: " << key << " value: " << v->ToString()<<std::endl;
      popToNBytes(capacityBytes - v->capacity());
      myList.emplace_front(key,v);
      // decltype(myMap[key])::nothing;
      myMap[key] = myList.begin();
      this->curBytes += v->capacity();
      return Status::OK();
    }
  } else {
    DLOG(INFO)<<"this keyvalue is too large to put int"<<std::endl;
    return Status::NotEnoughMemory("");
  }
}
template <class KV>

  std::list<KV> CacheManager<KV>::getLinkedList(){
    return myList;

  }

template <class KV>
typename KV::ValType CacheManager<KV>::get(const typename KV::KeyType& key) {
  auto found_iter = myMap.find(key);
  if (found_iter == myMap.end())  // key doesn't exist
    {
          DLOG(INFO)<< "not found key " << key; 

    return nullptr;}
    DLOG(INFO)<< "found key " << key; 

  myList.splice(
      myList.begin(), myList,
      found_iter->second);  // move the node corresponding to key to front
  return found_iter->second->value;
}
}  // namespace cache_manager

}  // namespace fuse
}  // namespace vineyard
#endif  // MODULES_FUSE_CACHE_MANAGER_MANAGER_HPP_