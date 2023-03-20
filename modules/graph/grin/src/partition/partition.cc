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

#include "graph/grin/src/predefine.h"
#include "graph/grin/include/partition/partition.h"
#include "graph/fragment/property_graph_types.h"


#ifdef GRIN_ENABLE_GRAPH_PARTITION
size_t grin_get_total_partitions_number(GRIN_PARTITIONED_GRAPH pg) {
    auto _pg = static_cast<GRIN_PARTITIONED_GRAPH_T*>(pg);
    return _pg->pg->total_frag_num();
}

GRIN_PARTITION_LIST grin_get_local_partition_list(GRIN_PARTITIONED_GRAPH pg) {
    auto _pg = static_cast<GRIN_PARTITIONED_GRAPH_T*>(pg);
    auto pl = new GRIN_PARTITION_LIST_T();
    for (auto fid = 0; fid < _pg->pg->total_frag_num(); ++fid) {
        if (_pg->lgs[fid] != nullptr) {
            pl->push_back(fid);
        }
    }
    return pl;
}

void grin_destroy_partition_list(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION_LIST pl) {
    auto _pl = static_cast<GRIN_PARTITION_LIST_T*>(pl);
    delete _pl;
}

GRIN_PARTITION_LIST grin_create_partition_list(GRIN_PARTITIONED_GRAPH pg) {
    auto pl = new GRIN_PARTITION_LIST_T();
    return pl;
}

bool grin_insert_partition_to_list(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION_LIST pl, GRIN_PARTITION p) {
    auto _pl = static_cast<GRIN_PARTITION_LIST_T*>(pl);
    auto _p = static_cast<GRIN_PARTITION_T*>(p);
    _pl->push_back(*_p);
    return true;
}

size_t grin_get_partition_list_size(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION_LIST pl) {
    auto _pl = static_cast<GRIN_PARTITION_LIST_T*>(pl);
    return _pl->size();
}

GRIN_PARTITION grin_get_partition_from_list(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION_LIST pl, size_t idx) {
    auto _pl = static_cast<GRIN_PARTITION_LIST_T*>(pl);
    auto p = new GRIN_PARTITION_T((*_pl)[idx]);
    return p;
}

bool grin_equal_partition(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION p1, GRIN_PARTITION p2) {
    auto _p1 = static_cast<GRIN_PARTITION_T*>(p1);
    auto _p2 = static_cast<GRIN_PARTITION_T*>(p2);
    return (*_p1 == *_p2);
}

void grin_destroy_partition(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION p) {
    auto _p = static_cast<GRIN_PARTITION_T*>(p);
    delete _p;
}

void* grin_get_partition_info(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION p) {
    return NULL;
}

GRIN_GRAPH grin_get_local_graph_from_partition(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION p) {
    auto _pg = static_cast<GRIN_PARTITIONED_GRAPH_T*>(pg);
    auto _p = static_cast<GRIN_PARTITION_T*>(p);
    return _pg->lgs[*_p];
}
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
GRIN_PARTITION grin_get_partition_from_id(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION_ID pid) {
    auto p = new GRIN_PARTITION_T(pid);
    return p;
}

GRIN_PARTITION_ID grin_get_partition_id(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION p) {
    auto _p = static_cast<GRIN_PARTITION_T*>(p);
    return *_p;
}
#endif

