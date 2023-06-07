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
#include "partition/partition.h"
#include "graph/fragment/property_graph_types.h"

#ifdef GRIN_ENABLE_GRAPH_PARTITION
GRIN_PARTITIONED_GRAPH grin_get_partitioned_graph_from_storage(int argc, char** argv) {
    if (argc < 2) {
        return nullptr;
    }
    auto pg = new GRIN_PARTITIONED_GRAPH_T();
    pg->socket = std::string(argv[0]);
    pg->client.Connect(argv[0]);
    vineyard::ObjectID obj_id;
    std::stringstream ss(argv[1]);
    ss >> obj_id;
    pg->pg = std::dynamic_pointer_cast<vineyard::ArrowFragmentGroup>(pg->client.GetObject(obj_id));
    pg->lgs.resize(pg->pg->total_frag_num(), 0);
    for (auto & [fid, location] : pg->pg->FragmentLocations()) {
        if (location == pg->client.instance_id()) {
            auto obj_id = pg->pg->Fragments().at(fid);
            pg->lgs[fid] = obj_id;
        }
    }
    return pg;
}

void grin_destroy_partitioned_graph(GRIN_PARTITIONED_GRAPH pg) {
    auto _pg = static_cast<GRIN_PARTITIONED_GRAPH_T*>(pg);
    delete _pg;
}

size_t grin_get_total_partitions_number(GRIN_PARTITIONED_GRAPH pg) {
    auto _pg = static_cast<GRIN_PARTITIONED_GRAPH_T*>(pg);
    return _pg->pg->total_frag_num();
}

GRIN_PARTITION_LIST grin_get_local_partition_list(GRIN_PARTITIONED_GRAPH pg) {
    auto _pg = static_cast<GRIN_PARTITIONED_GRAPH_T*>(pg);
    auto pl = new GRIN_PARTITION_LIST_T();
    for (unsigned fid = 0; fid < _pg->pg->total_frag_num(); ++fid) {
        if (_pg->lgs[fid] != 0) {
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
    _pl->push_back(p);
    return true;
}

size_t grin_get_partition_list_size(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION_LIST pl) {
    auto _pl = static_cast<GRIN_PARTITION_LIST_T*>(pl);
    return _pl->size();
}

GRIN_PARTITION grin_get_partition_from_list(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION_LIST pl, size_t idx) {
    auto _pl = static_cast<GRIN_PARTITION_LIST_T*>(pl);
    return (*_pl)[idx];
}

bool grin_equal_partition(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION p1, GRIN_PARTITION p2) {
    return (p1 == p2);
}

void grin_destroy_partition(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION p) {}

const void* grin_get_partition_info(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION p) {
    return NULL;
}

GRIN_GRAPH grin_get_local_graph_by_partition(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION p) {
    auto _pg = static_cast<GRIN_PARTITIONED_GRAPH_T*>(pg);
    auto g = new GRIN_GRAPH_T();
    
    g->client.Connect(_pg->socket);
    g->_g = std::dynamic_pointer_cast<_GRIN_GRAPH_T>(g->client.GetObject(_pg->lgs[p]));
    g->g = g->_g.get();
    _prepare_cache(g);
    return g;
}
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_PARTITION
GRIN_PARTITION grin_get_partition_by_id(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION_ID pid) {
    return pid;
}

GRIN_PARTITION_ID grin_get_partition_id(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION p) {
    return p;
}
#endif

