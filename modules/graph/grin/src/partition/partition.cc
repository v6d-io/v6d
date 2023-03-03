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
#include "client/client.h"


#ifdef GRIN_ENABLE_GRAPH_PARTITION
size_t grin_get_total_partitions_number(GRIN_PARTITIONED_GRAPH pg) {
    auto _pg = static_cast<GRIN_PARTITIONED_GRAPH_T*>(pg);
    return _pg->total_frag_num();
}

GRIN_PARTITION_LIST grin_get_local_partition_list(GRIN_PARTITIONED_GRAPH pg) {
    auto _pg = static_cast<GRIN_PARTITIONED_GRAPH_T*>(pg);
    auto pl = new GRIN_PARTITION_LIST_T();
    vineyard::Client client;
    client.Connect();
    for (auto & [fid, location] : _pg->FragmentLocations()) {
        if (location == client.instance_id()) {
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
    vineyard::Client client;
    client.Connect();
    return get_graph_by_object_id(client, _pg->Fragments().at(*_p));
}
#endif

#ifdef GRIN_NATURAL_PARTITION_ID_TRAIT
GRIN_PARTITION grin_get_partition_from_id(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION_ID pid) {
    auto p = new GRIN_PARTITION_T(pid);
    return p;
}

GRIN_PARTITION_ID grin_get_partition_id(GRIN_PARTITIONED_GRAPH pg, GRIN_PARTITION p) {
    auto _p = static_cast<GRIN_PARTITION_T*>(p);
    return *_p;
}
#endif


#if defined(GRIN_ENABLE_GRAPH_PARTITION) && defined(GRIN_ENABLE_VERTEX_LIST)
GRIN_VERTEX_LIST grin_get_master_vertices(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vl = new GRIN_VERTEX_LIST_T();
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        _vl->push_back(_g->InnerVertices(vtype));
    }
    return _vl;    
}

GRIN_VERTEX_LIST grin_get_mirror_vertices(GRIN_GRAPH g) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vl = new GRIN_VERTEX_LIST_T();
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        _vl->push_back(_g->OuterVertices(vtype));
    }
    return _vl;
}

GRIN_VERTEX_LIST grin_get_mirror_vertices_by_partition(GRIN_GRAPH g, GRIN_PARTITION p) {
    return GRIN_NULL_LIST;
}

#ifdef GRIN_WITH_VERTEX_PROPERTY
GRIN_VERTEX_LIST grin_get_master_vertices_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vtype = static_cast<GRIN_VERTEX_TYPE_T*>(vtype);
    auto _vl = new GRIN_VERTEX_LIST_T();
    _vl->push_back(_g->InnerVertices(*_vtype));
    return _vl;
}

GRIN_VERTEX_LIST grin_get_mirror_vertices_by_type(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vtype = static_cast<GRIN_VERTEX_TYPE_T*>(vtype);
    auto _vl = new GRIN_VERTEX_LIST_T();
    _vl->push_back(_g->OuterVertices(*_vtype));
    return _vl;
}

GRIN_VERTEX_LIST grin_get_mirror_vertices_by_type_partition(GRIN_GRAPH g, GRIN_VERTEX_TYPE vtype,
                                                  GRIN_PARTITION p) {
    return GRIN_NULL_LIST;
}
#endif
#endif

#if defined(GRIN_ENABLE_GRAPH_PARTITION) && defined(GRIN_ENABLE_ADJACENT_LIST)
GRIN_ADJACENT_LIST grin_get_adjacent_master_list(GRIN_GRAPH g, GRIN_DIRECTION d, GRIN_VERTEX v) {
    return GRIN_NULL_LIST;
}

GRIN_ADJACENT_LIST grin_get_adjacent_mirror_list(GRIN_GRAPH g, GRIN_DIRECTION d, GRIN_VERTEX v) {
    return GRIN_NULL_LIST;
}

GRIN_ADJACENT_LIST grin_get_adjacent_mirror_list_by_partition(GRIN_GRAPH g, GRIN_DIRECTION d,
                                                   GRIN_PARTITION p, GRIN_VERTEX v) {
    return GRIN_NULL_LIST;
}
#endif


#ifdef GRIN_ENABLE_VERTEX_REF
GRIN_VERTEX_REF grin_get_vertex_ref_for_vertex(GRIN_GRAPH g, GRIN_VERTEX v) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    auto gid = _g->Vertex2Gid(*_v);
    auto vr = new GRIN_VERTEX_REF_T(gid);
    return vr;
}

GRIN_VERTEX grin_get_vertex_from_vertex_ref(GRIN_GRAPH g, GRIN_VERTEX_REF vr) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vr = static_cast<GRIN_VERTEX_REF_T*>(vr);
    auto v = new GRIN_VERTEX_T();
    if (_g->Gid2Vertex(*_vr, *v)) {
        return v;
    }
    return GRIN_NULL_VERTEX;
}

bool grin_is_master_vertex(GRIN_GRAPH g, GRIN_VERTEX v) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    return _g->IsInnerVertex(*_v);
}

bool grin_is_mirror_vertex(GRIN_GRAPH g, GRIN_VERTEX v) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    return _g->IsOuterVertex(*_v);
}

GRIN_PARTITION grin_get_master_partition_from_vertex_ref(GRIN_GRAPH g, GRIN_VERTEX_REF vr) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _vr = static_cast<GRIN_VERTEX_REF_T*>(vr);
    auto id_parser = vineyard::IdParser<GRIN_VERTEX_REF_T>();
    id_parser.Init(_g->fnum(), _g->vertex_label_num());
    auto p = new GRIN_PARTITION_T(id_parser.GetFid(*_vr));
    return p;
}

const char* grin_serialize_vertex_ref(GRIN_GRAPH g, GRIN_VERTEX_REF vr) {
    auto _vr = static_cast<GRIN_VERTEX_REF_T*>(vr);
    std::stringstream ss;
    ss << *_vr;
    int len = ss.str().length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", ss.str().c_str());
    return out;
}

GRIN_VERTEX_REF grin_deserialize_to_vertex_ref(GRIN_GRAPH g, const char* msg) {
    std::stringstream ss(msg);
    GRIN_VERTEX_REF_T gid;
    ss >> gid;
    auto vr = new GRIN_VERTEX_REF_T(gid);
    return vr;
}
#endif

#ifdef GRIN_ENABLE_GRAPH_PARTITION
bool grin_is_vertex_neighbor_local_complete(GRIN_GRAPH g, GRIN_VERTEX v) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    return _g->IsInnerVertex(*_v);
}

GRIN_PARTITION_LIST grin_vertex_neighbor_complete_partitions(GRIN_GRAPH g, GRIN_VERTEX v) {
    auto _g = static_cast<GRIN_GRAPH_T*>(g);
    auto _v = static_cast<GRIN_VERTEX_T*>(v);
    auto pl = new GRIN_PARTITION_LIST_T();
    pl->push_back(_g->GetFragId(*_v));
    return pl;
}
#endif

#ifdef GRIN_WITH_VERTEX_PROPERTY
bool grin_is_vertex_property_local_complete(GRIN_GRAPH g, GRIN_VERTEX v) {
    return grin_is_vertex_neighbor_local_complete(g, v);
}

GRIN_PARTITION_LIST grin_vertex_property_complete_partitions(GRIN_GRAPH g, GRIN_VERTEX v) {
    return grin_vertex_neighbor_complete_partitions(g, v);
}
#endif


#ifdef GRIN_WITH_EDGE_PROPERTY
bool grin_is_edge_property_local_complete(GRIN_GRAPH g, GRIN_EDGE e) {
    return true;
}

GRIN_PARTITION_LIST edge_property_complete_partitions(GRIN_GRAPH g, GRIN_EDGE e) {
    return GRIN_NULL_LIST;
}
#endif
