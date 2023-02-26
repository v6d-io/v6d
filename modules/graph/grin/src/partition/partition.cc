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

#include "modules/graph/grin/src/predefine.h"
#include "modules/graph/grin/src/utils.h"
#include "modules/graph/grin/include/partition/partition.h"
#include "modules/graph/fragment/property_graph_types.h"
#include "src/client/client.h"


#ifdef ENABLE_GRAPH_PARTITION
size_t get_total_partitions_number(const PartitionedGraph pg) {
    auto _pg = static_cast<PartitionedGraph_T*>(pg);
    return _pg->total_frag_num();
}

PartitionList get_local_partition_list(const PartitionedGraph pg) {
    auto _pg = static_cast<PartitionedGraph_T*>(pg);
    auto pl = new PartitionList_T();
    vineyard::Client client;
    client.Connect();
    for (const auto & [fid, location] : _pg->FragmentLocations()) {
        if (location == client.instance_id()) {
            pl->push_back(fid);
        }
    }
    return pl;
}

void destroy_partition_list(PartitionList pl) {
    auto _pl = static_cast<PartitionList_T*>(pl);
    delete _pl;
}

PartitionList create_partition_list() {
    auto pl = new PartitionList_T();
    return pl;
}

bool insert_partition_to_list(PartitionList pl, const Partition p) {
    auto _pl = static_cast<PartitionList_T*>(pl);
    auto _p = static_cast<Partition_T*>(p);
    _pl->push_back(*_p);
    return true;
}

size_t get_partition_list_size(const PartitionList pl) {
    auto _pl = static_cast<PartitionList_T*>(pl);
    return _pl->size();
}

Partition get_partition_from_list(const PartitionList pl, const size_t idx) {
    auto _pl = static_cast<PartitionList_T*>(pl);
    auto p = new Partition_T((*_pl)[idx]);
    return p;
}

void destroy_partition(Partition p) {
    auto _p = static_cast<Partition_T*>(p);
    delete _p;
}

void* get_partition_info(const PartitionedGraph pg, const Partition p) {
    return NULL;
}

Graph get_local_graph_from_partition(const PartitionedGraph pg, const Partition p) {
    auto _pg = static_cast<PartitionedGraph_T*>(pg);
    auto _p = static_cast<Partition_T*>(p);
    vineyard::Client client;
    client.Connect();
    return get_graph_by_object_id(client, _pg->Fragments().at(*_p));
}
#endif

#ifdef NATURAL_PARTITION_ID_TRAIT
Partition get_partition_from_id(const PartitionID pid) {
    auto p = new Partition_T(pid);
    return p;
}

PartitionID get_partition_id(const Partition p) {
    auto _p = static_cast<Partition_T*>(p);
    return *_p;
}
#endif


#if defined(ENABLE_GRAPH_PARTITION) && defined(ENABLE_VERTEX_LIST)
VertexList get_master_vertices(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vl = new VertexList_T();
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        _vl->push_back(_g->InnerVertices(vtype));
    }
    return _vl;    
}

VertexList get_mirror_vertices(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vl = new VertexList_T();
    for (auto vtype = 0; vtype < _g->vertex_label_num(); ++vtype) {
        _vl->push_back(_g->OuterVertices(vtype));
    }
    return _vl;
}

VertexList get_mirror_vertices_by_partition(const Graph g, const Partition p) {
    return NULL_LIST;
}

#ifdef WITH_VERTEX_PROPERTY
VertexList get_master_vertices_by_type(const Graph g, const VertexType vtype) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vtype = static_cast<VertexType_T*>(vtype);
    auto _vl = new VertexList_T();
    _vl->push_back(_g->InnerVertices(*_vtype));
    return _vl;
}

VertexList get_mirror_vertices_by_type(const Graph g, const VertexType vtype) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vtype = static_cast<VertexType_T*>(vtype);
    auto _vl = new VertexList_T();
    _vl->push_back(_g->OuterVertices(*_vtype));
    return _vl;
}

VertexList get_mirror_vertices_by_type_partition(const Graph g, const VertexType vtype,
                                                  const Partition p) {
    return NULL_LIST;
}
#endif
#endif

#if defined(ENABLE_GRAPH_PARTITION) && defined(ENABLE_ADJACENT_LIST)
AdjacentList get_adjacent_master_list(const Graph g, const Direction d, const Vertex v) {
    return NULL_LIST;
}

AdjacentList get_adjacent_mirror_list(const Graph g, const Direction d, const Vertex v) {
    return NULL_LIST;
}

AdjacentList get_adjacent_mirror_list_by_partition(const Graph g, const Direction d,
                                                   const Partition p, const Vertex v) {
    return NULL_LIST;
}
#endif


#ifdef ENABLE_VERTEX_REF
VertexRef get_vertex_ref_for_vertex(const Graph g, const Partition p, const Vertex v) {
    auto _g = static_cast<Graph_T*>(g);
    auto _v = static_cast<Vertex_T*>(v);
    auto gid = _g->Vertex2Gid(*_v);
    auto vr = new VertexRef_T(gid);
    return vr;
}

Vertex get_vertex_from_vertex_ref(const Graph g, const VertexRef vr) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vr = static_cast<VertexRef_T*>(vr);
    auto v = new Vertex_T();
    if (_g->Gid2Vertex(*_vr, *v)) {
        return v;
    }
    return NULL_VERTEX;
}

Partition get_master_partition_from_vertex_ref(const Graph g, const VertexRef vr) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vr = static_cast<VertexRef_T*>(vr);
    auto id_parser = vineyard::IdParser<VertexRef_T>();
    id_parser.Init(_g->fnum(), _g->vertex_label_num());
    auto p = new Partition_T(id_parser.GetFid(*_vr));
    return p;
}

char* serialize_vertex_ref(const Graph g, const VertexRef vr) {
    auto _vr = static_cast<VertexRef_T*>(vr);
    std::stringstream ss;
    ss << *_vr;
    int len = ss.str().length() + 1;
    char* out = new char[len];
    snprintf(out, len, "%s", ss.str().c_str());
    return out;
}

VertexRef deserialize_to_vertex_ref(const Graph g, const char* msg) {
    std::stringstream ss(msg);
    VertexRef_T gid;
    ss >> gid;
    auto vr = new VertexRef_T(gid);
    return vr;
}
#endif

#ifdef ENABLE_GRAPH_PARTITION
bool is_vertex_neighbor_local_complete(const Graph g, const Vertex v) {
    auto _g = static_cast<Graph_T*>(g);
    auto _v = static_cast<Vertex_T*>(v);
    return _g->IsInnerVertex(*_v);
}

PartitionList vertex_neighbor_complete_partitions(const Graph g, const Vertex v) {
    auto _g = static_cast<Graph_T*>(g);
    auto _v = static_cast<Vertex_T*>(v);
    auto pl = new PartitionList_T();
    pl->push_back(_g->GetFragId(*_v));
    return pl;
}
#endif

#ifdef WITH_VERTEX_PROPERTY
bool is_vertex_property_local_complete(const Graph g, const Vertex v) {
    return is_vertex_neighbor_local_complete(g, v);
}

PartitionList vertex_property_complete_partitions(const Graph g, const Vertex v) {
    return vertex_neighbor_complete_partitions(g, v);
}
#endif


#ifdef WITH_EDGE_PROPERTY
bool is_edge_property_local_complete(const Graph g, const Edge e) {
    return true;
}

PartitionList edge_property_complete_partitions(const Graph g, const Edge e) {
    return NULL_LIST;
}
#endif
