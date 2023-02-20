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

#include "grin/src/predefine.h"
#include "grin/include/partition/partition.h"
#include "modules/graph/fragment/property_graph_types.h"

#ifdef ENABLE_GRAPH_PARTITION
// basic partition informations
size_t get_total_partitions_number(const PartitionedGraph pg) {
    // auto _pg = static_cast<PartitionedGraph_T*>(pg);
    // return _pg->fnum();
}

size_t get_total_vertices_number(const PartitionedGraph pg) {
    // auto _pg = static_cast<PartitionedGraph_T*>(pg);
    // return _pg->GetTotalVerticesNum();
}

// partition list
PartitionList get_local_partition_list(const PartitionedGraph pg) {
    // auto _pg = static_cast<PartitionedGraph_T*>(pg);
    // auto pl = new PartitionList_T();
    // pl->push_back(_pg->fid());
    // return pl;
}

void destroy_partition_list(PartitionList);

PartitionList create_partition_list();

bool insert_partition_to_list(PartitionList, const Partition);

size_t get_partition_list_size(const PartitionList);

Partition get_partition_from_list(const PartitionList, const size_t);

void destroy_partition(Partition);

void* get_partition_info(const PartitionedGraph, const Partition);

Graph get_local_graph_from_partition(const PartitionedGraph, const Partition);

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

// master & mirror vertices for vertexcut partition
// while they refer to inner & outer vertices in edgecut partition
#if defined(ENABLE_GRAPH_PARTITION) && defined(ENABLE_VERTEX_LIST)
VertexList get_master_vertices(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vl = new VertexList_T();
    for (VertexLabel_T vlabel = 0; vlabel < _g->vertex_label_num(); ++vlabel) {
        _vl->push_back(_g->InnerVertices(vlabel));
    }
    return _vl;    
}

VertexList get_mirror_vertices(const Graph g) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vl = new VertexList_T();
    for (VertexLabel_T vlabel = 0; vlabel < _g->vertex_label_num(); ++vlabel) {
        _vl->push_back(_g->OuterVertices(vlabel));
    }
    return _vl;
}

VertexList get_mirror_vertices_by_partition(const Graph g, const Partition p) {
    return NULL_LIST;
}

#ifdef WITH_VERTEX_LABEL
VertexList get_master_vertices_by_label(const Graph g, const VertexLabel vlabel) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vlabel = static_cast<VertexLabel_T*>(vlabel);
    auto _vl = new VertexList_T();
    _vl->push_back(_g->InnerVertices(*_vlabel));
    return _vl;
}

VertexList get_mirror_vertices_by_label(const Graph g, const VertexLabel vlabel) {
    auto _g = static_cast<Graph_T*>(g);
    auto _vlabel = static_cast<VertexLabel_T*>(vlabel);
    auto _vl = new VertexList_T();
    _vl->push_back(_g->OuterVertices(*_vlabel));
    return _vl;
}

VertexList get_mirror_vertices_by_label_partition(const Graph g, const VertexLabel vlabel,
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


// Vertex ref refers to the same vertex referred in other partitions,
// while edge ref is likewise. Both can be serialized to char* for
// message transporting and deserialized on the other end.
VertexRef get_vertex_ref_for_vertex(const Graph g, const Partition p, const Vertex v) {
    auto _g = static_cast<Graph_T*>(g);
    auto _v = static_cast<Vertex_T*>(v);
    auto gid = _g->Vertex2Gid(*_v);
    auto vr = new VertexRef(gid);
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
    return id_parser.GetFid(*_vr);
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

EdgeRef get_edge_ref_for_edge(const Graph g, const Partition p, const Edge e) {
    return NULL_EDGE_REF;
}

Edge get_edge_from_edge_ref(const Graph g, const EdgeRef e) {
    return NULL_EDGE;
}

Partition get_master_partition_from_edge_ref(const Graph g, const EdgeRef er) {
    return NULL_PARTITION;
}

char* serialize_edge_ref(const Graph g, const EdgeRef er) {
    return NULL;
}

EdgeRef deserialize_to_edge_ref(const Graph g, const char* msg) {
    return NULL_EDGE_REF;
}

// The concept of local_complete refers to whether we can get complete data or properties
// locally in the partition. It is orthogonal to the concept of master/mirror which
// is mainly designed for data aggregation. In some extremely cases, master vertices
// may NOT contain all the data or properties locally.
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

bool is_vertex_property_local_complete(const Graph g, const Vertex v) {
    return is_vertex_neighbor_local_complete(g, v);
}

PartitionList vertex_property_complete_partitions(const Graph g, const Vertex v) {
    return vertex_neighbor_complete_partitions(g, v);
}

bool is_edge_property_local_complete(const Graph g, const Edge e) {
    return true;
}

PartitionList edge_property_complete_partitions(const Graph g, const Edge e) {
    return NULL_LIST;
}
#endif
