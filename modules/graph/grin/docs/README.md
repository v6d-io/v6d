
# GRIN
GRIN is a series of C-style Graph Retrieval INterfaces for graph computing engines to access different
storage systems in a uniform way. 

-----

## Assumptions
### Property Graph Model
- Vertices have types, so as Edges. 
- The relationship between edge type and pairs of vertex types is many-to-many.
- Properties are bound to vertex and edge types, but some may have the same name.
- Labels can be assigned to vertices and edges (NOT their types) primarily for query filtering, and labels have no properties.

### Partition Strategies
#### Edge-cut Partition Strategy
- Vertex data are local complete for master vertices
- Edge data are local complete for all edges
- Neighbors are local complete for master vertices
- Vertex properties are local complete for master vertices
- Edge properties are local complete for all edges

#### Vertex-cut Partition Strategy
- Vertex data are local complete for all vertices
- Edge data are local complete for all edges
- Mirror partition list is available for master vertices to broadcast messages
- Vertex properties are local complete for all vertices
- Edge properties are local complete for all edges

### Assumption Macros
- GRIN also provides granula assumption macros to describe storage assumptions.
- Some assumptions may dominate others, so storage providers should take care when setting these assumptions.
- Take assumptions on vertex property local complete as example, GRIN provides four macros:
    1. GRIN_ASSUME_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE
    2. GRIN_ASSUME_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
    3. GRIN_ASSUME_BY_TYPE_ALL_VERTEX_PROPERTY_LOCAL_COMPLETE
    4. GRIN_ASSUME_BY_TYPE_MASTER_VERTEX_PROPERTY_LOCAL_COMPLETE
- Here 1. dominates others, 2. dominates 4., and 3. also dominates 4., that means 2. to 4. are undefined when 1. is defined.
- Suppose only 3. is defined, it means vertices of certain types have all the properties locally complete, no matter the vertex is master or mirror. In this case, GRIN provides an API to return
these vertex types.

-----

## Design Principles
### Handler
- GRIN provides a series of handlers for graph concepts, such as vertex, edge and graph itself. 
- Since almost everything in GRIN are handlers except of only a few string names, the type for a graph concept and its handler is always mixed-used in GRIN.
- For example, GRIN uses the type Vertex to represent the type of a vertex handler, instead of using VertexHandler for clean code.

### List
- A list handler, no matter what kind of list it represents, is available to the user only if the storage can provide the size of the list, and an element retrieval API by position (i.e., index of array). Otherwise, the storage should provide a list iterator, see next section.
- A vertex list example

    ```CPP
        /* grin/topology/vertexlist.h */

        VertexList get_vertex_list(Graph g);  // get the vertexlist of a graph

        size_t get_vertex_list_size(VertexList vl);  // the storage must implement the API to return the size of vertexlist

        Vertex get_vertex_from_list(VertexList vl, size_t idx);  // the storage must implement the API to return the element of vertexlist by position


        /* run.cc */
        {
            auto vertexlist = get_vertex_list(g); // with a graph (handler) g
            auto sz = get_vertex_list_size(vertexlist);

            for (auto i = 0; i < sz; ++i) {
                auto v = get_vertex_from_list(vertexlist, i);
            }
        }
    ```

### List Iterator
- A list iterator handler, no matter what kind of list it represents, is available to the user if the list size is unknown or for sequential scan efficiency. 
- A vertex list iterator example

    ```CPP
        /* grin/topology/vertexlist.h */

        VertexListIterator get_vertex_list_begin(Graph g);  // get the begin iterator of the vertexlist

        VertexListIterator get_next_vertex_list_iter(VertexListIterator);  // get next iterator

        bool is_vertex_list_end(VertexListIterator); // check if reaches the end

        Vertex get_vertex_from_iter(VertexListIterator vli); // get the vertex from the iterator


        /* run.cc */
        {
            auto iter = get_vertex_list_begin(g); // with a graph (handler) g

            while (!is_vertex_list_end(iter)) {
                auto v = get_vertex_from_iter(iter);
                iter = get_next_vertex_list_iter(iter);
            }
        }
    ```

### Property
- Properties are bound to vertex and edge types. It means even some properties may have the same name, as long as they are bound to different vertex or edge types, GRIN will provide distinct handlers for these properties. This is
because, although properties with the same name usually provide the same semantic in the graph, they may have 
different data types in the underlying storage for efficiency concerns (e.g., short date and long date).
- To avoid the incompatibility with storage engines, we made the design choice to bind properties under vertex and edge types. Meanwhile, GRIN provides an API to get all the property handlers with the (same) given property name.

    ```CPP
        /* grin/property/type.h */

        VertexType get_vertex_type_by_name(Graph g, const char* name);


        /* grin/property/property.h */

        VertexProperty get_vertex_property_by_name(Graph, VertexType, const char* name);

        VertexPropertyList get_vertex_properties_by_name(Graph, const char* name);


        /* run.cc */
        {
            auto vtype = get_vertex_type_by_name(g, "Person");  // get the vertex type of Person
            auto vprop = get_vertex_property_by_name(g, vtype, "Name");  // get the Name property bound to Person

            auto vpl = get_vertex_properties_by_name(g, "Name");  // get all the properties called Name under all the vertex types (e.g., Person, Company) in g
        }
    ```

### Label
- GRIN does NOT distinguish label on vertices and edges, that means a vertex and an edge may have a same label.
- However the storage can tell GRIN whether labels are enabled in vertices or edges seperatedly with macros of `WITH_VERTEX_LABEL` and `WITH_EDGE_LABEL` respectively.

### Reference
- GRIN introduces the reference concept in partitioned graph. It stands for the reference of an instance that can
be recognized in partitions other than the current partition where the instance is accessed.
- For example, a `VertexRef` is a reference of a `Vertex` that can be recognized in other partitions.

    ```CPP
        /* grin/partition/partition.h */
        
        VertexRef get_vertex_ref_for_vertex(Graph, Partition, Vertex);
        
        const char* serialize_vertex_ref(Graph, VertexRef);

        VertexRef deserialize_to_vertex_ref(Graph, const char*);

        Vertex get_vertex_from_vertex_ref(Graph, VertexRef);


        /* run.cc in machine 1 */
        {
            // p is the partition (handler) for the partition in machine 2

            auto vref = get_vertex_ref_for_vertex(g, p, v);  // get v's vertex ref which can be recgonized in machine 2

            const char* msg = serialize_vertex_ref(g, vref);  // serialize into a message

            // send the message to machine 2...
        }


        /* run.cc in machine 2 */
        {
            // recieve the message from machine 1...

            auto vref = deserialize_to_vertex_ref(g, msg);  // deserialize back to vertex ref

            auto v = get_vertex_from_vertex_ref(g, vref);  // cast to vertex if g can recognize the vertex ref
        }
    ```

### Master and Mirror
- Master & mirror vertices are the concept borrowed from vertexcut partition strategy. When a vertex is recognized in
serveral partitions, GRIN refers one of them as the master vertex while others as mirrors. This is primarily for data
aggregation purpose to share a common centural node for every one.
- While in edgecut partition, the concept becomes inner & outer vertices. GRIN uses master & mirror vertices to represent inner & outer vertices respectively to unify these concepts.

### Local Complete
- The concept of local complete is with repect to whether a graph component adhere to a vertex or an edge is locally complete within the partition.
- Take vertex and properties as example. GRIN considers the vertex is "property local complete" if it can get all the properties of the vertex locally in the partition.
- There are concepts like "edge property local complete", "vertex neighbor local complete" and so on.
- GRIN does NOT assume any local complete on master vertices. Since in some extremely cases, master vertices
may NOT contain all the data or properties locally.
- GRIN currently provides vertex-level/edge-level local complete judgement APIs, while the introduction of type-level judgement APIs is open for discussion.

-----

## Traits
### Natural ID Trait
- Concepts represent the schema of the graph, such as vertex type and properties bound to a certain edge type, are usually numbered naturally from `0` to its `num - 1` in many storage engines. To facilitate further optimizations
in the upper computing engines, GRIN provides the natural number ID trait. A storage can provide such a trait if
it also uses the natural numbering for graph schema concepts.

-----

## Examples
### Example A
A mixed example with structure, partition and property

```CPP
    void sync_property(void* partitioned_graph, void* partition, const char* edge_type_name, const char* vertex_property_name) {
    /*
        This example illustrates how to sync property values of vertices related to certain edge type.
        
        The input parameters are the partitioned_graph, the local partition,
        the edge_type_name (e.g., likes), the vertex_property_name (e.g., features)

        The task is to find all the destination vertices of "boundary edges" with type named "likes", and the vertices
        must have a property named "features". Here a boundary edge is an edge whose source vertex is a master vertex and
        the destination is a mirror vertex, given the context of "edge-cut" partition strategy that the underlying storage uses.
        Then for each of these vertices, we send the value of the "features" property to its master partition.
    */
        GRIN_GRAPH g = grin_get_local_graph_from_partition(partitioned_graph, partition);  // get local graph of partition

        GRIN_EDGE_TYPE etype = grin_get_edge_type_by_name(g, edge_type_name);  // get edge type from name
        GRIN_VERTEX_TYPE_LIST src_vtypes = grin_get_src_types_from_edge_type(g, etype);  // get related source vertex type list
        GRIN_VERTEX_TYPE_LIST dst_vtypes = grin_get_dst_types_from_edge_type(g, etype);  // get related destination vertex type list

        size_t src_vtypes_num = grin_get_vertex_type_list_size(g, src_vtypes);
        size_t dst_vtypes_num = grin_get_vertex_type_list_size(g, dst_vtypes);
        assert(src_vtypes_num == dst_vtypes_num);  // the src & dst vertex type lists must be aligned

        for (size_t i = 0; i < src_vtypes_num; ++i) {  // iterate all pairs of src & dst vertex type
            GRIN_VERTEX_TYPE src_vtype = grin_get_vertex_type_from_list(g, src_vtypes, i);  // get src type
            GRIN_VERTEX_TYPE dst_vtype = grin_get_vertex_type_from_list(g, dst_vtypes, i);  // get dst type

            GRIN_VERTEX_PROPERTY dst_vp = grin_get_vertex_property_by_name(g, dst_vtype, vertex_property_name);  // get the property called "features" under dst type
            if (dst_vp == GRIN_NULL_VERTEX_PROPERTY) continue;  // filter out the pairs whose dst type does NOT have such a property called "features"
            
            GRIN_VERTEX_PROPERTY_TABLE dst_vpt = grin_get_vertex_property_table_by_type(g, dst_vtype);  // prepare property table of dst vertex type for later use
            GRIN_DATATYPE dst_vp_dt = grin_get_vertex_property_data_type(g, dst_vp); // prepare property type for later use

            GRIN_VERTEX_LIST __src_vl = grin_get_vertex_list(g);  // get the vertex list
            GRIN_VERTEX_LIST _src_vl = grin_filter_type_for_vertex_list(g, src_vtype, __src_vl);  // filter the vertex of source type
            GRIN_VERTEX_LIST src_vl = grin_filter_master_for_vertex_list(g, _src_vl);  // filter master vertices under source type
            
            size_t src_vl_num = grin_get_vertex_list_size(g, src_vl);
            for (size_t j = 0; j < src_vl_num; ++j) { // iterate the src vertex
                GRIN_VERTEX v = grin_get_vertex_from_list(g, src_vl, j);

            #ifdef GRIN_TRAIT_FILTER_EDGE_TYPE_FOR_ADJACENT_LIST
                GRIN_ADJACENT_LIST _adj_list = grin_get_adjacent_list(g, GRIN_DIRECTION::OUT, v);  // get the outgoing adjacent list of v
                GRIN_ADJACENT_LIST adj_list = grin_filter_edge_type_for_adjacent_list(g, etype, _adj_list);  // filter edges under etype
            #else
                GRIN_ADJACENT_LIST adj_lsit = grin_get_adjacent_list(g, GRIN_DIRECTION::OUT, v);  // get the outgoing adjacent list of v
            #endif

                size_t al_sz = grin_get_adjacent_list_size(g, adj_list);
                for (size_t k = 0; k < al_sz; ++k) {
            #ifndef GRIN_TRAIT_FILTER_EDGE_TYPE_FOR_ADJACENT_LIST
                    GRIN_EDGE edge = grin_get_edge_from_adjacent_list(g, adj_list, k);
                    GRIN_EDGE_TYPE edge_type = grin_get_edge_type(g, edge);
                    if (!grin_equal_edge_type(g, edge_type, etype)) continue;
            #endif
                    GRIN_VERTEX u = grin_get_neighbor_from_adjacent_list(g, adj_list, k);  // get the dst vertex u
                    const void* value = grin_get_value_from_vertex_property_table(g, dst_vpt, u, dst_vp);  // get the property value of "features" of u

                    GRIN_VERTEX_REF uref = grin_get_vertex_ref_for_vertex(g, u);  // get the reference of u that can be recoginized by other partitions
                    GRIN_PARTITION u_master_partition = grin_get_master_partition_from_vertex_ref(g, uref);  // get the master partition for u

                    send_value(u_master_partition, uref, dst_vp_dt, value);  // the value must be casted to the correct type based on dst_vp_dt before sending
                }
            }
        }
    }
    
    void run(vineyard::Client& client, const grape::CommSpec& comm_spec,
                vineyard::ObjectID fragment_group_id) {
        LOG(INFO) << "Loaded graph to vineyard: " << fragment_group_id;

        auto pg = get_partitioned_graph_by_object_id(client, fragment_group_id);
        auto local_partitions = get_local_partition_list(pg);
        size_t pnum = get_partition_list_size(local_partitions);
        assert(pnum > 0);

        // we only sync the first partition as example
        auto partition = get_partition_from_list(local_partitions, 0);
        sync_property(pg, partition, "likes", "features");
    }
```

