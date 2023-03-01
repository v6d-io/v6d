
# GRIN
GRIN is a series of C-style Graph Retrieval INterfaces for graph computing engines to access different
storage systems in a uniform way. 

## Assumptions
### Property Graph Model
- Vertices have types, so as Edges. 
- The relationship between edge type and pairs of vertex types is many-to-many.
- Properties are bound to vertex and edge types, but some may have the same name.
- Labels can be assigned to vertices and edges (NOT their types) primarily for query filtering, and labels have no properties.

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

## Traits
### Natural ID Trait
- Concepts represent the schema of the graph, such as vertex type and properties bound to a certain edge type, are usually numbered naturally from `0` to its `num - 1` in many storage engines. To facilitate further optimizations
in the upper computing engines, GRIN provides the natural number ID trait. A storage can provide such a trait if
it also uses the natural numbering for graph schema concepts.

