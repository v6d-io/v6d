
# GRIN
## Assumptions
### Property Graph
- Vertices have types, so as Edges. 
- The relationship between edge type and pairs of vertex types is many-to-many.
- Properties are bound to vertex and edge types, but some may have the same name.
- Labels can be assigned to vertices and edges (NOT their types) primarily for query filtering, and labels have no properties.
## Design Principles
### Handler
- GRIN provides a series of handlers for graph concepts, such as vertex, edge and graph itself. 
- Since almost everything in GRIN are handlers except of only a few string names, the type for a graph concept and its handler is always mixed-used in GRIN.
- For example, we use the type Vertex to represent the type of a vertex handler, instead of using VertexHandler for clean code.

### List
- A list handler, no matter what kind of list it represents, is available to the user only if the storage can provide the size of the list, and an element retrieval API by position (i.e., index of array).
- For the instance of Vertex, when some GRIN API returns a `VertexList`(handler), the user can get the size of the `VertexList` by calling `get_vertex_list_size` to get the `size`, and `get_vertex_from_list` to get a `vertex` by providing an index value ranges from `0` to `size-1`.
### List Iterator
- A list iterator handler, no matter what kind of list it represents, is available to the user if the list size is unknown or for sequential scan efficiency. 
- Take Vertex as example again, users can get the iterator at the beginning using APIs like `get_vertex_list_begin`, and keeps on using `get_next_vertex_list_iter` to update the iterator till the end of the list when a `false` is returned. APIs like `get_vertex_from_iter` will return the `Vertex` from the vertex iterator.
### Property
- Properties are bound to vertex and edge types. It means even some properties may have the same name, as long as they are bound to different vertex or edge types, GRIN will provide distinct handlers for these properties. This is
because, although properties with the same name usually provide the same semantic in the graph, they may have 
different data types in the underlying storage for efficiency concerns (e.g., short date and long date).
- To avoid the incompatibility with storage engines, we made the design choice to bind properties under vertex and edge types. Meanwhile, GRIN provides an API to get all the property handlers with the (same) given property name.
### Label
- GRIN does NOT distinguish label on vertices and edges, that means a vertex and an edge may have a same label.
- However the storage can tell GRIN whether labels are enabled in vertices or edges seperatedly with macros of `WITH_VERTEX_LABEL` and `WITH_EDGE_LABEL` respectively.

### Reference
- GRIN introduces the reference concept in partitioned graph. It stands for the reference of an instance that can
be recognized in partitions other than the current partition where the instance is accessed.
- For example, a `VertexRef` is a reference of a `Vertex` that can be recognized in other partitions.

### Master and Mirror
- Master & mirror vertices are the concept borrowed from vertexcut partition strategy. When a vertex is recognized in
serveral partitions, we refer one of them as the master vertex while others as mirrors. This is primarily for data
aggregation purpose to share a common centural node for every one.
- While in edgecut partition, the concept becomes inner & outer vertices. GRIN uses `master == inner` and `mirror == outer` to unify these concepts.

### Local Complete
- The concept of local complete is with repect of whether a graph component adhere to a vertex or an edge is complete locally in the partition.
- Take vertex and properties as example. We say the vertex is "property local complete" if we can get all the properties of the vertex locally in the partition.
- We can also have "edge property local complete", "vertex neighbor local complete" and so on.
- We do NOT assume any local complete on master vertices. Since in some extremely cases, master vertices
may NOT contain all the data or properties locally.

## Traits
### Natural ID Trait
- Concepts represent the schema of the graph, such as vertex type and properties bound to a certain edge type, are usually numbered naturally from `0` to its `num - 1` in many storage engines. To facilitate further optimizations
in the upper computing engines, GRIN provides the natural number ID trait. A storage can provide such a trait if
it also uses the natural numbering for graph schema concepts.

