#[doc = "< incoming"]
pub const GRIN_DIRECTION_IN: GrinDirection = 0;
#[doc = "< outgoing"]
pub const GRIN_DIRECTION_OUT: GrinDirection = 1;
#[doc = "< incoming & outgoing"]
pub const GRIN_DIRECTION_BOTH: GrinDirection = 2;
#[doc = " Enumerates the directions of edges with respect to a certain vertex"]
pub type GrinDirection = ::std::os::raw::c_uint;
#[doc = "< other unknown types"]
pub const GRIN_DATATYPE_UNDEFINED: GrinDatatype = 0;
#[doc = "< int"]
pub const GRIN_DATATYPE_INT32: GrinDatatype = 1;
#[doc = "< unsigned int"]
pub const GRIN_DATATYPE_UINT32: GrinDatatype = 2;
#[doc = "< long int"]
pub const GRIN_DATATYPE_INT64: GrinDatatype = 3;
#[doc = "< unsigned long int"]
pub const GRIN_DATATYPE_UINT64: GrinDatatype = 4;
#[doc = "< float"]
pub const GRIN_DATATYPE_FLOAT: GrinDatatype = 5;
#[doc = "< double"]
pub const GRIN_DATATYPE_DOUBLE: GrinDatatype = 6;
#[doc = "< string"]
pub const GRIN_DATATYPE_STRING: GrinDatatype = 7;
#[doc = "< short date"]
pub const GRIN_DATATYPE_DATE32: GrinDatatype = 8;
#[doc = "< long date"]
pub const GRIN_DATATYPE_DATE64: GrinDatatype = 9;
#[doc = " Enumerates the datatype supported in the storage"]
pub type GrinDatatype = ::std::os::raw::c_uint;
#[doc = "@}"]
pub type GrinGraph = *mut ::std::os::raw::c_void;
pub type GrinVertex = *mut ::std::os::raw::c_void;
pub type GrinEdge = *mut ::std::os::raw::c_void;
pub type GrinVertexOriginalId = *mut ::std::os::raw::c_void;
pub type GrinVertexList = *mut ::std::os::raw::c_void;
pub type GrinVertexListIterator = *mut ::std::os::raw::c_void;
pub type GrinAdjacentList = *mut ::std::os::raw::c_void;
pub type GrinAdjacentListIterator = *mut ::std::os::raw::c_void;
pub type GrinPartitionedGraph = *mut ::std::os::raw::c_void;
pub type GrinPartition = *mut ::std::os::raw::c_void;
pub type GrinPartitionList = *mut ::std::os::raw::c_void;
pub type GrinPartitionId = ::std::os::raw::c_uint;
pub type GrinVertexRef = *mut ::std::os::raw::c_void;
pub type GrinVertexType = *mut ::std::os::raw::c_void;
pub type GrinVertexTypeList = *mut ::std::os::raw::c_void;
pub type GrinVertexProperty = *mut ::std::os::raw::c_void;
pub type GrinVertexPropertyList = *mut ::std::os::raw::c_void;
pub type GrinVertexPropertyTable = *mut ::std::os::raw::c_void;
pub type GrinVertexTypeId = ::std::os::raw::c_uint;
pub type GrinVertexPropertyId = ::std::os::raw::c_uint;
pub type GrinEdgeType = *mut ::std::os::raw::c_void;
pub type GrinEdgeTypeList = *mut ::std::os::raw::c_void;
pub type GrinVevType = *mut ::std::os::raw::c_void;
pub type GrinVevTypeList = *mut ::std::os::raw::c_void;
pub type GrinEdgeProperty = *mut ::std::os::raw::c_void;
pub type GrinEdgePropertyList = *mut ::std::os::raw::c_void;
pub type GrinEdgePropertyTable = *mut ::std::os::raw::c_void;
pub type GrinEdgeTypeId = ::std::os::raw::c_uint;
pub type GrinEdgePropertyId = ::std::os::raw::c_uint;
pub type GrinRow = *mut ::std::os::raw::c_void;
extern "C" {
    #[cfg(feature = "grin_enable_adjacent_list")]
    pub fn grin_get_adjacent_list(
        arg1: GrinGraph,
        arg2: GrinDirection,
        arg3: GrinVertex,
    ) -> GrinAdjacentList;

    #[cfg(feature = "grin_enable_adjacent_list")]
    pub fn grin_destroy_adjacent_list(arg1: GrinGraph, arg2: GrinAdjacentList);

    #[cfg(feature = "grin_enable_adjacent_list_array")]
    pub fn grin_get_adjacent_list_size(arg1: GrinGraph, arg2: GrinAdjacentList) -> usize;

    #[cfg(feature = "grin_enable_adjacent_list_array")]
    pub fn grin_get_neighbor_from_adjacent_list(
        arg1: GrinGraph,
        arg2: GrinAdjacentList,
        arg3: usize,
    ) -> GrinVertex;

    #[cfg(feature = "grin_enable_adjacent_list_array")]
    pub fn grin_get_edge_from_adjacent_list(
        arg1: GrinGraph,
        arg2: GrinAdjacentList,
        arg3: usize,
    ) -> GrinEdge;

    #[cfg(feature = "grin_enable_adjacent_list_iterator")]
    pub fn grin_get_adjacent_list_begin(
        arg1: GrinGraph,
        arg2: GrinAdjacentList,
    ) -> GrinAdjacentListIterator;

    #[cfg(feature = "grin_enable_adjacent_list_iterator")]
    pub fn grin_destroy_adjacent_list_iter(arg1: GrinGraph, arg2: GrinAdjacentListIterator);

    #[cfg(feature = "grin_enable_adjacent_list_iterator")]
    pub fn grin_get_next_adjacent_list_iter(arg1: GrinGraph, arg2: GrinAdjacentListIterator);

    #[cfg(feature = "grin_enable_adjacent_list_iterator")]
    pub fn grin_is_adjacent_list_end(arg1: GrinGraph, arg2: GrinAdjacentListIterator) -> bool;

    #[cfg(feature = "grin_enable_adjacent_list_iterator")]
    pub fn grin_get_neighbor_from_adjacent_list_iter(
        arg1: GrinGraph,
        arg2: GrinAdjacentListIterator,
    ) -> GrinVertex;

    #[cfg(feature = "grin_enable_adjacent_list_iterator")]
    pub fn grin_get_edge_from_adjacent_list_iter(
        arg1: GrinGraph,
        arg2: GrinAdjacentListIterator,
    ) -> GrinEdge;

    #[cfg(all(feature = "grin_assume_has_directed_graph", feature = "grin_assume_has_undirected_graph"))]
    pub fn grin_get_graph_from_storage(
        arg1: ::std::os::raw::c_int,
        arg2: *mut *mut ::std::os::raw::c_char,
    ) -> GrinGraph;

    #[cfg(all(feature = "grin_assume_has_directed_graph", feature = "grin_assume_has_undirected_graph"))]
    pub fn grin_destroy_graph(arg1: GrinGraph);

    #[cfg(all(feature = "grin_assume_has_directed_graph", feature = "grin_assume_has_undirected_graph"))]
    pub fn grin_is_directed(arg1: GrinGraph) -> bool;

    #[cfg(feature = "grin_assume_has_multi_edge_graph")]
    pub fn grin_is_multigraph(arg1: GrinGraph) -> bool;

    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_get_vertex_num(arg1: GrinGraph) -> usize;

    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_get_edge_num(arg1: GrinGraph) -> usize;

    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_destroy_vertex(arg1: GrinGraph, arg2: GrinVertex);

    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_equal_vertex(arg1: GrinGraph, arg2: GrinVertex, arg3: GrinVertex) -> bool;

    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_destroy_vertex_original_id(arg1: GrinGraph, arg2: GrinVertexOriginalId);

    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_get_vertex_original_id_type(arg1: GrinGraph) -> GrinDatatype;

    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_get_vertex_original_id(
        arg1: GrinGraph,
        arg2: GrinVertex,
    ) -> GrinVertexOriginalId;

    #[cfg(feature = "grin_with_vertex_data")]
    pub fn grin_destroy_value(
        arg1: GrinGraph,
        arg2: GrinDatatype,
        arg3: *const ::std::os::raw::c_void,
    );

    #[cfg(feature = "grin_with_vertex_data")]
    pub fn grin_destroy_name(arg1: GrinGraph, arg2: *const ::std::os::raw::c_char);

    #[cfg(feature = "grin_with_edge_data")]
    pub fn grin_destroy_edge(arg1: GrinGraph, arg2: GrinEdge);

    #[cfg(feature = "grin_with_edge_data")]
    pub fn grin_get_edge_src(arg1: GrinGraph, arg2: GrinEdge) -> GrinVertex;

    #[cfg(feature = "grin_with_edge_data")]
    pub fn grin_get_edge_dst(arg1: GrinGraph, arg2: GrinEdge) -> GrinVertex;

    #[cfg(feature = "grin_enable_vertex_list")]
    pub fn grin_get_vertex_list(arg1: GrinGraph) -> GrinVertexList;

    #[cfg(feature = "grin_enable_vertex_list")]
    pub fn grin_destroy_vertex_list(arg1: GrinGraph, arg2: GrinVertexList);

    #[cfg(feature = "grin_enable_vertex_list_array")]
    pub fn grin_get_vertex_list_size(arg1: GrinGraph, arg2: GrinVertexList) -> usize;

    #[cfg(feature = "grin_enable_vertex_list_array")]
    pub fn grin_get_vertex_from_list(
        arg1: GrinGraph,
        arg2: GrinVertexList,
        arg3: usize,
    ) -> GrinVertex;

    #[cfg(feature = "grin_enable_vertex_list_iterator")]
    pub fn grin_get_vertex_list_begin(
        arg1: GrinGraph,
        arg2: GrinVertexList,
    ) -> GrinVertexListIterator;

    #[cfg(feature = "grin_enable_vertex_list_iterator")]
    pub fn grin_destroy_vertex_list_iter(arg1: GrinGraph, arg2: GrinVertexListIterator);

    #[cfg(feature = "grin_enable_vertex_list_iterator")]
    pub fn grin_get_next_vertex_list_iter(arg1: GrinGraph, arg2: GrinVertexListIterator);

    #[cfg(feature = "grin_enable_vertex_list_iterator")]
    pub fn grin_is_vertex_list_end(arg1: GrinGraph, arg2: GrinVertexListIterator) -> bool;

    #[cfg(feature = "grin_enable_vertex_list_iterator")]
    pub fn grin_get_vertex_from_iter(
        arg1: GrinGraph,
        arg2: GrinVertexListIterator,
    ) -> GrinVertex;

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_partitioned_graph_from_storage(
        arg1: ::std::os::raw::c_int,
        arg2: *mut *mut ::std::os::raw::c_char,
    ) -> GrinPartitionedGraph;

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_destroy_partitioned_graph(arg1: GrinPartitionedGraph);

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_total_partitions_number(arg1: GrinPartitionedGraph) -> usize;

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_local_partition_list(arg1: GrinPartitionedGraph) -> GrinPartitionList;

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_destroy_partition_list(arg1: GrinPartitionedGraph, arg2: GrinPartitionList);

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_create_partition_list(arg1: GrinPartitionedGraph) -> GrinPartitionList;

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_insert_partition_to_list(
        arg1: GrinPartitionedGraph,
        arg2: GrinPartitionList,
        arg3: GrinPartition,
    ) -> bool;

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_partition_list_size(
        arg1: GrinPartitionedGraph,
        arg2: GrinPartitionList,
    ) -> usize;

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_partition_from_list(
        arg1: GrinPartitionedGraph,
        arg2: GrinPartitionList,
        arg3: usize,
    ) -> GrinPartition;

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_equal_partition(
        arg1: GrinPartitionedGraph,
        arg2: GrinPartition,
        arg3: GrinPartition,
    ) -> bool;

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_destroy_partition(arg1: GrinPartitionedGraph, arg2: GrinPartition);

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_partition_info(
        arg1: GrinPartitionedGraph,
        arg2: GrinPartition,
    ) -> *const ::std::os::raw::c_void;

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_local_graph_from_partition(
        arg1: GrinPartitionedGraph,
        arg2: GrinPartition,
    ) -> GrinGraph;

    #[cfg(feature = "grin_trait_natural_id_for_partition")]
    pub fn grin_get_partition_from_id(
        arg1: GrinPartitionedGraph,
        arg2: GrinPartitionId,
    ) -> GrinPartition;

    #[cfg(feature = "grin_trait_natural_id_for_partition")]
    pub fn grin_get_partition_id(
        arg1: GrinPartitionedGraph,
        arg2: GrinPartition,
    ) -> GrinPartitionId;

    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_get_vertex_ref_for_vertex(arg1: GrinGraph, arg2: GrinVertex) -> GrinVertexRef;

    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_destroy_vertex_ref(arg1: GrinGraph, arg2: GrinVertexRef);

    #[doc = " @brief get the local vertex from the vertex ref\n if the vertex ref is not regconized, a null vertex is returned\n @param GrinGraph the graph\n @param GrinVertexRef the vertex ref"]
    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_get_vertex_from_vertex_ref(arg1: GrinGraph, arg2: GrinVertexRef) -> GrinVertex;

    #[doc = " @brief get the master partition of a vertex ref.\n Some storage can still provide the master partition of the vertex ref,\n even if the vertex ref can NOT be recognized locally.\n @param GrinGraph the graph\n @param GrinVertexRef the vertex ref"]
    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_get_master_partition_from_vertex_ref(
        arg1: GrinGraph,
        arg2: GrinVertexRef,
    ) -> GrinPartition;

    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_serialize_vertex_ref(
        arg1: GrinGraph,
        arg2: GrinVertexRef,
    ) -> *const ::std::os::raw::c_char;

    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_destroy_serialized_vertex_ref(
        arg1: GrinGraph,
        arg2: *const ::std::os::raw::c_char,
    );

    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_deserialize_to_vertex_ref(
        arg1: GrinGraph,
        arg2: *const ::std::os::raw::c_char,
    ) -> GrinVertexRef;

    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_is_master_vertex(arg1: GrinGraph, arg2: GrinVertex) -> bool;

    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_is_mirror_vertex(arg1: GrinGraph, arg2: GrinVertex) -> bool;

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_total_vertex_num(arg1: GrinPartitionedGraph) -> usize;

    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_total_edge_num(arg1: GrinPartitionedGraph) -> usize;

    #[cfg(feature = "grin_trait_select_master_for_vertex_list")]
    pub fn grin_select_master_for_vertex_list(
        arg1: GrinGraph,
        arg2: GrinVertexList,
    ) -> GrinVertexList;

    #[cfg(feature = "grin_trait_select_master_for_vertex_list")]
    pub fn grin_select_mirror_for_vertex_list(
        arg1: GrinGraph,
        arg2: GrinVertexList,
    ) -> GrinVertexList;

    #[doc = " @brief get the vertex property name\n @param GrinGraph the graph\n @param GrinVertexProperty the vertex property"]
    #[cfg(feature = "grin_with_vertex_property_name")]
    pub fn grin_get_vertex_property_name(
        arg1: GrinGraph,
        arg2: GrinVertexProperty,
    ) -> *const ::std::os::raw::c_char;

    #[doc = " @brief get the vertex property with a given name under a specific vertex type\n @param GrinGraph the graph\n @param GrinVertexType the specific vertex type\n @param name the name"]
    #[cfg(feature = "grin_with_vertex_property_name")]
    pub fn grin_get_vertex_property_by_name(
        arg1: GrinGraph,
        arg2: GrinVertexType,
        name: *const ::std::os::raw::c_char,
    ) -> GrinVertexProperty;

    #[doc = " @brief get all the vertex properties with a given name\n @param GrinGraph the graph\n @param name the name"]
    #[cfg(feature = "grin_with_vertex_property_name")]
    pub fn grin_get_vertex_properties_by_name(
        arg1: GrinGraph,
        name: *const ::std::os::raw::c_char,
    ) -> GrinVertexPropertyList;

    #[doc = " @brief get the edge property name\n @param GrinGraph the graph\n @param GrinEdgeProperty the edge property"]
    #[cfg(feature = "grin_with_edge_property_name")]
    pub fn grin_get_edge_property_name(
        arg1: GrinGraph,
        arg2: GrinEdgeProperty,
    ) -> *const ::std::os::raw::c_char;

    #[doc = " @brief get the edge property with a given name under a specific edge type\n @param GrinGraph the graph\n @param GrinEdgeType the specific edge type\n @param name the name"]
    #[cfg(feature = "grin_with_edge_property_name")]
    pub fn grin_get_edge_property_by_name(
        arg1: GrinGraph,
        arg2: GrinEdgeType,
        name: *const ::std::os::raw::c_char,
    ) -> GrinEdgeProperty;

    #[doc = " @brief get all the edge properties with a given name\n @param GrinGraph the graph\n @param name the name"]
    #[cfg(feature = "grin_with_edge_property_name")]
    pub fn grin_get_edge_properties_by_name(
        arg1: GrinGraph,
        name: *const ::std::os::raw::c_char,
    ) -> GrinEdgePropertyList;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_equal_vertex_property(
        arg1: GrinGraph,
        arg2: GrinVertexProperty,
        arg3: GrinVertexProperty,
    ) -> bool;

    #[doc = " @brief destroy vertex property\n @param GrinVertexProperty vertex property"]
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_destroy_vertex_property(arg1: GrinGraph, arg2: GrinVertexProperty);

    #[doc = " @brief get property data type\n @param GrinVertexProperty vertex property"]
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_property_data_type(
        arg1: GrinGraph,
        arg2: GrinVertexProperty,
    ) -> GrinDatatype;

    #[doc = " @brief get the vertex type that the property is bound to\n @param GrinVertexProperty vertex property"]
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_property_vertex_type(
        arg1: GrinGraph,
        arg2: GrinVertexProperty,
    ) -> GrinVertexType;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_equal_edge_property(
        arg1: GrinGraph,
        arg2: GrinEdgeProperty,
        arg3: GrinEdgeProperty,
    ) -> bool;

    #[doc = " @brief destroy edge property\n @param GrinEdgeProperty edge property"]
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_destroy_edge_property(arg1: GrinGraph, arg2: GrinEdgeProperty);

    #[doc = " @brief get property data type\n @param GrinEdgeProperty edge property"]
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_property_data_type(
        arg1: GrinGraph,
        arg2: GrinEdgeProperty,
    ) -> GrinDatatype;

    #[doc = " @brief get the edge type that the property is bound to\n @param GrinEdgeProperty edge property"]
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_property_edge_type(
        arg1: GrinGraph,
        arg2: GrinEdgeProperty,
    ) -> GrinEdgeType;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_property_list_by_type(
        arg1: GrinGraph,
        arg2: GrinVertexType,
    ) -> GrinVertexPropertyList;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_property_list_size(
        arg1: GrinGraph,
        arg2: GrinVertexPropertyList,
    ) -> usize;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_property_from_list(
        arg1: GrinGraph,
        arg2: GrinVertexPropertyList,
        arg3: usize,
    ) -> GrinVertexProperty;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_create_vertex_property_list(arg1: GrinGraph) -> GrinVertexPropertyList;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_destroy_vertex_property_list(arg1: GrinGraph, arg2: GrinVertexPropertyList);

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_insert_vertex_property_to_list(
        arg1: GrinGraph,
        arg2: GrinVertexPropertyList,
        arg3: GrinVertexProperty,
    ) -> bool;

    #[cfg(feature = "grin_trait_natural_id_for_vertex_property")]
    pub fn grin_get_vertex_property_from_id(
        arg1: GrinGraph,
        arg2: GrinVertexType,
        arg3: GrinVertexPropertyId,
    ) -> GrinVertexProperty;

    #[cfg(feature = "grin_trait_natural_id_for_vertex_property")]
    pub fn grin_get_vertex_property_id(
        arg1: GrinGraph,
        arg2: GrinVertexType,
        arg3: GrinVertexProperty,
    ) -> GrinVertexPropertyId;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_property_list_by_type(
        arg1: GrinGraph,
        arg2: GrinEdgeType,
    ) -> GrinEdgePropertyList;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_property_list_size(
        arg1: GrinGraph,
        arg2: GrinEdgePropertyList,
    ) -> usize;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_property_from_list(
        arg1: GrinGraph,
        arg2: GrinEdgePropertyList,
        arg3: usize,
    ) -> GrinEdgeProperty;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_create_edge_property_list(arg1: GrinGraph) -> GrinEdgePropertyList;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_destroy_edge_property_list(arg1: GrinGraph, arg2: GrinEdgePropertyList);

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_insert_edge_property_to_list(
        arg1: GrinGraph,
        arg2: GrinEdgePropertyList,
        arg3: GrinEdgeProperty,
    ) -> bool;

    #[cfg(feature = "grin_trait_natural_id_for_edge_property")]
    pub fn grin_get_edge_property_from_id(
        arg1: GrinGraph,
        arg2: GrinEdgeType,
        arg3: GrinEdgePropertyId,
    ) -> GrinEdgeProperty;

    #[cfg(feature = "grin_trait_natural_id_for_edge_property")]
    pub fn grin_get_edge_property_id(
        arg1: GrinGraph,
        arg2: GrinEdgeType,
        arg3: GrinEdgeProperty,
    ) -> GrinEdgePropertyId;

    #[cfg(feature = "grin_enable_row")]
    pub fn grin_destroy_row(arg1: GrinGraph, arg2: GrinRow);

    #[doc = " @brief the value of a property from row by its position in row"]
    #[cfg(feature = "grin_enable_row")]
    pub fn grin_get_value_from_row(
        arg1: GrinGraph,
        arg2: GrinRow,
        arg3: GrinDatatype,
        arg4: usize,
    ) -> *const ::std::os::raw::c_void;

    #[doc = " @brief create a row, usually to get vertex/edge by primary keys"]
    #[cfg(feature = "grin_enable_row")]
    pub fn grin_create_row(arg1: GrinGraph) -> GrinRow;

    #[doc = " @brief insert a value to the end of the row"]
    #[cfg(feature = "grin_enable_row")]
    pub fn grin_insert_value_to_row(
        arg1: GrinGraph,
        arg2: GrinRow,
        arg3: GrinDatatype,
        arg4: *const ::std::os::raw::c_void,
    ) -> bool;

    #[doc = " @brief destroy vertex property table\n @param GrinVertexPropertyTable vertex property table"]
    #[cfg(feature = "grin_enable_vertex_property_table")]
    pub fn grin_destroy_vertex_property_table(arg1: GrinGraph, arg2: GrinVertexPropertyTable);

    #[doc = " @brief get the vertex property table of a certain vertex type\n No matter column or row store strategy is used in the storage,\n GRIN recommends to first get the property table of the vertex type,\n and then fetch values(rows) by vertex and property(list). However,\n GRIN does provide direct row fetching API when GrinAssumeColumnStoreForVertexProperty\n is NOT set.\n @param GrinGraph the graph\n @param GrinVertexType the vertex type"]
    #[cfg(feature = "grin_enable_vertex_property_table")]
    pub fn grin_get_vertex_property_table_by_type(
        arg1: GrinGraph,
        arg2: GrinVertexType,
    ) -> GrinVertexPropertyTable;

    #[doc = " @brief get vertex property value from table\n @param GrinVertexPropertyTable vertex property table\n @param GrinVertex the vertex which is the row index\n @param GrinVertexProperty the vertex property which is the column index\n @return can be casted to the property data type by the caller"]
    #[cfg(feature = "grin_enable_vertex_property_table")]
    pub fn grin_get_value_from_vertex_property_table(
        arg1: GrinGraph,
        arg2: GrinVertexPropertyTable,
        arg3: GrinVertex,
        arg4: GrinVertexProperty,
    ) -> *const ::std::os::raw::c_void;

    #[doc = " @brief get vertex row from table\n @param GrinVertexPropertyTable vertex property table\n @param GrinVertex the vertex which is the row index\n @param GrinVertexPropertyList the vertex property list as columns"]
    #[cfg(all(feature = "grin_enable_vertex_property_table", feature = "grin_enable_row"))]
    pub fn grin_get_row_from_vertex_property_table(
        arg1: GrinGraph,
        arg2: GrinVertexPropertyTable,
        arg3: GrinVertex,
        arg4: GrinVertexPropertyList,
    ) -> GrinRow;

    #[doc = " @brief destroy edge property table\n @param GrinEdgePropertyTable edge property table"]
    #[cfg(feature = "grin_enable_edge_property_table")]
    pub fn grin_destroy_edge_property_table(arg1: GrinGraph, arg2: GrinEdgePropertyTable);

    #[doc = " @brief get the edge property table of a certain edge type\n No matter column or row store strategy is used in the storage,\n GRIN recommends to first get the property table of the edge type,\n and then fetch values(rows) by edge and property(list). However,\n GRIN does provide direct row fetching API when GrinAssumeColumnStoreForEdgeProperty\n is NOT set.\n @param GrinGraph the graph\n @param GrinEdgeType the edge type"]
    #[cfg(feature = "grin_enable_edge_property_table")]
    pub fn grin_get_edge_property_table_by_type(
        arg1: GrinGraph,
        arg2: GrinEdgeType,
    ) -> GrinEdgePropertyTable;

    #[doc = " @brief get edge property value from table\n @param GrinEdgePropertyTable edge property table\n @param GrinEdge the edge which is the row index\n @param GrinEdgeProperty the edge property which is the column index\n @return can be casted to the property data type by the caller"]
    #[cfg(feature = "grin_enable_edge_property_table")]
    pub fn grin_get_value_from_edge_property_table(
        arg1: GrinGraph,
        arg2: GrinEdgePropertyTable,
        arg3: GrinEdge,
        arg4: GrinEdgeProperty,
    ) -> *const ::std::os::raw::c_void;

    #[doc = " @brief get edge row from table\n @param GrinEdgePropertyTable edge property table\n @param GrinEdge the edge which is the row index\n @param GrinEdgePropertyList the edge property list as columns"]
    #[cfg(all(feature = "grin_enable_edge_property_table", feature = "grin_enable_row"))]
    pub fn grin_get_row_from_edge_property_table(
        arg1: GrinGraph,
        arg2: GrinEdgePropertyTable,
        arg3: GrinEdge,
        arg4: GrinEdgePropertyList,
    ) -> GrinRow;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_num_by_type(arg1: GrinGraph, arg2: GrinVertexType) -> usize;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_num_by_type(arg1: GrinGraph, arg2: GrinEdgeType) -> usize;

    #[cfg(all(feature = "grin_enable_graph_partition", feature = "grin_with_vertex_property"))]
    pub fn grin_get_total_vertex_num_by_type(
        arg1: GrinPartitionedGraph,
        arg2: GrinVertexType,
    ) -> usize;

    #[cfg(all(feature = "grin_enable_graph_partition", feature = "grin_with_edge_property"))]
    pub fn grin_get_total_edge_num_by_type(
        arg1: GrinPartitionedGraph,
        arg2: GrinEdgeType,
    ) -> usize;

    #[cfg(feature = "grin_assume_by_type_vertex_original_id")]
    pub fn grin_get_vertex_from_original_id_by_type(
        arg1: GrinGraph,
        arg2: GrinVertexType,
        arg3: GrinVertexOriginalId,
    ) -> GrinVertex;

    #[cfg(feature = "grin_trait_select_type_for_vertex_list")]
    pub fn grin_select_type_for_vertex_list(
        arg1: GrinGraph,
        arg2: GrinVertexType,
        arg3: GrinVertexList,
    ) -> GrinVertexList;

    #[cfg(feature = "grin_trait_select_edge_type_for_adjacent_list")]
    pub fn grin_select_edge_type_for_adjacent_list(
        arg1: GrinGraph,
        arg2: GrinEdgeType,
        arg3: GrinAdjacentList,
    ) -> GrinAdjacentList;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_equal_vertex_type(
        arg1: GrinGraph,
        arg2: GrinVertexType,
        arg3: GrinVertexType,
    ) -> bool;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_type(arg1: GrinGraph, arg2: GrinVertex) -> GrinVertexType;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_destroy_vertex_type(arg1: GrinGraph, arg2: GrinVertexType);

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_type_list(arg1: GrinGraph) -> GrinVertexTypeList;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_destroy_vertex_type_list(arg1: GrinGraph, arg2: GrinVertexTypeList);

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_create_vertex_type_list(arg1: GrinGraph) -> GrinVertexTypeList;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_insert_vertex_type_to_list(
        arg1: GrinGraph,
        arg2: GrinVertexTypeList,
        arg3: GrinVertexType,
    ) -> bool;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_type_list_size(arg1: GrinGraph, arg2: GrinVertexTypeList) -> usize;

    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_type_from_list(
        arg1: GrinGraph,
        arg2: GrinVertexTypeList,
        arg3: usize,
    ) -> GrinVertexType;

    #[cfg(feature = "grin_with_vertex_type_name")]
    pub fn grin_get_vertex_type_name(
        arg1: GrinGraph,
        arg2: GrinVertexType,
    ) -> *const ::std::os::raw::c_char;

    #[cfg(feature = "grin_with_vertex_type_name")]
    pub fn grin_get_vertex_type_by_name(
        arg1: GrinGraph,
        arg2: *const ::std::os::raw::c_char,
    ) -> GrinVertexType;

    #[cfg(feature = "grin_trait_natural_id_for_vertex_type")]
    pub fn grin_get_vertex_type_id(arg1: GrinGraph, arg2: GrinVertexType)
        -> GrinVertexTypeId;

    #[cfg(feature = "grin_trait_natural_id_for_vertex_type")]
    pub fn grin_get_vertex_type_from_id(
        arg1: GrinGraph,
        arg2: GrinVertexTypeId,
    ) -> GrinVertexType;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_equal_edge_type(
        arg1: GrinGraph,
        arg2: GrinEdgeType,
        arg3: GrinEdgeType,
    ) -> bool;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_type(arg1: GrinGraph, arg2: GrinEdge) -> GrinEdgeType;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_destroy_edge_type(arg1: GrinGraph, arg2: GrinEdgeType);

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_type_list(arg1: GrinGraph) -> GrinEdgeTypeList;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_destroy_edge_type_list(arg1: GrinGraph, arg2: GrinEdgeTypeList);

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_create_edge_type_list(arg1: GrinGraph) -> GrinEdgeTypeList;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_insert_edge_type_to_list(
        arg1: GrinGraph,
        arg2: GrinEdgeTypeList,
        arg3: GrinEdgeType,
    ) -> bool;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_type_list_size(arg1: GrinGraph, arg2: GrinEdgeTypeList) -> usize;

    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_type_from_list(
        arg1: GrinGraph,
        arg2: GrinEdgeTypeList,
        arg3: usize,
    ) -> GrinEdgeType;

    #[cfg(feature = "grin_with_edge_type_name")]
    pub fn grin_get_edge_type_name(
        arg1: GrinGraph,
        arg2: GrinEdgeType,
    ) -> *const ::std::os::raw::c_char;

    #[cfg(feature = "grin_with_edge_type_name")]
    pub fn grin_get_edge_type_by_name(
        arg1: GrinGraph,
        arg2: *const ::std::os::raw::c_char,
    ) -> GrinEdgeType;

    #[cfg(feature = "grin_trait_natural_id_for_edge_type")]
    pub fn grin_get_edge_type_id(arg1: GrinGraph, arg2: GrinEdgeType) -> GrinEdgeTypeId;

    #[cfg(feature = "grin_trait_natural_id_for_edge_type")]
    pub fn grin_get_edge_type_from_id(arg1: GrinGraph, arg2: GrinEdgeTypeId) -> GrinEdgeType;

    #[doc = " @brief  the src vertex type list"]
    #[cfg(all(feature = "grin_with_vertex_property", feature = "grin_with_edge_property"))]
    pub fn grin_get_src_types_from_edge_type(
        arg1: GrinGraph,
        arg2: GrinEdgeType,
    ) -> GrinVertexTypeList;

    #[doc = " @brief get the dst vertex type list"]
    #[cfg(all(feature = "grin_with_vertex_property", feature = "grin_with_edge_property"))]
    pub fn grin_get_dst_types_from_edge_type(
        arg1: GrinGraph,
        arg2: GrinEdgeType,
    ) -> GrinVertexTypeList;

    #[doc = " @brief get the edge type list related to a given pair of vertex types"]
    #[cfg(all(feature = "grin_with_vertex_property", feature = "grin_with_edge_property"))]
    pub fn grin_get_edge_types_from_vertex_type_pair(
        arg1: GrinGraph,
        arg2: GrinVertexType,
        arg3: GrinVertexType,
    ) -> GrinEdgeTypeList;

    #[cfg(feature = "grin_assume_all_vertex_list_sorted")]
    pub fn grin_smaller_vertex(arg1: GrinGraph, arg2: GrinVertex, arg3: GrinVertex) -> bool;

    #[doc = " @brief get the position of a vertex in a sorted list\n caller must guarantee the input vertex list is sorted to get the correct result\n @param GrinGraph the graph\n @param GrinVertexList the sorted vertex list\n @param VERTEX the vertex to find\n @param pos the returned position of the vertex\n @return false if the vertex is not found"]
    #[cfg(all(feature = "grin_assume_all_vertex_list_sorted", feature = "grin_enable_vertex_list_array"))]
    pub fn grin_get_position_of_vertex_from_sorted_list(
        arg1: GrinGraph,
        arg2: GrinVertexList,
        arg3: GrinVertex,
    ) -> usize;
}
