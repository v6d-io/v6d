#[doc = "< incoming"]
pub const GRIN_DIRECTION_IN: GRIN_DIRECTION = 0;
#[doc = "< outgoing"]
pub const GRIN_DIRECTION_OUT: GRIN_DIRECTION = 1;
#[doc = "< incoming & outgoing"]
pub const GRIN_DIRECTION_BOTH: GRIN_DIRECTION = 2;
#[doc = " Enumerates the directions of edges with respect to a certain vertex"]
pub type GRIN_DIRECTION = ::std::os::raw::c_uint;
#[doc = "< other unknown types"]
pub const GRIN_DATATYPE_Undefined: GRIN_DATATYPE = 0;
#[doc = "< int"]
pub const GRIN_DATATYPE_Int32: GRIN_DATATYPE = 1;
#[doc = "< unsigned int"]
pub const GRIN_DATATYPE_UInt32: GRIN_DATATYPE = 2;
#[doc = "< long int"]
pub const GRIN_DATATYPE_Int64: GRIN_DATATYPE = 3;
#[doc = "< unsigned long int"]
pub const GRIN_DATATYPE_UInt64: GRIN_DATATYPE = 4;
#[doc = "< float"]
pub const GRIN_DATATYPE_Float: GRIN_DATATYPE = 5;
#[doc = "< double"]
pub const GRIN_DATATYPE_Double: GRIN_DATATYPE = 6;
#[doc = "< string"]
pub const GRIN_DATATYPE_String: GRIN_DATATYPE = 7;
#[doc = "< short date"]
pub const GRIN_DATATYPE_Date32: GRIN_DATATYPE = 8;
#[doc = "< long date"]
pub const GRIN_DATATYPE_Date64: GRIN_DATATYPE = 9;
#[doc = " Enumerates the datatype supported in the storage"]
pub type GRIN_DATATYPE = ::std::os::raw::c_uint;
#[doc = "@}"]
pub type GRIN_GRAPH = *mut ::std::os::raw::c_void;
pub type GRIN_VERTEX = *mut ::std::os::raw::c_void;
pub type GRIN_EDGE = *mut ::std::os::raw::c_void;
pub type GRIN_VERTEX_ORIGINAL_ID = *mut ::std::os::raw::c_void;
pub type GRIN_VERTEX_LIST = *mut ::std::os::raw::c_void;
pub type GRIN_VERTEX_LIST_ITERATOR = *mut ::std::os::raw::c_void;
pub type GRIN_ADJACENT_LIST = *mut ::std::os::raw::c_void;
pub type GRIN_ADJACENT_LIST_ITERATOR = *mut ::std::os::raw::c_void;
pub type GRIN_PARTITIONED_GRAPH = *mut ::std::os::raw::c_void;
pub type GRIN_PARTITION = *mut ::std::os::raw::c_void;
pub type GRIN_PARTITION_LIST = *mut ::std::os::raw::c_void;
pub type GRIN_PARTITION_ID = ::std::os::raw::c_uint;
pub type GRIN_VERTEX_REF = *mut ::std::os::raw::c_void;
pub type GRIN_VERTEX_TYPE = *mut ::std::os::raw::c_void;
pub type GRIN_VERTEX_TYPE_LIST = *mut ::std::os::raw::c_void;
pub type GRIN_VERTEX_PROPERTY = *mut ::std::os::raw::c_void;
pub type GRIN_VERTEX_PROPERTY_LIST = *mut ::std::os::raw::c_void;
pub type GRIN_VERTEX_PROPERTY_TABLE = *mut ::std::os::raw::c_void;
pub type GRIN_VERTEX_TYPE_ID = ::std::os::raw::c_uint;
pub type GRIN_VERTEX_PROPERTY_ID = ::std::os::raw::c_uint;
pub type GRIN_EDGE_TYPE = *mut ::std::os::raw::c_void;
pub type GRIN_EDGE_TYPE_LIST = *mut ::std::os::raw::c_void;
pub type GRIN_EDGE_PROPERTY = *mut ::std::os::raw::c_void;
pub type GRIN_EDGE_PROPERTY_LIST = *mut ::std::os::raw::c_void;
pub type GRIN_EDGE_PROPERTY_TABLE = *mut ::std::os::raw::c_void;
pub type GRIN_EDGE_TYPE_ID = ::std::os::raw::c_uint;
pub type GRIN_EDGE_PROPERTY_ID = ::std::os::raw::c_uint;
pub type GRIN_ROW = *mut ::std::os::raw::c_void;
extern "C" {
    #[cfg(feature = "grin_enable_adjacent_list")]
    pub fn grin_get_adjacent_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_DIRECTION,
        arg3: GRIN_VERTEX,
    ) -> GRIN_ADJACENT_LIST;
}
extern "C" {
    #[cfg(feature = "grin_enable_adjacent_list")]
    pub fn grin_destroy_adjacent_list(arg1: GRIN_GRAPH, arg2: GRIN_ADJACENT_LIST);
}
extern "C" {
    #[cfg(feature = "grin_enable_adjacent_list_array")]
    pub fn grin_get_adjacent_list_size(arg1: GRIN_GRAPH, arg2: GRIN_ADJACENT_LIST) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_enable_adjacent_list_array")]
    pub fn grin_get_neighbor_from_adjacent_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_ADJACENT_LIST,
        arg3: usize,
    ) -> GRIN_VERTEX;
}
extern "C" {
    #[cfg(feature = "grin_enable_adjacent_list_array")]
    pub fn grin_get_edge_from_adjacent_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_ADJACENT_LIST,
        arg3: usize,
    ) -> GRIN_EDGE;
}
extern "C" {
    #[cfg(feature = "grin_enable_adjacent_list_iterator")]
    pub fn grin_get_adjacent_list_begin(
        arg1: GRIN_GRAPH,
        arg2: GRIN_ADJACENT_LIST,
    ) -> GRIN_ADJACENT_LIST_ITERATOR;
}
extern "C" {
    #[cfg(feature = "grin_enable_adjacent_list_iterator")]
    pub fn grin_destroy_adjacent_list_iter(arg1: GRIN_GRAPH, arg2: GRIN_ADJACENT_LIST_ITERATOR);
}
extern "C" {
    #[cfg(feature = "grin_enable_adjacent_list_iterator")]
    pub fn grin_get_next_adjacent_list_iter(arg1: GRIN_GRAPH, arg2: GRIN_ADJACENT_LIST_ITERATOR);
}
extern "C" {
    #[cfg(feature = "grin_enable_adjacent_list_iterator")]
    pub fn grin_is_adjacent_list_end(arg1: GRIN_GRAPH, arg2: GRIN_ADJACENT_LIST_ITERATOR) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_enable_adjacent_list_iterator")]
    pub fn grin_get_neighbor_from_adjacent_list_iter(
        arg1: GRIN_GRAPH,
        arg2: GRIN_ADJACENT_LIST_ITERATOR,
    ) -> GRIN_VERTEX;
}
extern "C" {
    #[cfg(feature = "grin_enable_adjacent_list_iterator")]
    pub fn grin_get_edge_from_adjacent_list_iter(
        arg1: GRIN_GRAPH,
        arg2: GRIN_ADJACENT_LIST_ITERATOR,
    ) -> GRIN_EDGE;
}
extern "C" {
    #[cfg(all(feature = "grin_assume_has_directed_graph", feature = "grin_assume_has_undirected_graph"))]
    pub fn grin_get_graph_from_storage(
        arg1: ::std::os::raw::c_int,
        arg2: *mut *mut ::std::os::raw::c_char,
    ) -> GRIN_GRAPH;
}
extern "C" {
    #[cfg(all(feature = "grin_assume_has_directed_graph", feature = "grin_assume_has_undirected_graph"))]
    pub fn grin_destroy_graph(arg1: GRIN_GRAPH);
}
extern "C" {
    #[cfg(all(feature = "grin_assume_has_directed_graph", feature = "grin_assume_has_undirected_graph"))]
    pub fn grin_is_directed(arg1: GRIN_GRAPH) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_assume_has_multi_edge_graph")]
    pub fn grin_is_multigraph(arg1: GRIN_GRAPH) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_get_vertex_num(arg1: GRIN_GRAPH) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_get_edge_num(arg1: GRIN_GRAPH) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_destroy_vertex(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX);
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_equal_vertex(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX, arg3: GRIN_VERTEX) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_destroy_vertex_original_id(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_ORIGINAL_ID);
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_get_vertex_original_id_type(arg1: GRIN_GRAPH) -> GRIN_DATATYPE;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_original_id")]
    pub fn grin_get_vertex_original_id(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX,
    ) -> GRIN_VERTEX_ORIGINAL_ID;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_data")]
    pub fn grin_destroy_value(
        arg1: GRIN_GRAPH,
        arg2: GRIN_DATATYPE,
        arg3: *const ::std::os::raw::c_void,
    );
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_data")]
    pub fn grin_destroy_name(arg1: GRIN_GRAPH, arg2: *const ::std::os::raw::c_char);
}
extern "C" {
    #[cfg(feature = "grin_with_edge_data")]
    pub fn grin_destroy_edge(arg1: GRIN_GRAPH, arg2: GRIN_EDGE);
}
extern "C" {
    #[cfg(feature = "grin_with_edge_data")]
    pub fn grin_get_edge_src(arg1: GRIN_GRAPH, arg2: GRIN_EDGE) -> GRIN_VERTEX;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_data")]
    pub fn grin_get_edge_dst(arg1: GRIN_GRAPH, arg2: GRIN_EDGE) -> GRIN_VERTEX;
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_list")]
    pub fn grin_get_vertex_list(arg1: GRIN_GRAPH) -> GRIN_VERTEX_LIST;
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_list")]
    pub fn grin_destroy_vertex_list(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_LIST);
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_list_array")]
    pub fn grin_get_vertex_list_size(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_LIST) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_list_array")]
    pub fn grin_get_vertex_from_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_LIST,
        arg3: usize,
    ) -> GRIN_VERTEX;
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_list_iterator")]
    pub fn grin_get_vertex_list_begin(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_LIST,
    ) -> GRIN_VERTEX_LIST_ITERATOR;
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_list_iterator")]
    pub fn grin_destroy_vertex_list_iter(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_LIST_ITERATOR);
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_list_iterator")]
    pub fn grin_get_next_vertex_list_iter(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_LIST_ITERATOR);
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_list_iterator")]
    pub fn grin_is_vertex_list_end(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_LIST_ITERATOR) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_list_iterator")]
    pub fn grin_get_vertex_from_iter(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_LIST_ITERATOR,
    ) -> GRIN_VERTEX;
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_partitioned_graph_from_storage(
        arg1: ::std::os::raw::c_int,
        arg2: *mut *mut ::std::os::raw::c_char,
    ) -> GRIN_PARTITIONED_GRAPH;
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_destroy_partitioned_graph(arg1: GRIN_PARTITIONED_GRAPH);
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_total_partitions_number(arg1: GRIN_PARTITIONED_GRAPH) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_local_partition_list(arg1: GRIN_PARTITIONED_GRAPH) -> GRIN_PARTITION_LIST;
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_destroy_partition_list(arg1: GRIN_PARTITIONED_GRAPH, arg2: GRIN_PARTITION_LIST);
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_create_partition_list(arg1: GRIN_PARTITIONED_GRAPH) -> GRIN_PARTITION_LIST;
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_insert_partition_to_list(
        arg1: GRIN_PARTITIONED_GRAPH,
        arg2: GRIN_PARTITION_LIST,
        arg3: GRIN_PARTITION,
    ) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_partition_list_size(
        arg1: GRIN_PARTITIONED_GRAPH,
        arg2: GRIN_PARTITION_LIST,
    ) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_partition_from_list(
        arg1: GRIN_PARTITIONED_GRAPH,
        arg2: GRIN_PARTITION_LIST,
        arg3: usize,
    ) -> GRIN_PARTITION;
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_equal_partition(
        arg1: GRIN_PARTITIONED_GRAPH,
        arg2: GRIN_PARTITION,
        arg3: GRIN_PARTITION,
    ) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_destroy_partition(arg1: GRIN_PARTITIONED_GRAPH, arg2: GRIN_PARTITION);
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_partition_info(
        arg1: GRIN_PARTITIONED_GRAPH,
        arg2: GRIN_PARTITION,
    ) -> *const ::std::os::raw::c_void;
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_local_graph_from_partition(
        arg1: GRIN_PARTITIONED_GRAPH,
        arg2: GRIN_PARTITION,
    ) -> GRIN_GRAPH;
}
extern "C" {
    #[cfg(feature = "grin_trait_natural_id_for_partition")]
    pub fn grin_get_partition_from_id(
        arg1: GRIN_PARTITIONED_GRAPH,
        arg2: GRIN_PARTITION_ID,
    ) -> GRIN_PARTITION;
}
extern "C" {
    #[cfg(feature = "grin_trait_natural_id_for_partition")]
    pub fn grin_get_partition_id(
        arg1: GRIN_PARTITIONED_GRAPH,
        arg2: GRIN_PARTITION,
    ) -> GRIN_PARTITION_ID;
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_get_vertex_ref_for_vertex(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX) -> GRIN_VERTEX_REF;
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_destroy_vertex_ref(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_REF);
}
extern "C" {
    #[doc = " @brief get the local vertex from the vertex ref\n if the vertex ref is not regconized, a null vertex is returned\n @param GRIN_GRAPH the graph\n @param GRIN_VERTEX_REF the vertex ref"]
    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_get_vertex_from_vertex_ref(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_REF) -> GRIN_VERTEX;
}
extern "C" {
    #[doc = " @brief get the master partition of a vertex ref.\n Some storage can still provide the master partition of the vertex ref,\n even if the vertex ref can NOT be recognized locally.\n @param GRIN_GRAPH the graph\n @param GRIN_VERTEX_REF the vertex ref"]
    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_get_master_partition_from_vertex_ref(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_REF,
    ) -> GRIN_PARTITION;
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_serialize_vertex_ref(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_REF,
    ) -> *const ::std::os::raw::c_char;
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_destroy_serialized_vertex_ref(
        arg1: GRIN_GRAPH,
        arg2: *const ::std::os::raw::c_char,
    );
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_deserialize_to_vertex_ref(
        arg1: GRIN_GRAPH,
        arg2: *const ::std::os::raw::c_char,
    ) -> GRIN_VERTEX_REF;
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_is_master_vertex(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_enable_vertex_ref")]
    pub fn grin_is_mirror_vertex(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_total_vertex_num(arg1: GRIN_PARTITIONED_GRAPH) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_enable_graph_partition")]
    pub fn grin_get_total_edge_num(arg1: GRIN_PARTITIONED_GRAPH) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_trait_select_master_for_vertex_list")]
    pub fn grin_select_master_for_vertex_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_LIST,
    ) -> GRIN_VERTEX_LIST;
}
extern "C" {
    #[cfg(feature = "grin_trait_select_master_for_vertex_list")]
    pub fn grin_select_mirror_for_vertex_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_LIST,
    ) -> GRIN_VERTEX_LIST;
}
extern "C" {
    #[doc = " @brief get the vertex property name\n @param GRIN_GRAPH the graph\n @param GRIN_VERTEX_PROPERTY the vertex property"]
    #[cfg(feature = "grin_with_vertex_property_name")]
    pub fn grin_get_vertex_property_name(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_PROPERTY,
    ) -> *const ::std::os::raw::c_char;
}
extern "C" {
    #[doc = " @brief get the vertex property with a given name under a specific vertex type\n @param GRIN_GRAPH the graph\n @param GRIN_VERTEX_TYPE the specific vertex type\n @param name the name"]
    #[cfg(feature = "grin_with_vertex_property_name")]
    pub fn grin_get_vertex_property_by_name(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE,
        name: *const ::std::os::raw::c_char,
    ) -> GRIN_VERTEX_PROPERTY;
}
extern "C" {
    #[doc = " @brief get all the vertex properties with a given name\n @param GRIN_GRAPH the graph\n @param name the name"]
    #[cfg(feature = "grin_with_vertex_property_name")]
    pub fn grin_get_vertex_properties_by_name(
        arg1: GRIN_GRAPH,
        name: *const ::std::os::raw::c_char,
    ) -> GRIN_VERTEX_PROPERTY_LIST;
}
extern "C" {
    #[doc = " @brief get the edge property name\n @param GRIN_GRAPH the graph\n @param GRIN_EDGE_PROPERTY the edge property"]
    #[cfg(feature = "grin_with_edge_property_name")]
    pub fn grin_get_edge_property_name(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_PROPERTY,
    ) -> *const ::std::os::raw::c_char;
}
extern "C" {
    #[doc = " @brief get the edge property with a given name under a specific edge type\n @param GRIN_GRAPH the graph\n @param GRIN_EDGE_TYPE the specific edge type\n @param name the name"]
    #[cfg(feature = "grin_with_edge_property_name")]
    pub fn grin_get_edge_property_by_name(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_TYPE,
        name: *const ::std::os::raw::c_char,
    ) -> GRIN_EDGE_PROPERTY;
}
extern "C" {
    #[doc = " @brief get all the edge properties with a given name\n @param GRIN_GRAPH the graph\n @param name the name"]
    #[cfg(feature = "grin_with_edge_property_name")]
    pub fn grin_get_edge_properties_by_name(
        arg1: GRIN_GRAPH,
        name: *const ::std::os::raw::c_char,
    ) -> GRIN_EDGE_PROPERTY_LIST;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_equal_vertex_property(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_PROPERTY,
        arg3: GRIN_VERTEX_PROPERTY,
    ) -> bool;
}
extern "C" {
    #[doc = " @brief destroy vertex property\n @param GRIN_VERTEX_PROPERTY vertex property"]
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_destroy_vertex_property(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_PROPERTY);
}
extern "C" {
    #[doc = " @brief get property data type\n @param GRIN_VERTEX_PROPERTY vertex property"]
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_property_data_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_PROPERTY,
    ) -> GRIN_DATATYPE;
}
extern "C" {
    #[doc = " @brief get the vertex type that the property is bound to\n @param GRIN_VERTEX_PROPERTY vertex property"]
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_property_vertex_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_PROPERTY,
    ) -> GRIN_VERTEX_TYPE;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_equal_edge_property(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_PROPERTY,
        arg3: GRIN_EDGE_PROPERTY,
    ) -> bool;
}
extern "C" {
    #[doc = " @brief destroy edge property\n @param GRIN_EDGE_PROPERTY edge property"]
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_destroy_edge_property(arg1: GRIN_GRAPH, arg2: GRIN_EDGE_PROPERTY);
}
extern "C" {
    #[doc = " @brief get property data type\n @param GRIN_EDGE_PROPERTY edge property"]
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_property_data_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_PROPERTY,
    ) -> GRIN_DATATYPE;
}
extern "C" {
    #[doc = " @brief get the edge type that the property is bound to\n @param GRIN_EDGE_PROPERTY edge property"]
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_property_edge_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_PROPERTY,
    ) -> GRIN_EDGE_TYPE;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_property_list_by_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE,
    ) -> GRIN_VERTEX_PROPERTY_LIST;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_property_list_size(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_PROPERTY_LIST,
    ) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_property_from_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_PROPERTY_LIST,
        arg3: usize,
    ) -> GRIN_VERTEX_PROPERTY;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_create_vertex_property_list(arg1: GRIN_GRAPH) -> GRIN_VERTEX_PROPERTY_LIST;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_destroy_vertex_property_list(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_PROPERTY_LIST);
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_insert_vertex_property_to_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_PROPERTY_LIST,
        arg3: GRIN_VERTEX_PROPERTY,
    ) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_trait_natural_id_for_vertex_property")]
    pub fn grin_get_vertex_property_from_id(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE,
        arg3: GRIN_VERTEX_PROPERTY_ID,
    ) -> GRIN_VERTEX_PROPERTY;
}
extern "C" {
    #[cfg(feature = "grin_trait_natural_id_for_vertex_property")]
    pub fn grin_get_vertex_property_id(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE,
        arg3: GRIN_VERTEX_PROPERTY,
    ) -> GRIN_VERTEX_PROPERTY_ID;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_property_list_by_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_TYPE,
    ) -> GRIN_EDGE_PROPERTY_LIST;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_property_list_size(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_PROPERTY_LIST,
    ) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_property_from_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_PROPERTY_LIST,
        arg3: usize,
    ) -> GRIN_EDGE_PROPERTY;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_create_edge_property_list(arg1: GRIN_GRAPH) -> GRIN_EDGE_PROPERTY_LIST;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_destroy_edge_property_list(arg1: GRIN_GRAPH, arg2: GRIN_EDGE_PROPERTY_LIST);
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_insert_edge_property_to_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_PROPERTY_LIST,
        arg3: GRIN_EDGE_PROPERTY,
    ) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_trait_natural_id_for_edge_property")]
    pub fn grin_get_edge_property_from_id(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_TYPE,
        arg3: GRIN_EDGE_PROPERTY_ID,
    ) -> GRIN_EDGE_PROPERTY;
}
extern "C" {
    #[cfg(feature = "grin_trait_natural_id_for_edge_property")]
    pub fn grin_get_edge_property_id(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_TYPE,
        arg3: GRIN_EDGE_PROPERTY,
    ) -> GRIN_EDGE_PROPERTY_ID;
}
extern "C" {
    #[cfg(feature = "grin_enable_row")]
    pub fn grin_destroy_row(arg1: GRIN_GRAPH, arg2: GRIN_ROW);
}
extern "C" {
    #[doc = " @brief the value of a property from row by its position in row"]
    #[cfg(feature = "grin_enable_row")]
    pub fn grin_get_value_from_row(
        arg1: GRIN_GRAPH,
        arg2: GRIN_ROW,
        arg3: GRIN_DATATYPE,
        arg4: usize,
    ) -> *const ::std::os::raw::c_void;
}
extern "C" {
    #[doc = " @brief create a row, usually to get vertex/edge by primary keys"]
    #[cfg(feature = "grin_enable_row")]
    pub fn grin_create_row(arg1: GRIN_GRAPH) -> GRIN_ROW;
}
extern "C" {
    #[doc = " @brief insert a value to the end of the row"]
    #[cfg(feature = "grin_enable_row")]
    pub fn grin_insert_value_to_row(
        arg1: GRIN_GRAPH,
        arg2: GRIN_ROW,
        arg3: GRIN_DATATYPE,
        arg4: *const ::std::os::raw::c_void,
    ) -> bool;
}
extern "C" {
    #[doc = " @brief destroy vertex property table\n @param GRIN_VERTEX_PROPERTY_TABLE vertex property table"]
    #[cfg(feature = "grin_enable_vertex_property_table")]
    pub fn grin_destroy_vertex_property_table(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_PROPERTY_TABLE);
}
extern "C" {
    #[doc = " @brief get the vertex property table of a certain vertex type\n No matter column or row store strategy is used in the storage,\n GRIN recommends to first get the property table of the vertex type,\n and then fetch values(rows) by vertex and property(list). However,\n GRIN does provide direct row fetching API when GRIN_ASSUME_COLUMN_STORE_FOR_VERTEX_PROPERTY\n is NOT set.\n @param GRIN_GRAPH the graph\n @param GRIN_VERTEX_TYPE the vertex type"]
    #[cfg(feature = "grin_enable_vertex_property_table")]
    pub fn grin_get_vertex_property_table_by_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE,
    ) -> GRIN_VERTEX_PROPERTY_TABLE;
}
extern "C" {
    #[doc = " @brief get vertex property value from table\n @param GRIN_VERTEX_PROPERTY_TABLE vertex property table\n @param GRIN_VERTEX the vertex which is the row index\n @param GRIN_VERTEX_PROPERTY the vertex property which is the column index\n @return can be casted to the property data type by the caller"]
    #[cfg(feature = "grin_enable_vertex_property_table")]
    pub fn grin_get_value_from_vertex_property_table(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_PROPERTY_TABLE,
        arg3: GRIN_VERTEX,
        arg4: GRIN_VERTEX_PROPERTY,
    ) -> *const ::std::os::raw::c_void;
}
extern "C" {
    #[doc = " @brief get vertex row from table\n @param GRIN_VERTEX_PROPERTY_TABLE vertex property table\n @param GRIN_VERTEX the vertex which is the row index\n @param GRIN_VERTEX_PROPERTY_LIST the vertex property list as columns"]
    #[cfg(all(feature = "grin_enable_vertex_property_table", feature = "grin_enable_row"))]
    pub fn grin_get_row_from_vertex_property_table(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_PROPERTY_TABLE,
        arg3: GRIN_VERTEX,
        arg4: GRIN_VERTEX_PROPERTY_LIST,
    ) -> GRIN_ROW;
}
extern "C" {
    #[doc = " @brief destroy edge property table\n @param GRIN_EDGE_PROPERTY_TABLE edge property table"]
    #[cfg(feature = "grin_enable_edge_property_table")]
    pub fn grin_destroy_edge_property_table(arg1: GRIN_GRAPH, arg2: GRIN_EDGE_PROPERTY_TABLE);
}
extern "C" {
    #[doc = " @brief get the edge property table of a certain edge type\n No matter column or row store strategy is used in the storage,\n GRIN recommends to first get the property table of the edge type,\n and then fetch values(rows) by edge and property(list). However,\n GRIN does provide direct row fetching API when GRIN_ASSUME_COLUMN_STORE_FOR_EDGE_PROPERTY\n is NOT set.\n @param GRIN_GRAPH the graph\n @param GRIN_EDGE_TYPE the edge type"]
    #[cfg(feature = "grin_enable_edge_property_table")]
    pub fn grin_get_edge_property_table_by_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_TYPE,
    ) -> GRIN_EDGE_PROPERTY_TABLE;
}
extern "C" {
    #[doc = " @brief get edge property value from table\n @param GRIN_EDGE_PROPERTY_TABLE edge property table\n @param GRIN_EDGE the edge which is the row index\n @param GRIN_EDGE_PROPERTY the edge property which is the column index\n @return can be casted to the property data type by the caller"]
    #[cfg(feature = "grin_enable_edge_property_table")]
    pub fn grin_get_value_from_edge_property_table(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_PROPERTY_TABLE,
        arg3: GRIN_EDGE,
        arg4: GRIN_EDGE_PROPERTY,
    ) -> *const ::std::os::raw::c_void;
}
extern "C" {
    #[doc = " @brief get edge row from table\n @param GRIN_EDGE_PROPERTY_TABLE edge property table\n @param GRIN_EDGE the edge which is the row index\n @param GRIN_EDGE_PROPERTY_LIST the edge property list as columns"]
    #[cfg(all(feature = "grin_enable_edge_property_table", feature = "grin_enable_row"))]
    pub fn grin_get_row_from_edge_property_table(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_PROPERTY_TABLE,
        arg3: GRIN_EDGE,
        arg4: GRIN_EDGE_PROPERTY_LIST,
    ) -> GRIN_ROW;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_num_by_type(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_TYPE) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_num_by_type(arg1: GRIN_GRAPH, arg2: GRIN_EDGE_TYPE) -> usize;
}
extern "C" {
    #[cfg(all(feature = "grin_enable_graph_partition", feature = "grin_with_vertex_property"))]
    pub fn grin_get_total_vertex_num_by_type(
        arg1: GRIN_PARTITIONED_GRAPH,
        arg2: GRIN_VERTEX_TYPE,
    ) -> usize;
}
extern "C" {
    #[cfg(all(feature = "grin_enable_graph_partition", feature = "grin_with_edge_property"))]
    pub fn grin_get_total_edge_num_by_type(
        arg1: GRIN_PARTITIONED_GRAPH,
        arg2: GRIN_EDGE_TYPE,
    ) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_assume_by_type_vertex_original_id")]
    pub fn grin_get_vertex_from_original_id_by_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE,
        arg3: GRIN_VERTEX_ORIGINAL_ID,
    ) -> GRIN_VERTEX;
}
extern "C" {
    #[cfg(feature = "grin_trait_select_type_for_vertex_list")]
    pub fn grin_select_type_for_vertex_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE,
        arg3: GRIN_VERTEX_LIST,
    ) -> GRIN_VERTEX_LIST;
}
extern "C" {
    #[cfg(feature = "grin_trait_select_edge_type_for_adjacent_list")]
    pub fn grin_select_edge_type_for_adjacent_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_TYPE,
        arg3: GRIN_ADJACENT_LIST,
    ) -> GRIN_ADJACENT_LIST;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_equal_vertex_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE,
        arg3: GRIN_VERTEX_TYPE,
    ) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_type(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX) -> GRIN_VERTEX_TYPE;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_destroy_vertex_type(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_TYPE);
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_type_list(arg1: GRIN_GRAPH) -> GRIN_VERTEX_TYPE_LIST;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_destroy_vertex_type_list(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_TYPE_LIST);
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_create_vertex_type_list(arg1: GRIN_GRAPH) -> GRIN_VERTEX_TYPE_LIST;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_insert_vertex_type_to_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE_LIST,
        arg3: GRIN_VERTEX_TYPE,
    ) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_type_list_size(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_TYPE_LIST) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_property")]
    pub fn grin_get_vertex_type_from_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE_LIST,
        arg3: usize,
    ) -> GRIN_VERTEX_TYPE;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_type_name")]
    pub fn grin_get_vertex_type_name(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE,
    ) -> *const ::std::os::raw::c_char;
}
extern "C" {
    #[cfg(feature = "grin_with_vertex_type_name")]
    pub fn grin_get_vertex_type_by_name(
        arg1: GRIN_GRAPH,
        arg2: *const ::std::os::raw::c_char,
    ) -> GRIN_VERTEX_TYPE;
}
extern "C" {
    #[cfg(feature = "grin_trait_natural_id_for_vertex_type")]
    pub fn grin_get_vertex_type_id(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX_TYPE)
        -> GRIN_VERTEX_TYPE_ID;
}
extern "C" {
    #[cfg(feature = "grin_trait_natural_id_for_vertex_type")]
    pub fn grin_get_vertex_type_from_id(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE_ID,
    ) -> GRIN_VERTEX_TYPE;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_equal_edge_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_TYPE,
        arg3: GRIN_EDGE_TYPE,
    ) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_type(arg1: GRIN_GRAPH, arg2: GRIN_EDGE) -> GRIN_EDGE_TYPE;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_destroy_edge_type(arg1: GRIN_GRAPH, arg2: GRIN_EDGE_TYPE);
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_type_list(arg1: GRIN_GRAPH) -> GRIN_EDGE_TYPE_LIST;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_destroy_edge_type_list(arg1: GRIN_GRAPH, arg2: GRIN_EDGE_TYPE_LIST);
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_create_edge_type_list(arg1: GRIN_GRAPH) -> GRIN_EDGE_TYPE_LIST;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_insert_edge_type_to_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_TYPE_LIST,
        arg3: GRIN_EDGE_TYPE,
    ) -> bool;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_type_list_size(arg1: GRIN_GRAPH, arg2: GRIN_EDGE_TYPE_LIST) -> usize;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_property")]
    pub fn grin_get_edge_type_from_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_TYPE_LIST,
        arg3: usize,
    ) -> GRIN_EDGE_TYPE;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_type_name")]
    pub fn grin_get_edge_type_name(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_TYPE,
    ) -> *const ::std::os::raw::c_char;
}
extern "C" {
    #[cfg(feature = "grin_with_edge_type_name")]
    pub fn grin_get_edge_type_by_name(
        arg1: GRIN_GRAPH,
        arg2: *const ::std::os::raw::c_char,
    ) -> GRIN_EDGE_TYPE;
}
extern "C" {
    #[cfg(feature = "grin_trait_natural_id_for_edge_type")]
    pub fn grin_get_edge_type_id(arg1: GRIN_GRAPH, arg2: GRIN_EDGE_TYPE) -> GRIN_EDGE_TYPE_ID;
}
extern "C" {
    #[cfg(feature = "grin_trait_natural_id_for_edge_type")]
    pub fn grin_get_edge_type_from_id(arg1: GRIN_GRAPH, arg2: GRIN_EDGE_TYPE_ID) -> GRIN_EDGE_TYPE;
}
extern "C" {
    #[doc = " @brief  the src vertex type list"]
    #[cfg(all(feature = "grin_with_vertex_property", feature = "grin_with_edge_property"))]
    pub fn grin_get_src_types_from_edge_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_TYPE,
    ) -> GRIN_VERTEX_TYPE_LIST;
}
extern "C" {
    #[doc = " @brief get the dst vertex type list"]
    #[cfg(all(feature = "grin_with_vertex_property", feature = "grin_with_edge_property"))]
    pub fn grin_get_dst_types_from_edge_type(
        arg1: GRIN_GRAPH,
        arg2: GRIN_EDGE_TYPE,
    ) -> GRIN_VERTEX_TYPE_LIST;
}
extern "C" {
    #[doc = " @brief get the edge type list related to a given pair of vertex types"]
    #[cfg(all(feature = "grin_with_vertex_property", feature = "grin_with_edge_property"))]
    pub fn grin_get_edge_types_from_vertex_type_pair(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_TYPE,
        arg3: GRIN_VERTEX_TYPE,
    ) -> GRIN_EDGE_TYPE_LIST;
}
extern "C" {
    #[cfg(feature = "grin_assume_all_vertex_list_sorted")]
    pub fn grin_smaller_vertex(arg1: GRIN_GRAPH, arg2: GRIN_VERTEX, arg3: GRIN_VERTEX) -> bool;
}
extern "C" {
    #[doc = " @brief get the position of a vertex in a sorted list\n caller must guarantee the input vertex list is sorted to get the correct result\n @param GRIN_GRAPH the graph\n @param GRIN_VERTEX_LIST the sorted vertex list\n @param VERTEX the vertex to find\n @param pos the returned position of the vertex\n @return false if the vertex is not found"]
    #[cfg(all(feature = "grin_assume_all_vertex_list_sorted", feature = "grin_enable_vertex_list_array"))]
    pub fn grin_get_position_of_vertex_from_sorted_list(
        arg1: GRIN_GRAPH,
        arg2: GRIN_VERTEX_LIST,
        arg3: GRIN_VERTEX,
    ) -> usize;
}
