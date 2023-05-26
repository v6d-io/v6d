#include "../include/topology/adjacentlist.h"
#include "../include/topology/edgelist.h"
#include "../include/topology/structure.h"
#include "../include/topology/vertexlist.h"
#include "../include/partition/partition.h"
#include "../include/partition/reference.h"
#include "../include/partition/topology.h"
#include "../include/property/partition.h"
#include "../include/property/primarykey.h"
#include "../include/property/property.h"
#include "../include/property/propertylist.h"
#include "../include/property/row.h"
#include "../include/property/topology.h"
#include "../include/property/type.h"
#include "../include/index/label.h"
#include "../include/index/order.h"
#include "../include/common/error.h"


/// RUST_KEEP pub const GRIN_NULL_DATATYPE: GrinDatatype = GRIN_DATATYPE_UNDEFINED;
/// RUST_KEEP pub const GRIN_NULL_GRAPH: GrinGraph = std::ptr::null_mut();
/// RUST_KEEP pub const GRIN_NULL_VERTEX: GrinVertex = u64::MAX;
/// RUST_KEEP pub const GRIN_NULL_EDGE: GrinEdge = std::ptr::null_mut();
/// RUST_KEEP pub const GRIN_NULL_LIST: *mut ::std::os::raw::c_void = std::ptr::null_mut();
/// RUST_KEEP pub const GRIN_NULL_LIST_ITERATOR: *mut ::std::os::raw::c_void = std::ptr::null_mut();
/// RUST_KEEP pub const GRIN_NULL_PARTITION: GrinPartition = u32::MAX;
/// RUST_KEEP pub const GRIN_NULL_VERTEX_REF: GrinVertexRef = -1;
/// RUST_KEEP pub const GRIN_NULL_VERTEX_TYPE: GrinVertexType = u32::MAX;
/// RUST_KEEP pub const GRIN_NULL_EDGE_TYPE: GrinEdgeType = u32::MAX;
/// RUST_KEEP pub const GRIN_NULL_VERTEX_PROPERTY: GrinVertexProperty = u64::MAX;
/// RUST_KEEP pub const GRIN_NULL_EDGE_PROPERTY: GrinEdgeProperty = u64::MAX;
/// RUST_KEEP pub const GRIN_NULL_ROW: GrinRow = std::ptr::null_mut();
/// RUST_KEEP pub const GRIN_NULL_NATURAL_ID: u32 = u32::MAX;
/// RUST_KEEP pub const GRIN_NULL_SIZE: u32 = u32::MAX;
int ending;
