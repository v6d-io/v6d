#include <stdio.h>
#include "../include/index/label.h"
#include "../include/index/order.h"
#include "../include/partition/partition.h"
#include "../include/partition/reference.h"
#include "../include/partition/topology.h"
#include "../include/property/partition.h"
#include "../include/property/primarykey.h"
#include "../include/property/property.h"
#include "../include/property/propertylist.h"
#include "../include/property/propertytable.h"
#include "../include/property/topology.h"
#include "../include/property/type.h"
#include "../include/topology/adjacentlist.h"
#include "../include/topology/datatype.h"
#include "../include/topology/edgelist.h"
#include "../include/topology/structure.h"
#include "../include/topology/vertexlist.h"

GRIN_GRAPH get_graph(int argc, char** argv) {
#ifdef GRIN_ENABLE_GRAPH_PARTITION
  GRIN_PARTITIONED_GRAPH pg =
      grin_get_partitioned_graph_from_storage(argc - 1, &(argv[1]));
  GRIN_PARTITION_LIST local_partitions = grin_get_local_partition_list(pg);
  GRIN_PARTITION partition =
      grin_get_partition_from_list(pg, local_partitions, 0);
  GRIN_GRAPH g = grin_get_local_graph_from_partition(pg, partition);
#else
  GRIN_GRAPH g = grin_get_graph_from_storage(argc - 1, &(argv[1]));
#endif
  return g;
}

#ifdef GRIN_ENABLE_GRAPH_PARTITION
GRIN_PARTITIONED_GRAPH get_partitioend_graph(int argc, char** argv) {
  GRIN_PARTITIONED_GRAPH pg =
      grin_get_partitioned_graph_from_storage(argc - 1, &(argv[1]));
  return pg;
}
#endif

GRIN_VERTEX_TYPE get_one_vertex_type(GRIN_GRAPH g) {
  GRIN_VERTEX_TYPE_LIST vtl = grin_get_vertex_type_list(g);
  GRIN_VERTEX_TYPE vt = grin_get_vertex_type_from_list(g, vtl, 0);
  grin_destroy_vertex_type_list(g, vtl);
  return vt;
}

GRIN_EDGE_TYPE get_one_edge_type(GRIN_GRAPH g) {
  GRIN_EDGE_TYPE_LIST etl = grin_get_edge_type_list(g);
  GRIN_EDGE_TYPE et = grin_get_edge_type_from_list(g, etl, 0);
  grin_destroy_edge_type_list(g, etl);
  return et;
}

GRIN_VERTEX get_one_vertex(GRIN_GRAPH g) {
  GRIN_VERTEX_LIST vl = grin_get_vertex_list(g);
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
  GRIN_VERTEX v = grin_get_vertex_from_list(g, vl, 0);
#else
  GRIN_VERTEX_LIST_ITERATOR vli = grin_get_vertex_list_begin(g, vl);
  GRIN_VERTEX v = grin_get_vertex_from_iter(g, vli);
  grin_destroy_vertex_list_iter(g, vli);
#endif
  grin_destroy_vertex_list(g, vl);
  return v;
}

GRIN_VERTEX get_vertex_marco(GRIN_GRAPH g) {
  GRIN_VERTEX_LIST vl = grin_get_vertex_list(g);
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
  GRIN_VERTEX v = grin_get_vertex_from_list(g, vl, 3);
#else
  GRIN_VERTEX_LIST_ITERATOR vli = grin_get_vertex_list_begin(g, vl);
  for (int i = 0; i < 3; ++i) {
    grin_get_next_vertex_list_iter(g, vli);
  }
  GRIN_VERTEX v = grin_get_vertex_from_iter(g, vli);
  grin_destroy_vertex_list_iter(g, vli);
#endif
  grin_destroy_vertex_list(g, vl);
  return v;
}

void test_property_type(int argc, char** argv) {
  printf("+++++++++++++++++++++ Test property/type +++++++++++++++++++++\n");

  GRIN_GRAPH g = get_graph(argc, argv);

  printf("------------ Vertex Type ------------\n");
  GRIN_VERTEX_TYPE_LIST vtl = grin_get_vertex_type_list(g);
  size_t vtl_size = grin_get_vertex_type_list_size(g, vtl);
  printf("vertex type list size: %zu\n", vtl_size);

  for (size_t i = 0; i < vtl_size; ++i) {
    printf("------------ Iterate the %zu-th vertex type ------------\n", i);
    GRIN_VERTEX_TYPE vt = grin_get_vertex_type_from_list(g, vtl, i);
#ifdef GRIN_WITH_VERTEX_TYPE_NAME
    const char* vt_name = grin_get_vertex_type_name(g, vt);
    printf("vertex type name: %s\n", vt_name);
    GRIN_VERTEX_TYPE vt0 = grin_get_vertex_type_by_name(g, vt_name);
    grin_destroy_name(g, vt_name);
    if (!grin_equal_vertex_type(g, vt, vt0)) {
      printf("vertex type name not match\n");
    }
    grin_destroy_vertex_type(g, vt0);
#endif
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_TYPE
    printf("vertex type id: %u\n", grin_get_vertex_type_id(g, vt));
    GRIN_VERTEX_TYPE vt1 =
        grin_get_vertex_type_from_id(g, grin_get_vertex_type_id(g, vt));
    if (!grin_equal_vertex_type(g, vt, vt1)) {
      printf("vertex type id not match\n");
    }
    grin_destroy_vertex_type(g, vt1);
#endif
  }
  grin_destroy_vertex_type_list(g, vtl);

  printf(
      "------------ Create a vertex type list of one type \"person\" "
      "------------\n");
  GRIN_VERTEX_TYPE_LIST vtl2 = grin_create_vertex_type_list(g);
#ifdef GRIN_WITH_VERTEX_TYPE_NAME
  GRIN_VERTEX_TYPE vt2_w = grin_get_vertex_type_by_name(g, "knows");
  if (vt2_w == GRIN_NULL_VERTEX_TYPE) {
    printf("(Correct) vertex type of knows does not exists\n");
  }
  GRIN_VERTEX_TYPE vt2 = grin_get_vertex_type_by_name(g, "person");
  if (vt2 == GRIN_NULL_VERTEX_TYPE) {
    printf("(Wrong) vertex type of person can not be found\n");
  } else {
    const char* vt2_name = grin_get_vertex_type_name(g, vt2);
    printf("vertex type name: %s\n", vt2_name);
    grin_destroy_name(g, vt2_name);
  }
#else
  GRIN_VERTEX_TYPE vt2 = get_one_vertex_type(g);
#endif
  grin_insert_vertex_type_to_list(g, vtl2, vt2);
  size_t vtl2_size = grin_get_vertex_type_list_size(g, vtl2);
  printf("created vertex type list size: %zu\n", vtl2_size);
  GRIN_VERTEX_TYPE vt3 = grin_get_vertex_type_from_list(g, vtl2, 0);
  if (!grin_equal_vertex_type(g, vt2, vt3)) {
    printf("vertex type not match\n");
  }
  grin_destroy_vertex_type(g, vt2);
  grin_destroy_vertex_type(g, vt3);
  grin_destroy_vertex_type_list(g, vtl2);

  // edge
  printf("------------ Edge Type ------------\n");
  GRIN_EDGE_TYPE_LIST etl = grin_get_edge_type_list(g);
  size_t etl_size = grin_get_edge_type_list_size(g, etl);
  printf("edge type list size: %zu\n", etl_size);

  for (size_t i = 0; i < etl_size; ++i) {
    printf("------------ Iterate the %zu-th edge type ------------\n", i);
    GRIN_EDGE_TYPE et = grin_get_edge_type_from_list(g, etl, i);
#ifdef GRIN_WITH_EDGE_TYPE_NAME
    const char* et_name = grin_get_edge_type_name(g, et);
    printf("edge type name: %s\n", et_name);
    GRIN_EDGE_TYPE et0 = grin_get_edge_type_by_name(g, et_name);
    grin_destroy_name(g, et_name);
    if (!grin_equal_edge_type(g, et, et0)) {
      printf("edge type name not match\n");
    }
    grin_destroy_edge_type(g, et0);
#endif
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_TYPE
    printf("edge type id: %u\n", grin_get_edge_type_id(g, et));
    GRIN_EDGE_TYPE et1 =
        grin_get_edge_type_from_id(g, grin_get_edge_type_id(g, et));
    if (!grin_equal_edge_type(g, et, et1)) {
      printf("edge type id not match\n");
    }
    grin_destroy_edge_type(g, et1);
#endif
    // relation
    GRIN_VERTEX_TYPE_LIST src_vtl = grin_get_src_types_from_edge_type(g, et);
    size_t src_vtl_size = grin_get_vertex_type_list_size(g, src_vtl);
    printf("source vertex type list size: %zu\n", src_vtl_size);

    GRIN_VERTEX_TYPE_LIST dst_vtl = grin_get_dst_types_from_edge_type(g, et);
    size_t dst_vtl_size = grin_get_vertex_type_list_size(g, dst_vtl);
    printf("destination vertex type list size: %zu\n", dst_vtl_size);

    if (src_vtl_size != dst_vtl_size) {
      printf("source and destination vertex type list size not match\n");
    }
    for (size_t j = 0; j < src_vtl_size; ++j) {
      GRIN_VERTEX_TYPE src_vt = grin_get_vertex_type_from_list(g, src_vtl, j);
      GRIN_VERTEX_TYPE dst_vt = grin_get_vertex_type_from_list(g, dst_vtl, j);
      const char* src_vt_name = grin_get_vertex_type_name(g, src_vt);
      const char* dst_vt_name = grin_get_vertex_type_name(g, dst_vt);
      const char* et_name = grin_get_edge_type_name(g, et);
      printf("edge type name: %s-%s-%s\n", src_vt_name, et_name, dst_vt_name);
      grin_destroy_name(g, src_vt_name);
      grin_destroy_name(g, dst_vt_name);
      grin_destroy_name(g, et_name);
      grin_destroy_vertex_type(g, src_vt);
      grin_destroy_vertex_type(g, dst_vt);
    }
    grin_destroy_vertex_type_list(g, src_vtl);
    grin_destroy_vertex_type_list(g, dst_vtl);
  }
  grin_destroy_edge_type_list(g, etl);

  printf(
      "------------ Create an edge type list of one type \"created\" "
      "------------\n");
  GRIN_EDGE_TYPE_LIST etl2 = grin_create_edge_type_list(g);
#ifdef GRIN_WITH_EDGE_TYPE_NAME
  GRIN_EDGE_TYPE et2_w = grin_get_edge_type_by_name(g, "person");
  if (et2_w == GRIN_NULL_EDGE_TYPE) {
    printf("(Correct) edge type of person does not exists\n");
  }
  GRIN_EDGE_TYPE et2 = grin_get_edge_type_by_name(g, "created");
  if (et2 == GRIN_NULL_EDGE_TYPE) {
    printf("(Wrong) edge type of created can not be found\n");
  } else {
    const char* et2_name = grin_get_edge_type_name(g, et2);
    printf("edge type name: %s\n", et2_name);
    grin_destroy_name(g, et2_name);
  }
#else
  GRIN_EDGE_TYPE et2 = get_one_edge_type(g);
#endif
  grin_insert_edge_type_to_list(g, etl2, et2);
  size_t etl2_size = grin_get_edge_type_list_size(g, etl2);
  printf("created edge type list size: %zu\n", etl2_size);
  GRIN_EDGE_TYPE et3 = grin_get_edge_type_from_list(g, etl2, 0);
  if (!grin_equal_edge_type(g, et2, et3)) {
    printf("edge type not match\n");
  }
  grin_destroy_edge_type(g, et2);
  grin_destroy_edge_type(g, et3);
  grin_destroy_edge_type_list(g, etl2);

  grin_destroy_graph(g);
}

void test_property_topology(int argc, char** argv) {
  printf(
      "+++++++++++++++++++++ Test property/topology +++++++++++++++++++++\n");
  GRIN_GRAPH g = get_graph(argc, argv);
  GRIN_VERTEX_TYPE vt = get_one_vertex_type(g);
  GRIN_EDGE_TYPE et = get_one_edge_type(g);
  const char* vt_name = grin_get_vertex_type_name(g, vt);
  const char* et_name = grin_get_edge_type_name(g, et);

#ifdef GRIN_ENABLE_GRAPH_PARTITION
  GRIN_PARTITIONED_GRAPH pg = get_partitioend_graph(argc, argv);
  size_t tvnum = grin_get_total_vertex_num_by_type(pg, vt);
  printf("total vertex num of %s: %zu\n", vt_name, tvnum);
  size_t tenum = grin_get_total_edge_num_by_type(pg, et);
  printf("total edge num of %s: %zu\n", et_name, tenum);
  grin_destroy_partitioned_graph(pg);
#endif

#ifdef GRIN_ENABLE_VERTEX_LIST
  GRIN_VERTEX_LIST vl = grin_get_vertex_list(g);

#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
  size_t vl_size = grin_get_vertex_list_size(g, vl);
  printf("vertex list size: %zu\n", vl_size);
#endif

#ifdef GRIN_TRAIT_SELECT_TYPE_FOR_VERTEX_LIST
  GRIN_VERTEX_LIST typed_vl = grin_select_type_for_vertex_list(g, vt, vl);
  size_t typed_vnum = grin_get_vertex_num_by_type(g, vt);

#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
  size_t typed_vl_size = grin_get_vertex_list_size(g, typed_vl);
  printf("vertex number under type: %zu %zu\n", typed_vl_size, typed_vnum);

  for (size_t j = 0; j < typed_vl_size; ++j) {
    GRIN_VERTEX v = grin_get_vertex_from_list(g, typed_vl, j);
    GRIN_VERTEX_TYPE v_type = grin_get_vertex_type(g, v);
    if (!grin_equal_vertex_type(g, v_type, vt)) {
      printf("vertex type not match\n");
    }
    grin_destroy_vertex_type(g, v_type);
    grin_destroy_vertex(g, v);
  }
#else
  GRIN_VERTEX_LIST_ITERATOR vli = grin_get_vertex_list_begin(g, typed_vl);
  size_t typed_vl_size = 0;
  while (grin_is_vertex_list_end(g, vli) == false) {
    ++typed_vl_size;
    GRIN_VERTEX v = grin_get_vertex_from_iter(g, vli);
    GRIN_VERTEX_TYPE v_type = grin_get_vertex_type(g, v);
    if (!grin_equal_vertex_type(g, v_type, vt)) {
      printf("vertex type not match\n");
    }
    grin_destroy_vertex_type(g, v_type);
    grin_destroy_vertex(g, v);
    grin_get_next_vertex_list_iter(g, vli);
  }
  printf("vertex number under type: %zu %zu\n", typed_vl_size, typed_vnum);
  grin_destroy_vertex_list_iter(g, vli);
#endif

  grin_destroy_vertex_list(g, typed_vl);
#endif
  grin_destroy_vertex_list(g, vl);
#endif

#ifdef GRIN_ASSUME_BY_TYPE_VERTEX_ORIGINAL_ID
  GRIN_DATATYPE dt = grin_get_vertex_original_id_type(g);
  if (dt == Int64) {
    long int v0id = 4;
    GRIN_VERTEX v0 = grin_get_vertex_from_original_id_by_type(g, vt, &v0id);
    if (v0 == GRIN_NULL_VERTEX) {
      printf("(Wrong) vertex of id %ld can not be found\n", v0id);
    } else {
      printf("vertex of original id %ld found\n", v0id);
      GRIN_VERTEX_ORIGINAL_ID oid0 = grin_get_vertex_original_id(g, v0);
      printf("get vertex original id: %ld\n", *((long int*) oid0));
      grin_destroy_vertex_original_id(g, oid0);
    }
    grin_destroy_vertex(g, v0);
  }
#endif

#ifdef GRIN_ENABLE_EDGE_LIST
  GRIN_EDGE_LIST el = grin_get_edge_list(g);
#ifdef GRIN_ENABLE_EDGE_LIST_ARRAY
  size_t el_size = grin_get_edge_list_size(g, el);
  printf("edge list size: %zu\n", el_size);
#endif

#ifdef GRIN_TRAIT_SELECT_TYPE_FOR_EDGE_LIST
  GRIN_EDGE_LIST typed_el = grin_select_type_for_edge_list(g, et, el);
  size_t typed_enum = grin_get_edge_num_by_type(g, et);

#ifdef GRIN_ENABLE_EDGE_LIST_ARRAY
  size_t typed_el_size = grin_get_edge_list_size(g, typed_el);
  printf("edge number under type: %zu %zu\n", typed_el_size, typed_enum);

  for (size_t j = 0; j < typed_el_size; ++j) {
    GRIN_EDGE e = grin_get_edge_from_list(g, typed_el, j);
    GRIN_EDGE_TYPE e_type = grin_get_edge_type(g, e);
    if (!grin_equal_edge_type(g, e_type, et)) {
      printf("edge type not match\n");
    }
    grin_destroy_edge_type(g, e_type);
    grin_destroy_edge(g, e);
  }
#else
  GRIN_EDGE_LIST_ITERATOR eli = grin_get_edge_list_begin(g, typed_el);
  size_t typed_el_size = 0;
  while (grin_is_edge_list_end(g, eli) == false) {
    ++typed_el_size;
    GRIN_EDGE e = grin_get_edge_from_iter(g, eli);
    GRIN_EDGE_TYPE e_type = grin_get_edge_type(g, e);
    if (!grin_equal_edge_type(g, e_type, et)) {
      printf("edge type not match\n");
    }
    grin_destroy_edge_type(g, e_type);
    grin_destroy_edge(g, e);
    grin_get_next_edge_list_iter(g, eli);
  }
  printf("edge number under type: %zu %zu\n", typed_el_size, typed_enum);
  grin_destroy_edge_list_iter(g, eli);
#endif

  grin_destroy_edge_list(g, typed_el);
#endif
  grin_destroy_edge_list(g, el);
#endif

  grin_destroy_name(g, vt_name);
  grin_destroy_name(g, et_name);
  grin_destroy_vertex_type(g, vt);
  grin_destroy_edge_type(g, et);
  grin_destroy_graph(g);
}

void test_property_vertex_table(int argc, char** argv) {
  printf("+++++++++++++++++++++ Test property/table +++++++++++++++++++++\n");
  GRIN_GRAPH g = get_graph(argc, argv);

  printf("------------ Vertex property table ------------\n");
  GRIN_VERTEX_TYPE_LIST vtl = grin_get_vertex_type_list(g);
  size_t vtl_size = grin_get_vertex_type_list_size(g, vtl);
  for (size_t i = 0; i < vtl_size; ++i) {
    GRIN_VERTEX_TYPE vt = grin_get_vertex_type_from_list(g, vtl, i);

    GRIN_VERTEX_PROPERTY_LIST vpl =
        grin_get_vertex_property_list_by_type(g, vt);
    GRIN_VERTEX_PROPERTY_TABLE vpt =
        grin_get_vertex_property_table_by_type(g, vt);

    GRIN_VERTEX_LIST vl = grin_get_vertex_list(g);
    GRIN_VERTEX_LIST typed_vl = grin_select_type_for_vertex_list(g, vt, vl);
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    size_t typed_vl_size = grin_get_vertex_list_size(g, typed_vl);
#else
    size_t typed_vl_size = grin_get_vertex_num_by_type(g, vt);
#endif
    size_t vpl_size = grin_get_vertex_property_list_size(g, vpl);
    printf("vertex list size: %zu vertex property list size: %zu\n",
           typed_vl_size, vpl_size);

#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    for (size_t i = 0; i < typed_vl_size; ++i) {
      GRIN_VERTEX v = grin_get_vertex_from_list(g, typed_vl, i);
#else
    GRIN_VERTEX_LIST_ITERATOR vli = grin_get_vertex_list_begin(g, typed_vl);
    size_t i = 0;
    while (grin_is_vertex_list_end(g, vli) == 0) {
      GRIN_VERTEX v = grin_get_vertex_from_iter(g, vli);
#endif
      GRIN_ROW row = grin_get_row_from_vertex_property_table(g, vpt, v, vpl);
      for (size_t j = 0; j < vpl_size; ++j) {
        GRIN_VERTEX_PROPERTY vp = grin_get_vertex_property_from_list(g, vpl, j);
        GRIN_VERTEX_PROPERTY vt1 = grin_get_vertex_property_vertex_type(g, vp);
        if (!grin_equal_vertex_type(g, vt, vt1)) {
          printf("vertex type not match by property\n");
        }
        grin_destroy_vertex_type(g, vt1);
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_PROPERTY
        unsigned int id = grin_get_vertex_property_id(g, vt, vp);
        GRIN_VERTEX_PROPERTY vp1 = grin_get_vertex_property_from_id(g, vt, id);
        if (!grin_equal_vertex_property(g, vp, vp1)) {
          printf("vertex property not match by id\n");
        }
        grin_destroy_vertex_property(g, vp1);
#else
        unsigned int id = ~0;
#endif

#ifdef GRIN_WITH_VERTEX_PROPERTY_NAME
        const char* vp_name = grin_get_vertex_property_name(g, vp);
        GRIN_VERTEX_PROPERTY vp2 =
            grin_get_vertex_property_by_name(g, vt, vp_name);
        if (!grin_equal_vertex_property(g, vp, vp2)) {
          printf("vertex property not match by name\n");
        }
#else
        const char* vp_name = "unknown";
#endif
        GRIN_DATATYPE dt = grin_get_vertex_property_data_type(g, vp);
        const void* pv =
            grin_get_value_from_vertex_property_table(g, vpt, v, vp);
        const void* rv = grin_get_value_from_row(g, row, dt, j);
        if (dt == Int64) {
          printf("vp_id %u v%zu %s value: %ld %ld\n", id, i, vp_name,
                 *((long int*) pv), *((long int*) rv));
        } else if (dt == String) {
          printf("vp_id %u v%zu %s value: %s %s\n", id, i, vp_name, (char*) pv,
                 (char*) rv);
        }
        grin_destroy_value(g, dt, pv);
        grin_destroy_value(g, dt, rv);
        grin_destroy_vertex_property(g, vp);
      }
      grin_destroy_row(g, row);
      grin_destroy_vertex(g, v);
#ifdef GRIN_ENABLE_VERTEX_LIST_ARRAY
    }
#else
      grin_get_next_vertex_list_iter(g, vli);
      ++i;
    }
    grin_destroy_vertex_list_iter(g, vli);
#endif

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_VERTEX_PROPERTY
    GRIN_VERTEX_PROPERTY vp3 =
        grin_get_vertex_property_from_id(g, vt, vpl_size);
    if (vp3 == GRIN_NULL_VERTEX_PROPERTY) {
      printf("(Correct) vertex property of id %zu does not exist\n", vpl_size);
    } else {
      printf("(Wrong) vertex property of id %zu exists\n", vpl_size);
      grin_destroy_vertex_property(g, vp3);
    }
#endif

#ifdef GRIN_WITH_VERTEX_PROPERTY_NAME
    GRIN_VERTEX_PROPERTY vp4 =
        grin_get_vertex_property_by_name(g, vt, "unknown");
    if (vp4 == GRIN_NULL_VERTEX_PROPERTY) {
      printf("(Correct) vertex property of name \"unknown\" does not exist\n");
    } else {
      printf("(Wrong) vertex property of name \"unknown\" exists\n");
      grin_destroy_vertex_property(g, vp4);
    }

    GRIN_VERTEX_PROPERTY_LIST vpl1 =
        grin_get_vertex_properties_by_name(g, "unknown");
    if (vpl1 == GRIN_NULL_LIST) {
      printf(
          "(Correct) vertex properties of name \"unknown\" does not exist\n");
    } else {
      printf("(Wrong) vertex properties of name \"unknown\" exists\n");
      grin_destroy_vertex_property_list(g, vpl1);
    }

    GRIN_VERTEX_PROPERTY_LIST vpl2 =
        grin_get_vertex_properties_by_name(g, "name");
    if (vpl2 == GRIN_NULL_LIST) {
      printf("(Wrong) vertex properties of name \"name\" does not exist\n");
    } else {
      printf("(Correct) vertex properties of name \"name\" exists\n");
      size_t vpl2_size = grin_get_vertex_property_list_size(g, vpl2);
      for (size_t i = 0; i < vpl2_size; ++i) {
        GRIN_VERTEX_PROPERTY vp5 =
            grin_get_vertex_property_from_list(g, vpl2, i);
        GRIN_VERTEX_TYPE vt5 = grin_get_vertex_property_vertex_type(g, vp5);
        const char* vp5_name = grin_get_vertex_property_name(g, vp5);
        const char* vt5_name = grin_get_vertex_type_name(g, vt5);
        printf("vertex type name: %s, vertex property name: %s\n", vt5_name,
               vp5_name);
        grin_destroy_vertex_property(g, vp5);
        grin_destroy_vertex_type(g, vt5);
        grin_destroy_name(g, vt5_name);
        grin_destroy_name(g, vp5_name);
      }
      grin_destroy_vertex_property_list(g, vpl2);
    }
#endif

    grin_destroy_vertex_list(g, typed_vl);
    grin_destroy_vertex_list(g, vl);
    grin_destroy_vertex_property_list(g, vpl);
    grin_destroy_vertex_property_table(g, vpt);
  }
  grin_destroy_vertex_type_list(g, vtl);
  grin_destroy_graph(g);
}

void test_property_edge_table(int argc, char** argv) {
  printf("------------ Edge property table ------------\n");
  GRIN_GRAPH g = get_graph(argc, argv);
  // edge
  GRIN_VERTEX v = get_vertex_marco(g);
  GRIN_VERTEX_TYPE vt = grin_get_vertex_type(g, v);
  GRIN_ADJACENT_LIST al = grin_get_adjacent_list(g, OUT, v);
#ifdef GRIN_ENABLE_ADJACENT_LIST_ARRAY
  printf("adjacent list size: %zu\n", grin_get_adjacent_list_size(g, al));
#endif

  GRIN_EDGE_TYPE_LIST etl = grin_get_edge_type_list(g);
  size_t etl_size = grin_get_edge_type_list_size(g, etl);
  printf("edge type list size: %zu\n", etl_size);

  for (size_t i = 0; i < etl_size; ++i) {
    GRIN_EDGE_TYPE et = grin_get_edge_type_from_list(g, etl, i);
    GRIN_EDGE_PROPERTY_TABLE ept = grin_get_edge_property_table_by_type(g, et);
    GRIN_EDGE_PROPERTY_LIST epl = grin_get_edge_property_list_by_type(g, et);
    size_t epl_size = grin_get_edge_property_list_size(g, epl);
    printf("edge property list size: %zu\n", epl_size);

#ifdef GRIN_TRAIT_SELECT_EDGE_TYPE_FOR_ADJACENT_LIST
    GRIN_ADJACENT_LIST al1 = grin_select_edge_type_for_adjacent_list(g, et, al);

#ifdef GRIN_ENABLE_ADJACENT_LIST_ARRAY
    size_t al1_size = grin_get_adjacent_list_size(g, al1);
    printf("selected adjacent list size: %zu\n", al1_size);
#endif

#ifdef GRIN_ENABLE_ADJACENT_LIST_ARRAY
    for (size_t j = 0; j < al1_size; ++j) {
      GRIN_EDGE e = grin_get_edge_from_adjacent_list(g, al1, j);
#else
    GRIN_ADJACENT_LIST_ITERATOR ali = grin_get_adjacent_list_begin(g, al1);
    size_t j = 0;
    while (grin_is_adjacent_list_end(g, ali) == false) {
      GRIN_EDGE e = grin_get_edge_from_adjacent_list_iter(g, ali);
#endif
      GRIN_EDGE_TYPE et1 = grin_get_edge_type(g, e);
      if (!grin_equal_edge_type(g, et, et1)) {
        printf("edge type does not match\n");
      }

      GRIN_ROW row = grin_get_row_from_edge_property_table(g, ept, e, epl);
      for (size_t k = 0; k < epl_size; ++k) {
        GRIN_EDGE_PROPERTY ep = grin_get_edge_property_from_list(g, epl, k);
        GRIN_EDGE_TYPE et2 = grin_get_edge_property_edge_type(g, ep);
        if (!grin_equal_edge_type(g, et, et2)) {
          printf("edge type does not match\n");
        }
        grin_destroy_edge_type(g, et2);

        const char* ep_name = grin_get_edge_property_name(g, ep);
        printf("edge property name: %s\n", ep_name);

#ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_PROPERTY
        unsigned int id = grin_get_edge_property_id(g, et, ep);
        GRIN_EDGE_PROPERTY ep1 = grin_get_edge_property_from_id(g, et, id);
        if (!grin_equal_edge_property(g, ep, ep1)) {
          printf("edge property not match by id\n");
        }
        grin_destroy_edge_property(g, ep1);
#else
        unsigned int id = ~0;
#endif
        GRIN_DATATYPE dt = grin_get_edge_property_data_type(g, ep);
        const void* pv = grin_get_value_from_edge_property_table(g, ept, e, ep);
        const void* rv = grin_get_value_from_row(g, row, dt, k);
        if (dt == Int64) {
          printf("ep_id %u e%zu %s value: %ld %ld\n", id, j, ep_name,
                 *((long int*) pv), *((long int*) rv));
        } else if (dt == String) {
          printf("ep_id %u e%zu %s value: %s %s\n", id, j, ep_name, (char*) pv,
                 (char*) rv);
        } else if (dt == Double) {
          printf("ep_id %u e%zu %s value: %f %f\n", id, j, ep_name,
                 *((double*) pv), *((double*) rv));
        }
        grin_destroy_edge_property(g, ep);
        grin_destroy_name(g, ep_name);
        grin_destroy_value(g, dt, pv);
        grin_destroy_value(g, dt, rv);
      }

      grin_destroy_row(g, row);
      grin_destroy_edge_type(g, et1);
      grin_destroy_edge(g, e);

#ifdef GRIN_ENABLE_ADJACENT_LIST_ARRAY
    }
#else
      grin_get_next_adjacent_list_iter(g, ali);
      ++j;
    }
    grin_destroy_adjacent_list_iter(g, ali);
#endif

    grin_destroy_adjacent_list(g, al1);
#endif

    for (size_t j = 0; j < epl_size; ++j) {
      GRIN_EDGE_PROPERTY ep = grin_get_edge_property_from_list(g, epl, j);
      GRIN_EDGE_TYPE et1 = grin_get_edge_property_edge_type(g, ep);
      if (!grin_equal_edge_type(g, et, et1)) {
        printf("edge type does not match\n");
      }
      const char* ep_name1 = grin_get_edge_property_name(g, ep);
      const char* et_name = grin_get_edge_type_name(g, et);
      printf("edge property name: %s, edge property type name: %s\n", ep_name1,
             et_name);

      grin_destroy_edge_type(g, et1);
      grin_destroy_name(g, ep_name1);
      grin_destroy_name(g, et_name);

#ifdef GRIN_WITH_EDGE_PROPERTY_NAME
      const char* ep_name = grin_get_edge_property_name(g, ep);
      GRIN_EDGE_PROPERTY ep2 = grin_get_edge_property_by_name(g, et, ep_name);
      if (!grin_equal_edge_property(g, ep, ep2)) {
        printf("edge property not match by name\n");
      }
#else
      const char* ep_name = "unknown";
#endif
      grin_destroy_edge_property(g, ep);
    }
#ifdef GRIN_TRAIT_NATURAL_ID_FOR_EDGE_PROPERTY
    GRIN_EDGE_PROPERTY ep3 = grin_get_edge_property_from_id(g, et, epl_size);
    if (ep3 == GRIN_NULL_EDGE_PROPERTY) {
      printf("(Correct) edge property of id %zu does not exist\n", epl_size);
    } else {
      printf("(Wrong) edge property of id %zu exists\n", epl_size);
      grin_destroy_edge_property(g, ep3);
    }
#endif

#ifdef GRIN_WITH_EDGE_PROPERTY_NAME
    GRIN_EDGE_PROPERTY ep4 = grin_get_edge_property_by_name(g, et, "unknown");
    if (ep4 == GRIN_NULL_EDGE_PROPERTY) {
      printf("(Correct) edge property of name \"unknown\" does not exist\n");
    } else {
      printf("(Wrong) edge property of name \"unknown\" exists\n");
      grin_destroy_edge_property(g, ep4);
    }

    GRIN_EDGE_PROPERTY_LIST epl1 =
        grin_get_edge_properties_by_name(g, "unknown");
    if (epl1 == GRIN_NULL_LIST) {
      printf("(Correct) edge properties of name \"unknown\" does not exist\n");
    } else {
      printf("(Wrong) edge properties of name \"unknown\" exists\n");
      grin_destroy_edge_property_list(g, epl1);
    }

    GRIN_EDGE_PROPERTY_LIST epl2 =
        grin_get_edge_properties_by_name(g, "weight");
    if (epl2 == GRIN_NULL_LIST) {
      printf("(Wrong) edge properties of name \"weight\" does not exist\n");
    } else {
      printf("(Correct) edge properties of name \"weight\" exists\n");
      size_t epl2_size = grin_get_edge_property_list_size(g, epl2);
      for (size_t i = 0; i < epl2_size; ++i) {
        GRIN_EDGE_PROPERTY ep5 = grin_get_edge_property_from_list(g, epl2, i);
        GRIN_EDGE_TYPE et5 = grin_get_edge_property_edge_type(g, ep5);
        const char* ep5_name = grin_get_edge_property_name(g, ep5);
        const char* et5_name = grin_get_edge_type_name(g, et5);
        printf("edge type name: %s, edge property name: %s\n", et5_name,
               ep5_name);
        grin_destroy_edge_property(g, ep5);
        grin_destroy_edge_type(g, et5);
        grin_destroy_name(g, et5_name);
        grin_destroy_name(g, ep5_name);
      }
      grin_destroy_edge_property_list(g, epl2);
    }
#endif
    grin_destroy_edge_type(g, et);
  }

  grin_destroy_vertex(g, v);
  grin_destroy_vertex_type(g, vt);
  grin_destroy_adjacent_list(g, al);
  grin_destroy_edge_type_list(g, etl);
  grin_destroy_graph(g);
}

void test_property_primary_key(int argc, char** argv) {
  printf(
      "+++++++++++++++++++++ Test property/primary key "
      "+++++++++++++++++++++\n");
  GRIN_GRAPH g = get_graph(argc, argv);
  GRIN_VERTEX_TYPE_LIST vtl = grin_get_vertex_types_with_primary_keys(g);
  size_t vtl_size = grin_get_vertex_type_list_size(g, vtl);
  printf("vertex type list size: %zu\n", vtl_size);

  unsigned id_type[7] = {~0, 0, 0, 1, 0, 1, 0};

  for (size_t i = 0; i < vtl_size; ++i) {
    GRIN_VERTEX_TYPE vt = grin_get_vertex_type_from_list(g, vtl, i);
    const char* vt_name = grin_get_vertex_type_name(g, vt);
    printf("vertex type name: %s\n", vt_name);
    grin_destroy_name(g, vt_name);

    GRIN_VERTEX_PROPERTY_LIST vpl = grin_get_primary_keys_by_vertex_type(g, vt);
    size_t vpl_size = grin_get_vertex_property_list_size(g, vpl);
    printf("primary key list size: %zu\n", vpl_size);

    for (size_t j = 0; j < vpl_size; ++j) {
      GRIN_VERTEX_PROPERTY vp = grin_get_vertex_property_from_list(g, vpl, j);
      const char* vp_name = grin_get_vertex_property_name(g, vp);
      printf("primary key name: %s\n", vp_name);
      grin_destroy_name(g, vp_name);
      grin_destroy_vertex_property(g, vp);
    }

    GRIN_VERTEX_PROPERTY vp = grin_get_vertex_property_from_list(g, vpl, 0);
    GRIN_DATATYPE dt = grin_get_vertex_property_data_type(g, vp);

    for (size_t j = 1; j <= 6; ++j) {
      GRIN_ROW r = grin_create_row(g);
      grin_insert_value_to_row(g, r, dt, (void*) (&j));
      GRIN_VERTEX v = grin_get_vertex_by_primary_keys(g, vt, r);
      if (id_type[j] == i) {
        if (v == GRIN_NULL_VERTEX) {
          printf("(Wrong) vertex of primary keys %zu does not exist\n", j);
        } else {
          GRIN_VERTEX_ORIGINAL_ID oid0 = grin_get_vertex_original_id(g, v);
          printf("(Correct) vertex of primary keys %zu exists %ld\n", j,
                 *((long int*) oid0));
          grin_destroy_vertex_original_id(g, oid0);
          grin_destroy_vertex(g, v);
        }
      } else {
        if (v == GRIN_NULL_VERTEX) {
          printf("(Correct) vertex of primary keys %zu does not exist\n", j);
        } else {
          printf("(Wrong) vertex of primary keys %zu exists\n", j);
          grin_destroy_vertex(g, v);
        }
      }
      grin_destroy_row(g, r);
    }

    grin_destroy_vertex_property(g, vp);
    grin_destroy_vertex_property_list(g, vpl);
    grin_destroy_vertex_type(g, vt);
  }
}

void test_property(int argc, char** argv) {
  test_property_type(argc, argv);
  test_property_topology(argc, argv);
  test_property_vertex_table(argc, argv);
  test_property_edge_table(argc, argv);
  test_property_primary_key(argc, argv);
}

int main(int argc, char** argv) {
  test_property(argc, argv);
  return 0;
}
