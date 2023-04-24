#include <stdio.h>
#include "../include/topology/adjacentlist.h"
#include "../include/topology/datatype.h"
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
#include "../include/property/propertytable.h"
#include "../include/property/topology.h"
#include "../include/property/type.h"
#include "../include/index/label.h"
#include "../include/index/order.h"

GRIN_GRAPH get_graph(int argc, char** argv) {
#ifdef GRIN_ENABLE_GRAPH_PARTITION
    GRIN_PARTITIONED_GRAPH pg = grin_get_partitioned_graph_from_storage(argc-1, &(argv[1]));
    GRIN_PARTITION_LIST local_partitions = grin_get_local_partition_list(pg);
    GRIN_PARTITION partition = grin_get_partition_from_list(pg, local_partitions, 0);
    GRIN_GRAPH g = grin_get_local_graph_from_partition(pg, partition);
#else
    GRIN_GRAPH g = grin_get_graph_from_storage(argc-1, &(argv[1]));
#endif
    return g;
}

#ifdef GRIN_ENABLE_GRAPH_PARTITION
GRIN_PARTITIONED_GRAPH get_partitioend_graph(int argc, char** argv) {
    GRIN_PARTITIONED_GRAPH pg = grin_get_partitioned_graph_from_storage(argc-1, &(argv[1]));
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
        GRIN_VERTEX_TYPE vt1 = grin_get_vertex_type_from_id(g, grin_get_vertex_type_id(g, vt));
        if (!grin_equal_vertex_type(g, vt, vt1)) {
            printf("vertex type id not match\n");
        }
        grin_destroy_vertex_type(g, vt1);
#endif
    }
    grin_destroy_vertex_type_list(g, vtl);

    printf("------------ Create a vertex type list of one type \"person\" ------------\n");
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
        GRIN_EDGE_TYPE et1 = grin_get_edge_type_from_id(g, grin_get_edge_type_id(g, et));
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

    printf("------------ Create an edge type list of one type \"created\" ------------\n");
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
    printf("+++++++++++++++++++++ Test property/topology +++++++++++++++++++++\n");
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
    size_t vl_size = grin_get_vertex_list_size(g, vl);
    printf("vertex list size: %zu\n", vl_size);

    GRIN_VERTEX_LIST typed_vl = grin_select_type_for_vertex_list(g, vt, vl);
    size_t typed_vl_size = grin_get_vertex_list_size(g, typed_vl);
    size_t typed_vnum = grin_get_vertex_num_by_type(g, vt);
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
    grin_destroy_vertex_list(g, typed_vl);
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
            printf("get vertex original id: %ld\n", *((long int*)oid0));
            grin_destroy_vertex_original_id(g, oid0);
        }
        grin_destroy_vertex(g, v0);
    }
#endif

#ifdef GRIN_ENABLE_EDGE_LIST
    GRIN_EDGE_LIST el = grin_get_edge_list(g);
    size_t el_size = grin_get_edge_list_size(g, el);
    printf("edge list size: %zu\n", el_size);

    GRIN_EDGE_LIST typed_el = grin_select_type_for_edge_list(g, et, el);
    size_t typed_el_size = grin_get_edge_list_size(g, typed_el);
    size_t typed_enum = grin_get_edge_num_by_type(g, et);
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
    grin_destroy_edge_list(g, typed_el);
    grin_destroy_edge_list(g, el);
#endif

    grin_destroy_name(g, vt_name);
    grin_destroy_name(g, et_name);
    grin_destroy_vertex_type(g, vt);
    grin_destroy_edge_type(g, et);
    grin_destroy_graph(g);
}

void test_property_table(int argc, char** argv) {
    printf("+++++++++++++++++++++ Test property/table +++++++++++++++++++++\n");
    GRIN_GRAPH g = get_graph(argc, argv);
    GRIN_VERTEX_TYPE vt = get_one_vertex_type(g);
    GRIN_EDGE_TYPE et = get_one_edge_type(g);
    
    printf("------------ Vertex property table ------------\n");
    GRIN_VERTEX_PROPERTY_LIST vpl = grin_get_vertex_property_list_by_type(g, vt);
    GRIN_VERTEX_PROPERTY_TABLE vpt = grin_get_vertex_property_table_by_type(g, vt);

    GRIN_VERTEX_LIST vl = grin_get_vertex_list(g);
    GRIN_VERTEX_LIST typed_vl = grin_select_type_for_vertex_list(g, vt, vl);
    size_t typed_vl_size = grin_get_vertex_list_size(g, typed_vl);
    size_t vpl_size = grin_get_vertex_property_list_size(g, vpl);
    printf("vertex list size: %zu vertex property list size: %zu\n", typed_vl_size, vpl_size);

    for (size_t i = 0; i < typed_vl_size; ++i) {
        GRIN_VERTEX v = grin_get_vertex_from_list(g, typed_vl, i);
        GRIN_ROW row = grin_get_row_from_vertex_property_table(g, vpt, v, vpl);
        for (size_t j = 0; j < vpl_size; ++j) {
            GRIN_VERTEX_PROPERTY vp = grin_get_vertex_property_from_list(g, vpl, j);
            GRIN_DATATYPE dt = grin_get_vertex_property_data_type(g, vp);
            const void* pv = grin_get_value_from_vertex_property_table(g, vpt, v, vp);
            const void* rv = grin_get_value_from_row(g, row, dt, j);
            if (dt == Int64) {
                printf("v%zu p%zu value: %ld %ld\n", i, j, *((long int*)pv), *((long int*)rv));
            } else if (dt == String) {
                printf("v%zu p%zu value: %s %s\n", i, j, (char*)pv, (char*)rv);
            }
            grin_destroy_value(g, dt, pv);
            grin_destroy_value(g, dt, rv);
            grin_destroy_vertex_property(g, vp);
        }
        grin_destroy_row(g, row);
        grin_destroy_vertex(g, v);
    }

    grin_destroy_vertex_list(g, typed_vl);
    grin_destroy_vertex_list(g, vl);
    grin_destroy_vertex_property_list(g, vpl);
    grin_destroy_vertex_property_table(g, vpt);
}  



int main(int argc, char** argv) {
    test_property_type(argc, argv);
    test_property_topology(argc, argv);
    test_property_table(argc, argv);
    return 0;
}