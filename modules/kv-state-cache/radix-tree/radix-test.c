#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "radix.h"

typedef struct test_data {
    int k_index;
    int v_index;
} test_data;

// token list to insert
int *to_insert[] = {
    (int[]){1, 1, 1, 0},
    (int[]){1, 1, 2, 0},
    (int[]){1, 1, 3, 0},
    (int[]){1, 2, 3, 4, 0},
    (int[]){2, 1, 1, 0},
    (int[]){1, 2, 0},
    (int[]){1, 2, 1, 0},
    (int[]){1, 2, 3, 0},
    (int[]){1, 2, 2, 0},
    (int[]){1, 2, 0},
    (int[]){2, 1, 0},
    (int[]){2, 2, 0},
    (int[]){2, 3, 10, 0},
    (int[]){2, 0},
    (int[]){1, 0},
    (int[]){1, 1, 0},
    (int[]){1, 2, 9, 0},
    (int[]){2, 2, 9, 0},
    (int[]){2, 3, 9, 0},
    (int[]){1, 2, 3, 5, 0},
    (int[]){2, 3, 9, 10, 0},
    (int[]){1, 2, 3, 5, 6, 0},
    /* delete test case1*/
    //(int[]){1, 1, 0},
    //(int[]){2, 4, 0},
    //(int[]){3, 1, 0},
    /* delete test case2*/
    //(int[]){1, 1, 0},
    //(int[]){3, 1, 0},
    /* delete test case3*/
    //(int[]){3, 1, 2, 3, 5, 0},
    //(int[]){1, 1, 0},
    //(int[]){3, 1, 2, 4, 0},
    /*(int[]){1, 1, 1, 1, 3, 0},
    (int[]){1, 1, 2, 4, 6, 0},*/
    //(int[]){103, 343, 123, 454, 0},
    //(int[]){102, 343, 4564, 546, 342, 0},
    //(int[]){103, 343, 4564, 546, 342, 0},
    //(int[]){435, 7645, 4564, 546, 0},
    //(int[]){435, 7645, 4564, 232, 454, 943, 0},
    //(int[]){435, 343, 454, 123, 4533, 0},
    //(int[]){435, 7645, 4564, 232, 454, 943, 0},
    NULL,
};

// token list to delete
int *to_remove[] = {
    //(int[]){2, 3, 10, 0},
    //(int[]){2, 3, 9, 10, 0},
    /* delete test case1*/
    //(int[]){2,4, 0},
    /* delete test case2*/
    //(int[]){3, 1, 0},
    /* delete test case3*/
    //(int[]){3, 1, 2, 4, 0},
    (int[]){2,1,1,0},
    (int[]){1, 2, 9, 0},
    (int[]){2, 2, 9, 0},
    (int[]){2, 3, 9, 0},
    (int[]){1, 2, 3, 5, 0},
    NULL,
};

// token list to find
int *to_find[] = {
    /*(int[]){435, 343, 454, 123, 4533, 0},
    (int[]){103, 343, 0},*/
    (int[]){2,1,1,0},
    (int[]){1, 22, 9, 0},
    (int[]){2, 2, 9, 0},
    (int[]){2, 3, 93, 0},
    (int[]){1, 2, 3, 5, 0},
    NULL,
};

void print_uint_array_as_string(unsigned int array[]) {
    printf("[");
    for (int i = 0; array[i] != 0; i++) {
        printf("%u", array[i]);
        if (array[i + 1] != 0) printf(", ");
    }
    printf("]");
}

void print_uint_array_with_len_as_string(unsigned int array[], int len) {
    printf("[");
    for (int i = 0; i<len; i++) {
        printf("%u", array[i]);
        if (array[i + 1] != 0) printf(", ");
    }
    printf("]");
}

size_t int_array_len(int *s) {
    size_t len = 0;
    while (s[len] != 0) len++;
    return len;
}

unsigned long insert_data_to_rax(rax *t) {
    unsigned long failed_insertions = 0;
    for (int i = 0; to_insert[i] != NULL; i++) {
        struct test_data *td = (struct test_data *)malloc(sizeof(struct test_data));
        td->k_index = i;
        td->v_index = i;
        int retval = raxInsert(t, (int *)to_insert[i], int_array_len(to_insert[i]), (void *)td, NULL);
        if (retval == 0) {
            if (errno == 0) {
                printf("Overwritten token list: ");
                print_uint_array_as_string(to_insert[i]);
                printf(", data: {k_index: %d, v_index: %d}\n", td->k_index, td->v_index);
            } else {
                printf("Failed to insert for OOM:\n");
                print_uint_array_as_string(to_insert[i]);
            }
        } else {
            printf("Added token list: ");
            print_uint_array_as_string(to_insert[i]);
            printf(", data: {k_index: %d, v_index: %d}\n", td->k_index, td->v_index);
        }
        raxShow(t);
    }
    return failed_insertions;
}

unsigned long insert_data_to_rax_and_return_node(rax *t) {
    unsigned len = 0;
    unsigned long failed_insertions = 0;
    for (int i = 0; to_insert[i] != NULL; i++) {
        struct test_data *td = (struct test_data *)malloc(sizeof(struct test_data));
        td->k_index = i;
        td->v_index = i;
        raxNode *node = raxInsertAndReturnDataNode(t, (int *)to_insert[i], int_array_len(to_insert[i]), (void *)td, NULL);
        if (node != NULL) {
            printf("node is not null\n");
            struct test_data *new_td = (struct test_data *)malloc(sizeof(struct test_data));
            new_td->k_index = i+1;
            new_td->v_index = i+1;
            raxSetData(node,new_td);
            printf("Added token list: ");
            print_uint_array_as_string(to_insert[i]);
            printf(", new data: {k_index: %d, v_index: %d}\n", new_td->k_index, new_td->v_index);
        }
    }
    return failed_insertions;
}

// remove data from rax
void remove_data_from_rax(rax *t) {
    for (int i = 0; to_insert[i] != NULL; i++) {
        if (raxRemove(t, (int *)to_insert[i], int_array_len(to_insert[i]), NULL)) {
            printf("raxRemove success, deleted token list: ");
        } else {
            printf("raxRemove failed for token list: ");
        }
        print_uint_array_as_string(to_insert[i]);
        printf("\n");
        raxShow(t);
        printf("t->numele: %d, t->numnodes: %d\n", t->numele, t->numnodes);
    }
}

// find data in rax
void find_data_in_rax(rax *t) {
    for (int i = 0; to_find[i] != NULL; i++) {
        void *data = raxFind(t, (int *)to_find[i], int_array_len(to_find[i]));
        if (data == raxNotFound) {
            printf("Token list ");
            print_uint_array_as_string(to_find[i]);
            printf(" is not found\n");
        } else {
            test_data *td = (test_data *)data;
            printf("Token list ");
            print_uint_array_as_string(to_find[i]);
            printf(" found, data is: {k_index: %d, v_index: %d}\n", td->k_index, td->v_index);
        }
    }
}

int main() {
    rax *t = raxNew();
    if (t == NULL) return 1;

    // insert token list
    insert_data_to_rax(t);

    remove_data_from_rax(t);

    raxFree(t);
    return 0;
}
