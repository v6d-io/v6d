#include "include/partition/partition.h"

int main(int argc, char** argv) {
    const char *a[] = {"/home/graphscope/gie-grin/temp/vineyard.sock", "130724717286376948"};
    void* h = grin_get_partitioned_graph_from_storage(2, (char**)a);
    return 0;
}