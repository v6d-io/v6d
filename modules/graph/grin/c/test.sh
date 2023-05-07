rm -rf ./test
gcc test.c -I. -L/home/graphscope/gie-grin/v6d/build/shared-lib/ -lvineyard_grin -lvineyard_graph -lvineyard_basic -o test
./test /workspaces/v6d/build/tmp.sock 3493598830228844