# launch vineyardd & load modern graph
# 1. copy modern_graph folder from GraphScope/charts/gie-standalone/data to build
# 2. copy v6d_modern_loader to modern_graph folder
# 3. in build folder, run: bin/vineyardd --socket=tmp.sock
# 4. in build folder, run: bin/vineyard-graph-loader --socket=tmp.sock --config modern_graph/config.json

rm -rf ./test
gcc test.c -I. -L/home/graphscope/gie-grin/v6d/build/shared-lib/ -lvineyard_grin -lvineyard_graph -lvineyard_basic -o test
./test /workspaces/v6d/build/tmp.sock 3493598830228844