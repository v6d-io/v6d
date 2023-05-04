rm -rf ./test
gcc test.c -I. -L/home/graphscope/gie-grin/v6d/build/shared-lib/ -lvineyard_grin -lvineyard_graph -lvineyard_basic -o test
./test /home/graphscope/gie-grin/temp/vineyard.sock 130878194502466848