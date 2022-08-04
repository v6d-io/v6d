BENCH_CPP_FLAGS 		:= -std=c++11

BENCH_CPP_FLAGS 		+= -I../../src
BENCH_CPP_FLAGS 		+= -I../../modules
BENCH_CPP_FLAGS 		+= -I../../thirdparty
BENCH_CPP_FLAGS 		+= -I../../thirdparty/ctti/include

BENCH_CPP_FLAGS 		+= -L../../build/shared-lib
BENCH_CPP_FLAGS 		+= -L../../build/static-lib
BENCH_CPP_FLAGS 		+= -lvineyard_client
BENCH_CPP_FLAGS 		+= -lvineyard_malloc
BENCH_CPP_FLAGS 		+= -lmimalloc
BENCH_CPP_FLAGS 		+= -lglog

BENCH_CPP_FLAGS 		+= -lpthread
DEBUG_CPP_FLAGS			:= -g -ggdb -O0
RELEASE_CPP_FLAGS		:= -O2 -DNDEBUG

ifeq ($(DEBUG), true)
	BENCH_CPP_FLAGS		+= $(DEBUG_CPP_FLAGS)
	SUFFIX				:= _dbg
else
	BENCH_CPP_FLAGS		+= $(RELEASE_CPP_FLAGS)
	SUFFIX				:= 
endif

DIST_BIN_DIR			:= bin/

all: bench_system bench_jemalloc bench_mimalloc bench_vineyard bench_vineyard_arena

dist:
	mkdir -p $(DIST_BIN_DIR)
.PHONY: dist

clean:
	rm -rf $(DIST_BIN_DIR)
.PHONY: clean

bench_system: dist alloc_test.h bench_allocator.cpp
	g++ bench_allocator.cpp -D BENCH_SYSTEM -o $(DIST_BIN_DIR)/bench_allocator_system$(SUFFIX) $(BENCH_CPP_FLAGS)

bench_jemalloc: dist alloc_test.h bench_allocator.cpp
	g++ bench_allocator.cpp -D BENCH_JEMALLOC -o $(DIST_BIN_DIR)/bench_allocator_jemalloc$(SUFFIX) $(BENCH_CPP_FLAGS)

bench_mimalloc: dist alloc_test.h bench_allocator.cpp
	g++ bench_allocator.cpp -D BENCH_MIMALLOC -o $(DIST_BIN_DIR)/bench_allocator_mimalloc$(SUFFIX) $(BENCH_CPP_FLAGS)

bench_vineyard: dist alloc_test.h bench_allocator.cpp
	g++ bench_allocator.cpp -D BENCH_VINEYARD -o $(DIST_BIN_DIR)/bench_allocator_vineyard$(SUFFIX) $(BENCH_CPP_FLAGS)

bench_vineyard_arena: dist alloc_test.h bench_allocator.cpp
	g++ bench_allocator.cpp -D BENCH_VINEYARD_ARENA -o $(DIST_BIN_DIR)/bench_allocator_vineyard_arena$(SUFFIX) $(BENCH_CPP_FLAGS)
