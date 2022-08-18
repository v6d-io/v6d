#ifndef SRC_SERVER_MEMORY_COMMON_H_
#define SRC_SERVER_MEMORY_COMMON_H_

#include <sys/mount.h>
#define PAGE_SIZE 0X1000
struct mount_data {
    unsigned long this_addr;
	unsigned long vineyard_storage_user_addr;
	unsigned long page_num;
};

#endif