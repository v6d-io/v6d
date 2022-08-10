#pragma once
#include <linux/wait.h>

struct vineyard_msg_mem_header {
    int     has_msg;
    int     lock;
    int     head_point;
    int     tail_point;
    int     close;
};

struct vineyard_result_mem_header {
    int     has_msg;
    int     lock;
    int     head_point;
    int     tail_point;
};

extern void *vineyard_storage_kernel_addr;
extern struct vineyard_msg_mem_header *vineyard_msg_mem_header;
extern struct vineyard_result_mem_header *vineyard_result_mem_header;

extern void *vineyard_msg_buffer_addr;
extern void *vineyard_result_buffer_addr;

extern struct wait_queue_head vineyard_msg_wait;
extern struct wait_queue_head vineyard_fs_wait;

int net_link_init(void);
void net_link_release(void);
void vineyard_spin_lock(volatile int *addr);
void vineyard_spin_unlock(volatile int *addr);