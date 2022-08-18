#pragma once
#include <linux/wait.h>

#define NETLINK_VINEYARD  22
#define NETLINK_PORT      100

enum REQUEST_TYPE {
  BLOB = 1,
};

enum REQUEST_OPT {
    VOPEN,
    VREAD,
    VWRITE,
    VCLOSE,
    VFSYNC,
};

enum USER_KERN_OPT {
    VSET,
    VINIT,
    VWAIT,
    VFOPT,
    VEXIT,
};

struct vineyard_result_msg {
    enum USER_KERN_OPT opt;
    uint64_t        obj_id;
    uint64_t        offset;
    uint64_t        size;
    int             ret;
};

struct vineyard_request_msg {
    enum REQUEST_OPT  opt;
    struct fopt_param {
        // read/write/sync
        uint64_t          obj_id;
        uint64_t          offset;
        // open
        enum REQUEST_TYPE type;
    } _fopt_param;
};

struct vineyard_kern_user_msg {
    enum USER_KERN_OPT  opt;
    uint64_t            request_mem;
    uint64_t            result_mem;
};

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

// kernel space
extern void *vineyard_storage_kernel_addr;
extern struct vineyard_msg_mem_header *vineyard_msg_mem_header;
extern struct vineyard_result_mem_header *vineyard_result_mem_header;
extern void *vineyard_msg_buffer_addr;
extern void *vineyard_result_buffer_addr;

// user space
extern void *vineyard_msg_mem_user_addr;
extern void *vineyard_result_mem_user_addr;

extern struct wait_queue_head vineyard_msg_wait;
extern struct wait_queue_head vineyard_fs_wait;

int net_link_init(void);
void net_link_release(void);
void vineyard_spin_lock(volatile int *addr);
void vineyard_spin_unlock(volatile int *addr);