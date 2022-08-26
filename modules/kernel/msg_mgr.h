#pragma once
#include <linux/wait.h>

#define NETLINK_VINEYARD  22
#define NETLINK_PORT      100

enum OBJECT_TYPE {
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

struct fopt_param {
    // read/write/sync
    uint64_t            obj_id;
    uint64_t            offset;
    // open
    enum OBJECT_TYPE    type;
    uint64_t            length;
};

struct vineyard_request_msg {
    enum REQUEST_OPT  opt;
    struct fopt_param _fopt_param;
};

struct vineyard_kern_user_msg {
    enum USER_KERN_OPT  opt;
    uint64_t            request_mem;
    uint64_t            result_mem;
    uint64_t            obj_info_mem;
};

struct vineyard_msg_mem_header {
    int             has_msg;
    unsigned int    lock;
    int             head_point;
    int             tail_point;
    int             close;
};

struct vineyard_result_mem_header {
    int             has_msg;
    unsigned int    lock;
    int             head_point;
    int             tail_point;
};

struct vineyard_rw_lock {
    unsigned int r_lock;
    unsigned int w_lock;
};

struct vineyard_object_info_header {
    struct vineyard_rw_lock rw_lock;
    int total_file;
};

// msg_mgr.c
int net_link_init(void);
void net_link_release(void);
void vineyard_spin_lock(volatile unsigned int *addr);
void vineyard_spin_unlock(volatile unsigned int *addr);
void inline vineyard_read_lock(struct vineyard_rw_lock *rw_lock);
void inline vineyard_read_unlock(struct vineyard_rw_lock *rw_lock);
void send_exit_msg(void);
void send_request_msg(struct vineyard_request_msg *msg);
void receive_result_msg(struct vineyard_result_msg *msg);

static inline int msg_empty(int head, int tail)
{
    return head == tail;
}