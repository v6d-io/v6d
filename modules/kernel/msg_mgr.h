#pragma once
#include <linux/wait.h>

enum OBJECT_TYPE {
  BLOB = 1,
};

enum MSG_OPT {
    VINEYARD_WAIT,
    VINEYARD_MOUNT,
    VINEYARD_SET_BULK_ADDR,
    VINEYARD_EXIT,
    VINEYARD_OPEN,
    VINEYARD_READ,
    VINEYARD_WRITE,
    VINEYARD_CLOSE,
    VINEYARD_FSYNC,
    VINEYARD_READDIR,
};

struct fopt_ret {
    uint64_t    obj_id;
    uint64_t    offset;
    uint64_t    size;
    int         ret;
};

struct set_ret {
    uint64_t    bulk_addr;
    uint64_t    bulk_size;
    int         ret;
};

struct vineyard_result_msg {
    enum MSG_OPT    opt;
    int             has_msg;
    union {
        struct fopt_ret _fopt_ret;
        struct set_ret  _set_ret;
    } ret;
};

struct fopt_param {
    // read/write/sync
    uint64_t            obj_id;
    uint64_t            offset;
    // open
    enum OBJECT_TYPE    type;
    uint64_t            length;
};

struct set_param {
    uint64_t    obj_info_mem;
};

struct vineyard_request_msg {
    enum MSG_OPT    opt;
    int             has_msg;
    union {
        struct fopt_param   _fopt_param;
        struct set_param    _set_param;
    } param;
};

struct vineyard_rw_lock {
    unsigned int r_lock;
    unsigned int w_lock;
};

struct vineyard_object_info_header {
    struct vineyard_rw_lock rw_lock;
    int total_file;
};

struct vineyard_msg {
    union {
        struct vineyard_request_msg request;
        struct vineyard_result_msg  result;
    } msg;
};

extern int vineyard_connect;

// msg_mgr.c
int net_link_init(void);
void net_link_release(void);
void vineyard_spin_lock(volatile unsigned int *addr);
void vineyard_spin_unlock(volatile unsigned int *addr);
void inline vineyard_read_lock(struct vineyard_rw_lock *rw_lock);
void inline vineyard_read_unlock(struct vineyard_rw_lock *rw_lock);
void send_exit_msg(void);
int send_request_msg(struct vineyard_request_msg *msg);
int receive_result_msg(struct vineyard_result_msg *msg);
