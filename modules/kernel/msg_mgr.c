#include <linux/stat.h>
#include <linux/atmioc.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/netlink.h>
#include <net/sock.h>
#include "vineyard_fs.h"
#include "vineyard_i.h"
#include "msg_mgr.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("yuansm");
MODULE_DESCRIPTION("Vineyard filesystem for Linux.");
MODULE_VERSION("0.01");

#define NETLINK_VINEYARD 22
#define USER_PORT        100

struct sock *nl_socket = NULL;
extern struct net init_net;

DECLARE_WAIT_QUEUE_HEAD(vineyard_msg_wait);
DECLARE_WAIT_QUEUE_HEAD(vineyard_fs_wait);

// tools
static int send_msg(void *pbuf, uint16_t len)
{
    struct sk_buff *nl_skb;
    struct nlmsghdr *nlh;
    int ret;

    nl_skb = nlmsg_new(len, GFP_ATOMIC);
    if (!nl_skb) {
        printk(KERN_INFO PREFIX "Netlink alloc failure\n");
    }

    nlh = nlmsg_put(nl_skb, 0, 0, NETLINK_VINEYARD, len, 0);
    if (!nlh) {
        printk(KERN_INFO PREFIX "nlmsg_put failure\n");
        nlmsg_free(nl_skb);
        return -1;
    }
    memcpy(nlmsg_data(nlh), pbuf, len);
    ret = netlink_unicast(nl_socket, nl_skb, USER_PORT, MSG_DONTWAIT);

    return ret;
}

static inline int msg_full(int head, int tail, int capacity)
{
    if (tail == capacity) {
        return head == 0;
    }
    return head == (tail - 1);
} 

static void handle_init(void)
{
    struct vineyard_kern_user_msg msg;

    msg.opt = VSET;
    msg.request_mem = (uint64_t)vineyard_msg_mem_user_addr;
    msg.result_mem = (uint64_t)vineyard_result_mem_user_addr;
    msg.obj_info_mem = (uint64_t)vineyard_object_info_user_addr;

    send_msg(&msg, sizeof(msg));
}

static void handle_wait(void)
{
    int ret;
    struct vineyard_msg_mem_header *header;
    struct vineyard_kern_user_msg msg;

    header = (struct vineyard_msg_mem_header *)vineyard_msg_mem_header;

    // It is broken up by signal when the return value is negative.
    do {
        ret = wait_event_interruptible(vineyard_msg_wait,
                                       !msg_empty(header->head_point,header->tail_point));
    } while (ret != 0);

    if (unlikely(header->close)) {
        msg.opt = VEXIT;
    } else {
        msg.opt = VFOPT;
    }
    send_msg(&msg, sizeof(msg));
    printk(KERN_INFO PREFIX "wait end!\n");
}

static void handle_fopt(void)
{
    printk(KERN_INFO PREFIX "%s\n", __func__);
    //1. send handler result to user
    wake_up(&vineyard_fs_wait);
    //2. call handler_wait
    handle_wait();
}

static void netlink_rcv_msg(struct sk_buff *skb)
{
    struct nlmsghdr *nlh = NULL;
    struct vineyard_kern_user_msg *umsg = NULL;

    if (skb->len >= nlmsg_total_size(0)) {
        nlh = nlmsg_hdr(skb);
        umsg = (struct vineyard_kern_user_msg *)NLMSG_DATA(nlh);
        if (umsg)
        {
            printk(KERN_INFO PREFIX "Kernel recv from user %s\n", (char *)umsg);

            switch(umsg->opt) {
            case VINIT:
                printk(KERN_INFO PREFIX "Receive opt: VINT\n");
                handle_init();
                break;
            case VWAIT:
                printk(KERN_INFO PREFIX "Receive opt: VWAIT\n");
                handle_wait();
                break;
            case VFOPT:
                printk(KERN_INFO PREFIX "Receive opt: VFOPT\n");
                handle_fopt();
                break;
            default:
                printk(KERN_INFO PREFIX "error!\n");
                break;
            }
        }
    }
}

struct netlink_kernel_cfg cfg = {
    .input = netlink_rcv_msg,
};

int net_link_init(void)
{
    nl_socket = (struct sock *)netlink_kernel_create(&init_net, NETLINK_VINEYARD, &cfg);
    if (!nl_socket) {
        printk(KERN_INFO PREFIX "Create netlink error!\n");
        return -1;
    }

    return 0;
}

void net_link_release(void)
{
    if (nl_socket) {
        netlink_kernel_release(nl_socket);
        nl_socket = NULL;
    }
}

// interfaces
static int count = 0;
void vineyard_spin_lock(volatile unsigned int *addr)
{
    while(!(__sync_bool_compare_and_swap(addr, 0, 1))) {
        // for test dead lock
        if (count == 80) {
            printk(KERN_INFO "what the fuck?\n");
            break;
        }
        count++;
    }
}

void vineyard_spin_unlock(volatile unsigned int *addr)
{
    *addr = 0;
}

void inline vineyard_read_lock(struct vineyard_rw_lock *rw_lock)
{
    vineyard_spin_lock(&rw_lock->w_lock);
    rw_lock->r_lock++;
    printk(KERN_INFO PREFIX "now there exist(s) %d readder\n", rw_lock->r_lock);
    vineyard_spin_unlock(&rw_lock->w_lock);
}

void inline vineyard_read_unlock(struct vineyard_rw_lock *rw_lock)
{
    rw_lock->r_lock--;
    printk(KERN_INFO PREFIX "now there exist(s) %d readder\n", rw_lock->r_lock);
}

void send_request_msg(struct vineyard_request_msg *msg)
{
    struct vineyard_request_msg *msgs;
    struct vineyard_msg_mem_header *header;
    msgs = (struct vineyard_request_msg *)(uint64_t)vineyard_msg_buffer_addr;
    header = (struct vineyard_msg_mem_header *)vineyard_msg_mem_header;

    vineyard_spin_lock(&header->lock);
    // TODO: ring buffer reset header point, judge that if the buffer is full.
    memcpy(&msgs[header->head_point], msg, sizeof(*msg));
    header->head_point++;
    vineyard_spin_unlock(&header->lock);

    wake_up(&vineyard_msg_wait);
}

void send_exit_msg(void)
{
    struct vineyard_msg_mem_header *header;
    header = (struct vineyard_msg_mem_header *)vineyard_msg_mem_header;

    vineyard_spin_lock(&header->lock);
    // TODO: ring buffer reset header point, judge that if the buffer is full.
    header->close = 1;
    header->head_point++;
    vineyard_spin_unlock(&header->lock);

    wake_up(&vineyard_msg_wait);
}

void receive_result_msg(struct vineyard_result_msg *rmsg)
{
    struct vineyard_result_msg *rmsgs;
    struct vineyard_result_mem_header *rheader;
    rmsgs = (struct vineyard_result_msg *)vineyard_result_buffer_addr;
    rheader = (struct vineyard_result_mem_header *)vineyard_result_mem_header;

    // TODO: muti thread will failed
    wait_event_interruptible(vineyard_fs_wait, (!msg_empty(rheader->head_point, rheader->tail_point)));

    vineyard_spin_lock(&rheader->lock);
    memcpy(rmsg, &rmsgs[rheader->tail_point], sizeof(*rmsg));
    printk(KERN_INFO PREFIX "%s %llu", __func__, rmsgs[rheader->tail_point].offset);
    rheader->tail_point++;
    vineyard_spin_unlock(&rheader->lock);
}
