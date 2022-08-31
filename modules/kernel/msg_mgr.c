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

#define NETLINK_VINEYARD 23
#define USER_PORT        101

#define PAGE_DOWN(x) ((x) & (~(PAGE_SIZE - 1)))
#define PAGE_UP(x)   PAGE_DOWN(x + PAGE_SIZE - 1)

struct sock *nl_socket = NULL;
extern struct net init_net;

int vineyard_connect = 0;

static struct page **pages;
static unsigned long page_num;

// vineyard bulk
void *vineyard_storage_kernel_addr = NULL;
void *vineyard_bulk_kernel_addr = NULL;

// file entry buffer
void *vineyard_object_info_kernel_addr = NULL;
struct vineyard_object_info_header *vineyard_object_info_header;
void *vineyard_object_info_buffer_addr;

void *vineyard_object_info_user_addr = NULL;

int msg_lock = 0;
static struct vineyard_request_msg global_msg = { 0 };
static struct vineyard_result_msg  global_rmsg ={ 0 } ;

DECLARE_WAIT_QUEUE_HEAD(vineyard_msg_wait);
DECLARE_WAIT_QUEUE_HEAD(vineyard_fs_wait);

// tools
static int send_msg(void)
{
    struct sk_buff *nl_skb;
    struct nlmsghdr *nlh;
    int ret;
    unsigned long int length;
    struct vineyard_msg msg;

    length = sizeof(msg);
    nl_skb = nlmsg_new(length, GFP_ATOMIC);

    if (!nl_skb) {
        printk(KERN_INFO PREFIX "Netlink alloc failure\n");
    }

    nlh = nlmsg_put(nl_skb, 0, 0, NETLINK_VINEYARD, length, 0);
    if (!nlh) {
        printk(KERN_INFO PREFIX "nlmsg_put failure\n");
        nlmsg_free(nl_skb);
        return -1;
    }

    memcpy(&msg.msg.request, &global_msg, sizeof(struct vineyard_request_msg));
    global_msg.has_msg = 0;
    memcpy(nlmsg_data(nlh), &msg, sizeof(msg));
    printk(KERN_INFO "opt:%d\n", global_msg.opt);

    ret = netlink_unicast(nl_socket, nl_skb, USER_PORT, MSG_DONTWAIT);

    return ret;
}

static int map_vineyard_user_storage_buffer(unsigned long user_addr)
{
	struct vm_area_struct *user_vma;
	int i;
	int ret;

	printk(KERN_INFO PREFIX "%s\n", __func__);
	pages = vmalloc(page_num * sizeof(struct page *));
	memset(pages, 0, sizeof(struct page *) * page_num);

	user_vma = find_vma(current->mm, user_addr);
	printk(KERN_INFO "addr: %lx\n", user_addr);
	ret = get_user_pages_fast(user_addr, page_num, 1, pages);
	if (ret < 0) {
		printk(KERN_INFO "pined page error %d\n", ret);
		vfree(pages);
		return -1;
	}

	vineyard_storage_kernel_addr = vmap(pages, page_num, VM_MAP, PAGE_KERNEL);

	// test write data
	if (vineyard_storage_kernel_addr) {
		for (i = 0; i < page_num * PAGE_SIZE / sizeof(int); i++) {
			((int *)vineyard_storage_kernel_addr)[i] = i;
		}
		printk(KERN_INFO PREFIX "number:%lu\n", page_num * PAGE_SIZE / sizeof(int));
	} else {
		printk(KERN_INFO PREFIX "map error!\n");
		for (i = 0; i < ret; i++) {
			put_page(pages[i]);
		}
		vfree(pages);
		return -1;
	}

	return 0;
}

static unsigned long prepare_user_buffer(void **kernel_addr, size_t size)
{
	unsigned long user_addr;
	struct vm_area_struct *vma;

	*kernel_addr = kzalloc(size, GFP_KERNEL);
	if (!(*kernel_addr)) {
		printk(KERN_INFO PREFIX "kmalloc error!\n");
		return 0;
	}

	user_addr = vm_mmap(NULL, 0, size,
			    PROT_READ | PROT_WRITE,
			    MAP_SHARED, 0);
	vma = find_vma(current->mm, user_addr);
	remap_pfn_range(vma, user_addr, PFN_DOWN(__pa(*kernel_addr)), size, vma->vm_page_prot);
	printk(KERN_INFO PREFIX "kern:%px\n", kernel_addr);

	return user_addr;
}

static void unmap_vineyard_msg_result_buffer(void)
{
	kfree(vineyard_object_info_kernel_addr);
}

static void unmap_vineyard_user_storage_buffer(void)
{
	int i;

	printk(KERN_INFO PREFIX "%s\n", __func__);
	if (vineyard_storage_kernel_addr) {
		printk(KERN_INFO "unmap and put pages\n");
		vunmap(vineyard_storage_kernel_addr);
		for (i = 0; i < page_num; i++) {
			put_page(pages[i]);
		}
		vfree(pages);
		vineyard_storage_kernel_addr = NULL;
	}
}

static void init_object_info_buffer(void)
{
    vineyard_object_info_user_addr = (void *)prepare_user_buffer(&vineyard_object_info_kernel_addr, PAGE_SIZE);
    vineyard_object_info_header = vineyard_object_info_kernel_addr;
    vineyard_object_info_buffer_addr = vineyard_object_info_kernel_addr + sizeof(struct vineyard_object_info_header);
    global_msg.param._set_param.obj_info_mem = (uint64_t)vineyard_object_info_user_addr;
}

static void handle_wait(void)
{
    int ret;

    if (unlikely(vineyard_connect == 0))
        vineyard_connect = 1;

    do {
        ret = wait_event_interruptible(vineyard_msg_wait, global_msg.has_msg);
    } while (ret != 0);

    if (unlikely(global_msg.opt == VINEYARD_EXIT)) {
        vineyard_connect = 0;
        unmap_vineyard_user_storage_buffer();
	    unmap_vineyard_msg_result_buffer();
        vineyard_spin_unlock(&msg_lock);
    } else if (unlikely(global_msg.opt == VINEYARD_MOUNT)) {
        init_object_info_buffer();
    }

    send_msg();
    printk(KERN_INFO PREFIX "wait end!\n");
}

static void handle_set_bulk(struct vineyard_result_msg *rmsg)
{
    int err;
    uint64_t bulk_addr;
    uint64_t bulk_size;

    bulk_addr = rmsg->ret._set_ret.bulk_addr;
    bulk_size = rmsg->ret._set_ret.bulk_size;
    printk(KERN_INFO PREFIX "%llx %llx %d\n", bulk_addr, bulk_size, rmsg->ret._set_ret.ret);

    page_num = (PAGE_UP(bulk_addr + bulk_size) - PAGE_DOWN(bulk_addr)) / PAGE_SIZE;
    printk(KERN_INFO PREFIX "page_num:%lu size:%lu\n", page_num, page_num * 0x1000);
    err = map_vineyard_user_storage_buffer(PAGE_DOWN(bulk_addr));

    vineyard_bulk_kernel_addr = bulk_addr - PAGE_DOWN(bulk_addr) + vineyard_storage_kernel_addr;
    printk(KERN_INFO PREFIX "err:%d\n", err);

    memcpy(&global_rmsg, rmsg, sizeof(*rmsg));
    global_rmsg.has_msg = 1;
    global_rmsg.ret._set_ret.ret |= err;
    wake_up(&vineyard_fs_wait);
    handle_wait();
}

static void handle_fopt(struct vineyard_result_msg *rmsg)
{
    printk(KERN_INFO PREFIX "%s\n", __func__);
    memcpy(&global_rmsg, rmsg, sizeof(*rmsg));
    global_rmsg.has_msg = 1;
    //1. send handler result to user
    wake_up(&vineyard_fs_wait);
    //2. call handler_wait
    handle_wait();
}

static void netlink_rcv_msg(struct sk_buff *skb)
{
    struct nlmsghdr *nlh = NULL;
    // struct vineyard_kern_user_msg *umsg = NULL;
    struct vineyard_msg *rmsg;

    if (skb->len >= nlmsg_total_size(0)) {
        nlh = nlmsg_hdr(skb);
        rmsg = (struct vineyard_msg *)NLMSG_DATA(nlh);
        if (rmsg)
        {
            printk(KERN_INFO PREFIX "Kernel recv from user %s\n", (char *)rmsg);

            switch(rmsg->msg.result.opt) {
            case VINEYARD_WAIT:
                printk(KERN_INFO PREFIX "Receive opt: WAIT\n");
                handle_wait();
                break;
            case VINEYARD_SET_BULK_ADDR:
                printk(KERN_INFO PREFIX "Receive opt: VWAIT\n");
                handle_set_bulk(&rmsg->msg.result);
                break;
            case VINEYARD_OPEN:
            case VINEYARD_READ:
            case VINEYARD_WRITE:
            case VINEYARD_CLOSE:
            case VINEYARD_FSYNC:
                printk(KERN_INFO PREFIX "Receive opt: VFOPT\n");
                handle_fopt(&rmsg->msg.result);
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
            printk(KERN_INFO "There must be something wrong?\n");
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

int send_request_msg(struct vineyard_request_msg *msg)
{
    if (!vineyard_connect) {
        return -1;
    }
    vineyard_spin_lock(&msg_lock);
    memcpy(&global_msg, msg, sizeof(*msg));
    global_msg.has_msg = 1;
    wake_up(&vineyard_msg_wait);

    return 0;
}

void receive_result_msg(struct vineyard_result_msg *rmsg)
{
    int ret;

    do {
        ret = wait_event_interruptible(vineyard_fs_wait, global_rmsg.has_msg);
    } while(ret != 0);

    memcpy(rmsg, &global_rmsg, sizeof(*rmsg));
    global_rmsg.has_msg = 0;
    vineyard_spin_unlock(&msg_lock);
}
