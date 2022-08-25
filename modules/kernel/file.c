#include "vineyard_fs.h"
#include "vineyard_i.h"
#include "msg_mgr.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("yuansm");
MODULE_DESCRIPTION("Vineyard filesystem for Linux.");
MODULE_VERSION("0.01");

void translate_u64_to_char(uint64_t num, char *name)
{
    int i = 0, j = 0;
    int temp;
    char c;

    while(num) {
        temp = num % 10;
        num /= 10;
        name[i] = temp + '0';
        i++;
    }

    for (; j < i / 2; j++) {
        c = name[j];
        name[j] = name[i - j - 1];
        name[i - j - 1] = c;
    }
}

uint64_t translate_char_to_u64(const char *name)
{
    uint64_t ret = 0;
    int i = 0;

    while (name[i] != 0) {
        if (name[i] >= '0' && name[i] <= '9') {
            ret = ret * 10 + name[i] - '0';
            i++;
        } else {
            ret = 0;
            break;
        }
    }
    return ret;
}

static int vineyard_fs_open(struct inode *inode, struct file *file)
{
    struct vineyard_inode_info *i_info;
    printk(KERN_INFO PREFIX "fake %s\n", __func__);

    i_info = get_vineyard_inode_info(inode);
    printk(KERN_INFO PREFIX "open: %llu\n", i_info->obj_id);
    return 0;
}

extern void *vineyard_msg_mem_kernel_addr;
static ssize_t vineyard_fs_read(struct file *file, char __user *user, size_t length, loff_t *start)
{
    struct vineyard_inode_info *i_info;
    struct vineyard_request_msg *msgs;
    struct vineyard_msg_mem_header *header;
    struct vineyard_result_msg *rmsgs;
    struct vineyard_result_mem_header *rheader;
    size_t size;
    uint64_t offset;
    int ret;
    printk(KERN_INFO PREFIX "fake %s\n", __func__);

    i_info = get_vineyard_inode_info(file_inode(file));
    msgs = (struct vineyard_request_msg *)(uint64_t)vineyard_msg_buffer_addr;
    header = (struct vineyard_msg_mem_header *)vineyard_msg_mem_header;
    rmsgs = (struct vineyard_result_msg *)vineyard_result_buffer_addr;
    rheader = (struct vineyard_result_mem_header *)vineyard_result_mem_header;

    printk(KERN_INFO PREFIX "lock:%d %px\n", header->lock, &header->lock);
    vineyard_spin_lock(&header->lock);
    // TODO: ring buffer reset header point
    msgs[header->head_point].opt = VREAD;
    msgs[header->head_point]._fopt_param.obj_id = i_info->obj_id;
    msgs[header->head_point]._fopt_param.type = i_info->obj_type;
    msgs[header->head_point]._fopt_param.offset = *start;
    msgs[header->head_point]._fopt_param.length = length;
    header->head_point++;
    vineyard_spin_unlock(&header->lock);

    wake_up(&vineyard_msg_wait);
    // TODO: muti thread will failed
    wait_event_interruptible(vineyard_fs_wait, (!msg_empty(rheader->head_point, rheader->tail_point)));

    vineyard_spin_lock(&rheader->lock);
    size = rmsgs[rheader->tail_point].size;
    offset = rmsgs[rheader->tail_point].offset;
    rheader->tail_point++;
    vineyard_spin_unlock(&rheader->lock);

    printk(KERN_INFO PREFIX "size:%lu, offset:%llu\n", size, offset);
    size = size > length ? length : size;
    ret = copy_to_user(user, vineyard_storage_kernel_addr + offset, size);
    if (ret)
        return ret;
    return size;
    // return size;
}

static ssize_t vineyard_fs_write(struct file *file, const char __user *user, size_t length, loff_t *start)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return -1;
}

static int vineyard_fs_flush(struct file *file, fl_owner_t id)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return -1;
}

int vineyard_fs_release(struct inode *inode, struct file *file)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return 0;
}

int vineyard_setattr(struct user_namespace *mnt_userns, struct dentry * entry, struct iattr *attr)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return -1;
}

int vineyard_getattr(struct user_namespace *mnt_userns, const struct path *path, struct kstat *stat, u32 mask, unsigned int flags)
{
    struct inode *inode;

    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    inode = d_inode(path->dentry);
    if (!mask) {
        stat->result_mask = 0;
        return 0;
    }

    if (stat)
        generic_fillattr(mnt_userns, inode, stat);
    return 0;
}

int vineyard_update_time(struct inode *inode, struct timespec64 *now, int flags)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
	return 0;
}

static const struct file_operations vineyard_file_operations = {
    .open = vineyard_fs_open,
    .read = vineyard_fs_read,
    .write = vineyard_fs_write,
    .flush = vineyard_fs_flush,
    .release = vineyard_fs_release,
};

static const struct inode_operations vineyard_file_inode_operations = {
    .setattr        = vineyard_setattr,
    .getattr        = vineyard_getattr,
    .update_time    = vineyard_update_time,
};

void vineyard_fs_init_file_inode(struct inode *inode)
{
    printk(KERN_INFO PREFIX "%s\n", __func__);
    inode->i_fop = &vineyard_file_operations;
    inode->i_op = &vineyard_file_inode_operations;
}
