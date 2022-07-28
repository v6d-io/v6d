#include "vineyard_fs.h"
#include "vineyard_i.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("yuansm");
MODULE_DESCRIPTION("Vineyard filesystem for Linux.");
MODULE_VERSION("0.01");

static int vineyard_fs_open(struct inode *inode, struct file *file)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return -1;
}

static ssize_t vineyard_fs_read(struct file *file, char __user *user, size_t length, loff_t *start)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return -1;
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
    return -1;
}

static const struct file_operations vineyard_file_operations = {
    .open = vineyard_fs_open,
    .read = vineyard_fs_read,
    .write = vineyard_fs_write,
    .flush = vineyard_fs_flush,
    .release = vineyard_fs_release,
};

void vineyard_fs_init_file_inode(struct inode *inode)
{
    printk(KERN_INFO PREFIX "%s\n", __func__);
}
