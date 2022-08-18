#include <linux/string.h>
#include <linux/futex.h>
#include <linux/wait.h>
#include "vineyard_fs.h"
#include "vineyard_i.h"
#include "msg_mgr.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("yuansm");
MODULE_DESCRIPTION("Vineyard filesystem for Linux.");
MODULE_VERSION("0.01");

static inline int vineyard_get_entry(struct file *dir_file, int *cpos)
{
    // TODO: communicate with vineyard
    // temp code for test
    int ret = 0;
    if ((*cpos) == 2) {
        ret = 0;
    } else {
        ret = -1;
    }

    return ret;
}

static int vineyard_fs_dir_read(struct file *file, struct dir_context *ctx)
{
    int ret;
    struct dentry *dentry;
    const unsigned char *path;
    struct inode *temp, *dir_i;
    struct super_block *sb;
    int cpos = 0;
    ret = 0;

    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    dir_i = file_inode(file);
    sb = dir_i->i_sb;
    dentry = dget(file->f_path.dentry);
    if (!dentry)
        return -ENOENT;

    path = dentry->d_name.name;
    printk(KERN_INFO PREFIX "%s\n", path);
    if (strcmp(path, "/") != 0) {
        printk(KERN_INFO PREFIX "oh... wrong path!\n");
        dput(dentry);
        return -ENOENT;
    }

    cpos = ctx->pos;
    ret = dir_emit_dots(file, ctx);
    

get_new:
    if (vineyard_get_entry(file, &cpos) == -1) {
        goto out;
    }

    // temp code for test
    if (cpos == 2) {
        temp = vineyard_fs_build_inode(sb);
        if (temp) {
            dir_emit(ctx, "a", cpos + 1, temp->i_ino, DT_REG);
        }
        cpos++;
    }

    ctx->pos = cpos;
    goto get_new;

out:
    dput(dentry);

    // for test
    // if (!list_empty(&vineyard_msg_wait.head)) {
    //     vineyard_spin_lock(&vineyard_msg_mem_header->lock);
    //     vineyard_msg_mem_header->has_msg = 1;
    //     vineyard_spin_unlock(&vineyard_msg_mem_header->lock);

    //     wake_up(&vineyard_msg_wait);
    //     printk(KERN_INFO PREFIX "readdir sleep\n");
    //     ret = wait_event_interruptible(vineyard_fs_wait, vineyard_result_mem_header->has_msg);
    //     if (ret == -ERESTARTSYS) {
    //         return 0;
    //     }
    //     printk(KERN_INFO PREFIX "readdir awake\n");

    //     vineyard_spin_lock(&vineyard_result_mem_header->lock);
    //     vineyard_result_mem_header->has_msg = 0;
    //     vineyard_spin_unlock(&vineyard_result_mem_header->lock);
    // }
    return 0;
}

// static int vineyard_fs_dir_release(struct inode *inode, struct file *file)
// {
//     printk(KERN_INFO PREFIX "fake %s\n", __func__);
//     return -1;
// }

static int vineyard_fs_dir_fsync(struct file *file, loff_t start, loff_t end, int datasync)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return -1;
}

static struct dentry *vineyard_fs_dir_lookup(struct inode *inode, struct dentry *dentry, unsigned int flags)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return NULL;
}

static int vineyard_fs_dir_setattr(struct user_namespace *mnt_userns, struct dentry * entry, struct iattr *attr)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return -1;
}

static int vineyard_fs_dir_create(struct user_namespace *mnt_userns, struct inode *dir, struct dentry *entry, umode_t mode, bool excl)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return -1;
}

static int vineyard_fs_dir_permission(struct user_namespace *mnt_userns, struct inode *inode, int mask)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return 0;
}

static int vineyard_fs_dir_getattr(struct user_namespace *mnt_userns, const struct path *path, struct kstat *stat, u32 mask, unsigned int flags)
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

static ssize_t vineyard_fs_dir_listxattr(struct dentry *entry, char *list, size_t size)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return 0;
}

static int vineyard_fs_dir_fileattr_get(struct dentry *entry, struct fileattr *fa)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return -1;
}

static int vineyard_fs_dir_fileattr_set(struct user_namespace *mnt_userns,struct dentry *dentry, struct fileattr *fa)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);
    return -1;
}

static const struct file_operations vineyard_fs_dir_operations = {
	.iterate	= vineyard_fs_dir_read,
	.fsync		= vineyard_fs_dir_fsync,
    .read       = generic_read_dir,
};

static const struct inode_operations vineyard_dir_inode_operations = {
	.lookup		= vineyard_fs_dir_lookup,
	.setattr	= vineyard_fs_dir_setattr,
	.create		= vineyard_fs_dir_create,
	.permission	= vineyard_fs_dir_permission,
	.getattr	= vineyard_fs_dir_getattr,
	.listxattr	= vineyard_fs_dir_listxattr,
	.fileattr_get	= vineyard_fs_dir_fileattr_get,
	.fileattr_set	= vineyard_fs_dir_fileattr_set,
};

void vineyard_fs_init_dir_inode(struct inode *inode)
{
    printk(KERN_INFO PREFIX "%s\n", __func__);
    inode->i_op = &vineyard_dir_inode_operations;
    inode->i_fop = &vineyard_fs_dir_operations;
}