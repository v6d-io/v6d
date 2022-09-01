#include <linux/stat.h>
#include <linux/atmioc.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/futex.h>
#include <linux/wait.h>
#include "vineyard_fs.h"
#include "vineyard_i.h"
#include "msg_mgr.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("yuansm");
MODULE_DESCRIPTION("Vineyard filesystem for Linux.");
MODULE_VERSION("0.01");
MODULE_ALIAS_FS("vineyardfs");

struct vineyard_fs_context {
    int place_holder;
};

const struct dentry_operations vineyard_root_dentry_operations = { };

static const struct export_operations vineyard_export_operations = { };

static void vineyard_fs_init_inode(struct inode *inode, struct vineyard_attr *attr)
{
	printk(KERN_INFO PREFIX "%s\n", __func__);
    inode->i_mode = attr->mode;
	inode->i_ino = attr->ino;
	inode->i_size = attr->size;

	if (S_ISREG(inode->i_mode)) {
		printk(KERN_INFO PREFIX "Hello! File inode\n");
		vineyard_fs_init_file_inode(inode);
	} else if (S_ISDIR(inode->i_mode)) {
		printk(KERN_INFO PREFIX "Hello! Dir inode\n");
		vineyard_fs_init_dir_inode(inode);
	} else {
		printk(KERN_INFO PREFIX "What the fucking type :%d in %s: %d\n", inode->i_mode, __func__, __LINE__);
	}
}

static struct inode *vineyard_fs_iget(struct super_block *sb, struct vineyard_attr *attr)
{
	struct inode *inode;

	inode = new_inode(sb);

	if (inode)
		vineyard_fs_init_inode(inode, attr);

	return inode;
}

static struct inode *vineyard_get_root_inode(struct super_block *sb)
{
    struct vineyard_attr attr;

	printk(KERN_INFO PREFIX "%s\n", __func__);
    memset(&attr, 0, sizeof(attr));

    attr.mode = S_IFDIR | S_IRUSR | S_IRGRP | S_IROTH;
    attr.ino = 1;

	return vineyard_fs_iget(sb, &attr);
}

static struct inode *vineyard_fs_search_inode(struct super_block *sb, const char *name)
{
	struct vineyard_sb_info *sbi;
	struct vineyard_inode_info *i_info;

	printk(KERN_INFO PREFIX "%s\n", __func__);
	sbi = VINEYARD_SB_INFO(sb);
	list_for_each_entry(i_info, &sbi->inode_list_head, inode_list_node) {
		if (i_info->obj_id == translate_char_to_u64(name)) {
			printk(KERN_INFO PREFIX "find:%llu\n", i_info->obj_id);
			return &i_info->vfs_inode;
		}
	}
	return NULL;
}

static void vineyard_attach_node(struct super_block *sb, struct inode *inode)
{
	struct vineyard_sb_info *sbi;
	printk(KERN_INFO PREFIX "%s\n", __func__);

	sbi = sb->s_fs_info;
	list_add(&get_vineyard_inode_info(inode)->inode_list_node, &sbi->inode_list_head);
}

// vineyard inode operations
static struct inode *vineyard_alloc_inode(struct super_block *sb)
{
	struct vineyard_inode_info *info;
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	info = kzalloc(sizeof(struct inode), GFP_KERNEL);
	if (!info) {
		return NULL;
	}
	INIT_LIST_HEAD(&info->inode_list_node);
	return &info->vfs_inode;
}

static void vineyard_free_inode(struct inode *inode)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	return kfree(inode);
}

static int vineyard_write_inode(struct inode *inode, struct writeback_control *wbc)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	return 0;
}

static void vineyard_evict_inode(struct inode *inode)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
}

static void vineyard_put_super(struct super_block * sb)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
}

static int vineyard_statfs(struct dentry *d_entry, struct kstatfs *kstatfs)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	return 0;
}

static int vineyard_remount(struct super_block *sb, int *a, char *b)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	return -1;
}

static const struct super_operations vineyard_super_ops = {
	.alloc_inode	= vineyard_alloc_inode,
	.free_inode	= vineyard_free_inode,
	.write_inode	= vineyard_write_inode,
	.evict_inode	= vineyard_evict_inode,
	.put_super	= vineyard_put_super,
	.statfs		= vineyard_statfs,
	.remount_fs	= vineyard_remount,
};

// vineyard xattr operations
static int vineyard_xattr_get(const struct xattr_handler *handler,
			 struct dentry *dentry, struct inode *inode,
			 const char *name, void *value, size_t size)
{
    printk(KERN_INFO PREFIX "%s\n", __func__);
    return -1;
}

static int vineyard_xattr_set(const struct xattr_handler *handler,
			  struct user_namespace *mnt_userns,
			  struct dentry *dentry, struct inode *inode,
			  const char *name, const void *value, size_t size,
			  int flags)
{
    printk(KERN_INFO PREFIX "%s\n", __func__);
    return -1;
}

static const struct xattr_handler vineyard_xattr_handler = {
	.prefix = "vineyardfs",
	.get    = vineyard_xattr_get,
	.set    = vineyard_xattr_set,
};

const struct xattr_handler *vineyard_xattr_handlers[] = {
	&vineyard_xattr_handler,
	NULL
};

static void vineyard_sb_defaults(struct super_block *sb)
{
	sb->s_magic = VINEYARD_SUPER_MAGIC;
	sb->s_op = &vineyard_super_ops;
	sb->s_xattr = vineyard_xattr_handlers;
	sb->s_maxbytes = MAX_LFS_FILESIZE;
	sb->s_time_gran = 1;
	sb->s_export_op = &vineyard_export_operations;
	sb->s_iflags |= SB_I_IMA_UNVERIFIABLE_SIGNATURE;
	if (sb->s_user_ns != &init_user_ns)
		sb->s_iflags |= SB_I_UNTRUSTED_MOUNTER;
	sb->s_flags &= ~(SB_NOSEC | SB_I_VERSION);
}

static int vineyard_set_super_block(struct super_block *sb, struct fs_context *fsc)
{
    struct vineyard_fs_context *ctx;

    printk(KERN_INFO PREFIX "%s\n", __func__);

	ctx = fsc->fs_private;
    vineyard_sb_defaults(sb);

    sb->s_blocksize = PAGE_SIZE;
    sb->s_blocksize_bits = PAGE_SHIFT;

    return 0;
}

// vineayrd fs context operations
static void vineyard_free_fsc(struct fs_context *fsc)
{
    struct vineyard_fs_context *ctx;

    printk(KERN_INFO PREFIX "%s\n", __func__);
	ctx = fsc->fs_private;
    kfree(ctx);
}

static int vineyard_parse_param(struct fs_context *fsc, struct fs_parameter *param)
{
	printk(KERN_INFO "fake %s\n", __func__);
	return 0;
}

static int vineyard_parse_monolithic(struct fs_context *fsc, void *data)
{
    printk(KERN_INFO PREFIX "fake %s\n", __func__);

    return 0;
}

static void vineyard_init_sbi(struct vineyard_sb_info *sbi)
{
	INIT_LIST_HEAD(&sbi->inode_list_head);
}

static int vineyard_get_tree(struct fs_context *fsc)
{
    struct vineyard_request_msg msg;
    struct vineyard_result_msg rmsg;

	struct super_block *sb;
	struct inode *root;
	struct dentry *root_dentry;
	struct vineyard_sb_info *sbi;
    int err = 0;

    printk(KERN_INFO PREFIX "%s\n", __func__);

	msg.opt = VINEYARD_MOUNT;
	err = send_request_msg(&msg);
	if (err)
		return err;

	receive_result_msg(&rmsg);

	if (rmsg.ret._set_ret.ret != 0) {
		return rmsg.ret._set_ret.ret;
	}

	if (!vineyard_bulk_kernel_addr) {
		printk(KERN_INFO PREFIX "Vineyard server is not start up! Please retry later!\n");
		return -1;
	}
    sb = sget_fc(fsc, NULL, vineyard_set_super_block);
	sbi = kzalloc(sizeof(struct vineyard_sb_info), GFP_KERNEL);
	vineyard_init_sbi(sbi);
	sb->s_fs_info = sbi;

    root = vineyard_get_root_inode(sb);
    sb->s_d_op = &vineyard_root_dentry_operations;
    root_dentry = d_make_root(root);
    sb->s_root = root_dentry;
	fsc->root = dget(sb->s_root);
	sbi->vineyard_inode = root;

	printk(KERN_INFO PREFIX "%s end\n", __func__);
    return err;
}

static const struct fs_context_operations vineyard_context_ops = {
	.free				= vineyard_free_fsc,
	.parse_monolithic	= vineyard_parse_monolithic,
	.parse_param		= vineyard_parse_param,
	.get_tree			= vineyard_get_tree,
};

// vineyard file system type operations.
static int vineyard_init_fs_context(struct fs_context *fsc)
{
	struct vineyard_fs_context *ctx;

    printk(KERN_INFO PREFIX "%s\n", __func__);

	ctx = kzalloc(sizeof(struct vineyard_fs_context), GFP_KERNEL);
	if (!ctx)
		return -ENOMEM;

	fsc->fs_private = ctx;
	fsc->ops = &vineyard_context_ops;
	return 0;
}

static void vineyard_kill_sb(struct super_block *sb)
{
	struct vineyard_request_msg msg;
	printk(KERN_INFO PREFIX "fake %s\n", __func__);

	msg.opt = VINEYARD_EXIT;
	send_request_msg(&msg);

	kfree(sb->s_fs_info);
}

static struct file_system_type vineyard_fs_type = {
	.owner		= THIS_MODULE,
	.name		= "vineyardfs",
    .init_fs_context = vineyard_init_fs_context,
	.kill_sb	= vineyard_kill_sb,
	.fs_flags	= 0,
};

// interfaces
uint64_t get_next_vineyard_ino(void)
{
	static volatile uint64_t ino = 1;
	ino = __sync_fetch_and_add(&ino, 1);
	return ino;
}

struct vineyard_inode_info *get_vineyard_inode_info(struct inode *inode)
{
	return container_of(inode, struct vineyard_inode_info, vfs_inode);
}

struct inode *vineyard_fs_build_inode(struct super_block *sb, const char *name)
{
	struct inode *inode;
	struct vineyard_attr attr;
	struct vineyard_inode_info *i_info;
	struct vineyard_sb_info *sbi;

	printk(KERN_INFO PREFIX "%s\n", __func__);
	inode = vineyard_fs_search_inode(sb, name);
	if (inode)
		goto out;

	sbi = sb->s_fs_info;
	// TODO: We suppose that the file in vineyard is all regular file.
	attr.mode = S_IFREG | S_IRUSR | S_IRGRP | S_IROTH;
	attr.ino = get_next_vineyard_ino();

	inode = vineyard_fs_iget(sb, &attr);

	vineyard_attach_node(sb, inode);
	i_info = get_vineyard_inode_info(inode);
	i_info->obj_id = translate_char_to_u64(name);
	printk(KERN_INFO PREFIX "trans id:%llu\n", i_info->obj_id);
	printk(KERN_INFO PREFIX "trans name:%s\n", name);
out:
	return inode;
}

static int __init init_vineyard_fs(void) {
    int err;

    printk(KERN_INFO PREFIX "Hello, World!\n");
	err = register_filesystem(&vineyard_fs_type);
	if (!err) {
		err = net_link_init();
	}

    printk(KERN_INFO PREFIX "err num:%d\n", err);
    return err;
}

static void __exit vineyard_fs_exit(void) {
    printk(KERN_INFO PREFIX "Goodbye, World!\n");
    unregister_filesystem(&vineyard_fs_type);
	net_link_release();
}

module_init(init_vineyard_fs);
module_exit(vineyard_fs_exit);
