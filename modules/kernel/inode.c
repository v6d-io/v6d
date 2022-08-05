#include <linux/stat.h>
#include <linux/atmioc.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include "vineyard_fs.h"
#include "vineyard_i.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("yuansm");
MODULE_DESCRIPTION("Vineyard filesystem for Linux.");
MODULE_VERSION("0.01");

#define PAGE_NUM	(1024 * 1024)
#define TOTAL_SIZE (PAGE_NUM * PAGE_SIZE)
static struct page **pages;
static void *vineyard_storage_kernel_addr;
static void *vineyard_msg_buffer_kernel_addr;
static void *vineyard_result_buffer_kernel_addr;

struct mount_data {
	unsigned long this_addr;
	unsigned long vineyard_storage_user_addr;
	unsigned long vineyard_msg_buffer_user_addr;
	unsigned long vineyard_result_buffer_user_addr;
};

struct vineyard_fs_context {
    int key:1;
};

struct vineyard_sb_info {

};

static inline void vineyard_lock_build_inode(struct vineyard_sb_info *sbi)
{
	// TBD
}

static inline void vineyard_unlock_build_inode(struct vineyard_sb_info *sbi)
{
	// TBD
}

static struct inode *vineyard_alloc_inode(struct super_block *sb)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	return kzalloc(sizeof(struct inode), GFP_KERNEL);
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

// static int vineyard_show_options(struct seq_file *sf, struct dentry *d)
// {
// 	printk(KERN_INFO PREFIX "fake %s\n", __func__);
// 	return 0;
// }

static const struct super_operations vineyard_super_ops = {
	.alloc_inode	= vineyard_alloc_inode,
	.free_inode	= vineyard_free_inode,
	.write_inode	= vineyard_write_inode,
	.evict_inode	= vineyard_evict_inode,
	.put_super	= vineyard_put_super,
	.statfs		= vineyard_statfs,
	.remount_fs	= vineyard_remount,

	// .show_options	= vineyard_show_options,
};

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

static int map_vineyard_user_storage_buffer(unsigned long user_addr)
{
	struct vm_area_struct *user_vma;
	int i;
	int ret;

	printk(KERN_INFO PREFIX "%s\n", __func__);
	pages = vmalloc(PAGE_NUM * sizeof(struct page *));
	memset(pages, 0, sizeof(struct page *) * PAGE_NUM);

	user_vma = find_vma(current->mm, user_addr);
	printk(KERN_INFO "addr: %lx\n", user_addr);
	ret = get_user_pages_fast(user_addr, PAGE_NUM, 1, pages);
	if (ret < 0) {
		printk(KERN_INFO "pined page error %d\n", ret);
		vfree(pages);
		return -1;
	}

	vineyard_storage_kernel_addr = vmap(pages, PAGE_NUM, VM_MAP, PAGE_KERNEL);

	// test write data
	if (vineyard_storage_kernel_addr) {
		for (i = 0; i < TOTAL_SIZE / sizeof(int); i++) {
			((int *)vineyard_storage_kernel_addr)[i] = i;
		}
		printk(KERN_INFO PREFIX "number:%lu\n", TOTAL_SIZE / sizeof(int));
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

static unsigned long prepare_user_buffer(void **kernel_addr)
{
	unsigned long user_addr;
	struct vm_area_struct *vma;
	int i;

	*kernel_addr = kmalloc(PAGE_SIZE, GFP_KERNEL);
	if (!vineyard_msg_buffer_kernel_addr) {
		return 0;
	}

	user_addr = vm_mmap(NULL, 0, PAGE_SIZE,
			    PROT_READ | PROT_WRITE,
			    MAP_SHARED, 0);
	vma = find_vma(current->mm, user_addr);
	remap_pfn_range(vma, user_addr, PFN_DOWN(__pa(*kernel_addr)), PAGE_SIZE, vma->vm_page_prot);

	//for test
	for (i = 0; i < 100; i++) {
		((int *)(*kernel_addr))[i] = i;
	}

	return user_addr;
}

static void unmap_vineyard_msg_result_buffer(void)
{
	kfree(vineyard_msg_buffer_kernel_addr);
	kfree(vineyard_result_buffer_kernel_addr);
}

static void unmap_vineyard_user_storage_buffer(void)
{
	int i;

	printk(KERN_INFO PREFIX "%s\n", __func__);
	if (vineyard_storage_kernel_addr) {
		printk(KERN_INFO "unmap and put pages\n");
		vunmap(vineyard_storage_kernel_addr);
		for (i = 0; i < PAGE_NUM; i++) {
			put_page(pages[i]);
		}
		vfree(pages);
		vineyard_storage_kernel_addr = NULL;
	}
}

static int vineyard_parse_monolithic(struct fs_context *fsc, void *data)
{
	int err = 0;
	struct mount_data *mount_data;

    printk(KERN_INFO PREFIX "%s\n", __func__);
	mount_data = (struct mount_data *)data;

	if (mount_data) {
		// map user buffer to kernel
		err = map_vineyard_user_storage_buffer(mount_data->vineyard_storage_user_addr);
		mount_data->vineyard_msg_buffer_user_addr = prepare_user_buffer(&vineyard_msg_buffer_kernel_addr);
		mount_data->vineyard_result_buffer_user_addr = prepare_user_buffer(&vineyard_msg_buffer_kernel_addr);
		err = copy_to_user((void *)(mount_data->this_addr), mount_data, sizeof(struct mount_data));
		printk(KERN_INFO PREFIX "%lx %lx %lx %lx\n", mount_data->this_addr, mount_data->vineyard_storage_user_addr, mount_data->vineyard_msg_buffer_user_addr, mount_data->vineyard_result_buffer_user_addr);
	}

    return err;
}

const struct dentry_operations vineyard_root_dentry_operations = {
};

static const struct export_operations vineyard_export_operations = {
};

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

void vineyard_sb_defaults(struct super_block *sb)
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

int _open(struct inode *, struct file *)
{
	printk(KERN_INFO "%s\n", __func__);
	return -1;
}

static void vineyard_fs_init_inode(struct inode *inode, struct vineyard_attr *attr)
{
	printk(KERN_INFO PREFIX "%s\n", __func__);
    inode->i_mode = attr->mode;
	inode->i_ino = attr->ino;
	inode->i_size = attr->size;
	inode->i_mtime.tv_sec  = attr->mtime;
	inode->i_mtime.tv_nsec = attr->mtimensec;
	inode->i_ctime.tv_sec  = attr->ctime;
	inode->i_ctime.tv_nsec = attr->ctimensec;

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

struct inode *vineyard_fs_iget(struct super_block *sb, struct vineyard_attr *attr)
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

    attr.mode = S_IFDIR;
    attr.ino = 1;

	return vineyard_fs_iget(sb, &attr);
}

int vineyard_set_super_block(struct super_block *sb, struct fs_context *fsc)
{
    struct vineyard_fs_context *ctx;

    printk(KERN_INFO PREFIX "%s\n", __func__);

	ctx = fsc->fs_private;
    vineyard_sb_defaults(sb);

    sb->s_blocksize = PAGE_SIZE;
    sb->s_blocksize_bits = PAGE_SHIFT;

    return 0;
}

uint64_t get_next_vineyard_ino(void)
{
	static uint64_t ino = 1;
	// TODO: atomic ins
	ino++;
	return ino;
}

struct inode *vineyard_fs_build_inode(struct super_block *sb)
{
	struct inode *inode;
	struct vineyard_attr attr;

	attr.mode = S_IFREG;
	attr.ino = get_next_vineyard_ino();
	vineyard_lock_build_inode(NULL);

	inode = vineyard_fs_iget(sb, &attr);

	vineyard_unlock_build_inode(NULL);
	return inode;
}

static int vineyard_get_tree(struct fs_context *fsc)
{
	struct super_block *sb;
	struct inode *root;
	struct dentry *root_dentry;
    int err = 0;

    printk(KERN_INFO PREFIX "%s\n", __func__);

    sb = sget_fc(fsc, NULL, vineyard_set_super_block);

    root = vineyard_get_root_inode(sb);
    sb->s_d_op = &vineyard_root_dentry_operations;
    root_dentry = d_make_root(root);
    sb->s_root = root_dentry;
	fsc->root = dget(sb->s_root);

	printk(KERN_INFO PREFIX "%s end\n", __func__);
    return err;
}

static const struct fs_context_operations vineyard_context_ops = {
	.free		= vineyard_free_fsc,
	.parse_monolithic	= vineyard_parse_monolithic,
	.parse_param	= vineyard_parse_param,
	.get_tree	= vineyard_get_tree,
};

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

void fake_kill_sb(struct super_block *sb)
{
    printk(KERN_INFO PREFIX "fake kill sb for unmount\n");
    kfree(sb);
}

static void vineyard_kill_sb(struct super_block *sb)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	unmap_vineyard_user_storage_buffer();
	unmap_vineyard_msg_result_buffer();
}

static struct file_system_type vineyard_fs_type = {
	.owner		= THIS_MODULE,
	.name		= "vineyardfs",
    .init_fs_context = vineyard_init_fs_context,
	.kill_sb	= vineyard_kill_sb,
	.fs_flags	= 0,
};
MODULE_ALIAS_FS("vineyardfs");

static int __init init_vineyard_fs(void) {
    int err;

    printk(KERN_INFO PREFIX "Hello, World!\n");
	err = register_filesystem(&vineyard_fs_type);

    printk(KERN_INFO PREFIX "err num:%d\n", err);
    return err;
}

static void __exit vineyard_fs_exit(void) {
    printk(KERN_INFO PREFIX "Goodbye, World!\n");
    unregister_filesystem(&vineyard_fs_type);
}

module_init(init_vineyard_fs);
module_exit(vineyard_fs_exit);