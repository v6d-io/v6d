/** Copyright 2020-2022 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <linux/string.h>
#include <linux/futex.h>
#include <linux/wait.h>
#include <linux/mutex.h>
#include "vineyard_fs.h"
#include "vineyard_i.h"
#include "msg_mgr.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Vineyard authors");
MODULE_DESCRIPTION("Vineyard filesystem for Linux.");
MODULE_VERSION("0.01");

// tools
static int sync_dir_info(void)
{
	int ret;
	struct vineyard_request_msg msg;
	struct vineyard_result_msg rmsg;

	msg.opt = VINEYARD_READDIR;
	ret = send_request_msg(&msg);
	if (ret)
		return ret;

	ret = receive_result_msg(&rmsg);
	if (ret)
		return ret;

	return rmsg.ret._fopt_ret.ret;
}

static inline int vineyard_get_entry(struct file *dir_file, int *cpos,
				     struct vineyard_entry **entry, char *name)
{
	// TODO: communicate with vineyard
	struct vineyard_entry *entrys;

	// TODO: need a machinasm to find out file pos. Now we suppose that
	// vineyard will not delete object.
	printk(KERN_INFO PREFIX "%s\n", __func__);
	if (!vineyard_connect) {
		printk(KERN_INFO PREFIX "Vineyard disconnected!\n");
		return -1;
	}

	printk(KERN_INFO PREFIX "cpos:%d total_file:%d\n", *cpos,
	       vineyard_object_info_header->total_file);
	if (*cpos < (vineyard_object_info_header->total_file)) {
		entrys = (struct vineyard_entry *)
			vineyard_object_info_buffer_addr;
		*entry = &(entrys[(*cpos)]);
		printk(KERN_INFO "id:%llu \n", (*entry)->obj_id);
		if ((*entry)->inode_id == 0) {
			(*entry)->inode_id = get_next_vineyard_ino();
		}
		*cpos = *cpos + 1;
		translate_u64_to_char((*entry)->obj_id, name);
		return 0;
	}
	return -1;
}

// vineyard dir operations
static int vineyard_fs_dir_read(struct file *file, struct dir_context *ctx)
{
	int ret;
	struct dentry *dentry;
	const unsigned char *path;
	struct inode *dir_i;
	struct super_block *sb;
	struct vineyard_entry *entry;
	char name[32] = { 0 };
	int cpos = 0;
	int find = 0;
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

	// TODO: fake . and ..
	cpos = ctx->pos;
	printk(KERN_INFO PREFIX "cpos:%d\n", cpos);
	// dir_emit_dots(file, ctx);
	sync_dir_info();

	vineyard_read_lock(&vineyard_object_info_header->rw_lock);
get_new:
	if (vineyard_get_entry(file, &cpos, &entry, name) == -1) {
		if (!find)
			ret = 0;
		goto out;
	}
	find = 1;

	// we suppose that vineyard just contain file.
	dir_emit(ctx, name, strlen(name), entry->inode_id, DT_REG);

	ctx->pos = cpos;
	goto get_new;

out:
	vineyard_read_unlock(&vineyard_object_info_header->rw_lock);
	dput(dentry);

	printk(KERN_INFO PREFIX "ret : %d\n", ret);
	return ret;
}

static int vineyard_fs_dir_fsync(struct file *file, loff_t start, loff_t end,
				 int datasync)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	return -1;
}

static const struct file_operations vineyard_fs_dir_operations = {
	.iterate = vineyard_fs_dir_read,
	.fsync = vineyard_fs_dir_fsync,
	.read = generic_read_dir,
};

// vineyard dir inode operations
static int vineyard_find(struct inode *dir, struct qstr d_name)
{
	return 0;
}

static struct dentry *vineyard_fs_dir_lookup(struct inode *dir,
					     struct dentry *dentry,
					     unsigned int flags)
{
	struct super_block *sb;
	struct inode *inode;
	int err;

	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	printk(KERN_INFO PREFIX "inode no:%lu\n", dir->i_ino);
	printk(KERN_INFO PREFIX "name:%s\n", dentry->d_name.name);

	sb = dir->i_sb;
	mutex_lock(&(VINEYARD_SB_INFO(sb)->s_lock));

	// check if there exists a file in the file system
	err = vineyard_find(dir, dentry->d_name);
	if (err) {
		if (err == -ENOENT) {
			inode = NULL;
			goto out;
		}
		goto error;
	}

	inode = vineyard_fs_build_inode(sb, dentry->d_name.name);
	if (IS_ERR(inode)) {
		err = PTR_ERR(inode);
		goto error;
	}

out:
	mutex_unlock(&(VINEYARD_SB_INFO(sb)->s_lock));
	return d_splice_alias(inode, dentry);
error:
	mutex_unlock(&VINEYARD_SB_INFO(sb)->s_lock);
	return ERR_PTR(err);
}

static int vineyard_fs_dir_create(struct user_namespace *mnt_userns,
				  struct inode *dir, struct dentry *entry,
				  umode_t mode, bool excl)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	return -1;
}

static int vineyard_fs_dir_permission(struct user_namespace *mnt_userns,
				      struct inode *inode, int mask)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	return 0;
}

static ssize_t vineyard_fs_dir_listxattr(struct dentry *entry, char *list,
					 size_t size)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	return 0;
}

static int vineyard_fs_dir_fileattr_get(struct dentry *entry,
					struct fileattr *fa)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	return -1;
}

static int vineyard_fs_dir_fileattr_set(struct user_namespace *mnt_userns,
					struct dentry *dentry,
					struct fileattr *fa)
{
	printk(KERN_INFO PREFIX "fake %s\n", __func__);
	return -1;
}

static const struct inode_operations vineyard_dir_inode_operations = {
	.lookup = vineyard_fs_dir_lookup,
	.create = vineyard_fs_dir_create,
	.permission = vineyard_fs_dir_permission,
	.setattr = vineyard_setattr,
	.getattr = vineyard_getattr,
	.listxattr = vineyard_fs_dir_listxattr,
	.fileattr_get = vineyard_fs_dir_fileattr_get,
	.fileattr_set = vineyard_fs_dir_fileattr_set,
};

// interfaces
void vineyard_fs_init_dir_inode(struct inode *inode)
{
	printk(KERN_INFO PREFIX "%s\n", __func__);
	inode->i_op = &vineyard_dir_inode_operations;
	inode->i_fop = &vineyard_fs_dir_operations;
}
