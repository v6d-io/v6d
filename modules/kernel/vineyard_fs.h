#pragma once
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/user_namespace.h>
#include <linux/fs_context.h>
#include <linux/xattr.h>
#include <linux/exportfs.h>
#include <linux/mm.h>
#include <linux/mman.h>
#include <linux/list.h>
#include "msg_mgr.h"

#define PREFIX "[vineyardfs]"
#define VINEYARD_SUPER_MAGIC 0xabcdef

#define VINEYARD_HASHSIZE (1 << 8)
struct vineyard_attr {
	uint64_t	ino;
	uint64_t	size;
	uint32_t	mode;
};

struct vineyard_entry {
	uint64_t			obj_id; // as name
	uint64_t			file_size;
	enum OBJECT_TYPE	type;
	unsigned long 		inode_id;
};

struct vineyard_private_data {
	void 		*pointer;
	uint64_t 	size;
};

// kernel space
extern void *vineyard_bulk_kernel_addr;
extern struct vineyard_msg_mem_header *vineyard_msg_mem_header;
extern struct vineyard_result_mem_header *vineyard_result_mem_header;
extern struct vineyard_object_info_header *vineyard_object_info_header;
extern void *vineyard_msg_buffer_addr;
extern void *vineyard_result_buffer_addr;
extern void *vineyard_object_info_buffer_addr;

// user space
extern void *vineyard_object_info_user_addr;

extern struct wait_queue_head vineyard_msg_wait;
extern struct wait_queue_head vineyard_fs_wait;

void translate_u64_to_char(uint64_t num, char *name);
uint64_t translate_char_to_u64(const char *name);
int vineyard_setattr(struct user_namespace *mnt_userns, struct dentry * entry, struct iattr *attr);
int vineyard_getattr(struct user_namespace *mnt_userns, const struct path *path, struct kstat *stat, u32 mask, unsigned int flags);
