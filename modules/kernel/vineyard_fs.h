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
#pragma once
#include <linux/exportfs.h>
#include <linux/fs.h>
#include <linux/fs_context.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/list.h>
#include <linux/mm.h>
#include <linux/mman.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/user_namespace.h>
#include <linux/xattr.h>
#include "msg_mgr.h"

#define PREFIX "[vineyardfs]"
#define VINEYARD_SUPER_MAGIC 0xabcdef

#define VINEYARD_HASHSIZE (1 << 8)
struct vineyard_attr {
	uint64_t ino;
	uint64_t size;
	uint32_t mode;
};

struct vineyard_entry {
	uint64_t obj_id; // as name
	uint64_t file_size;
	enum OBJECT_TYPE type;
	uint64_t inode_id;
};

struct vineyard_private_data {
	void *pointer;
	uint64_t size;
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
int vineyard_setattr(struct user_namespace *mnt_userns, struct dentry *entry,
		     struct iattr *attr);
int vineyard_getattr(struct user_namespace *mnt_userns, const struct path *path,
		     struct kstat *stat, u32 mask, unsigned int flags);
