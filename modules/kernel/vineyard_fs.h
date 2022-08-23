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
#include "msg_mgr.h"

#define PREFIX "[vineyardfs]"
#define VINEYARD_SUPER_MAGIC 0xabcdef

struct vineyard_attr {
	uint64_t	ino;
	uint64_t	size;
	uint64_t	blocks;
	uint64_t	atime;
	uint64_t	mtime;
	uint64_t	ctime;
	uint32_t	atimensec;
	uint32_t	mtimensec;
	uint32_t	ctimensec;
	uint32_t	mode;
	uint32_t	nlink;
	uint32_t	uid;
	uint32_t	gid;
	uint32_t	rdev;
	uint32_t	blksize;
	uint32_t	flags;
};

struct vineyard_entry {
	uint64_t			obj_id; // as name
	uint64_t			file_size;
	enum OBJECT_TYPE	type;
	unsigned long 		inode_id;
};
