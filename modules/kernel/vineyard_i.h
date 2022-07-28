#pragma once
#include "vineyard_fs.h"

void vineyard_fs_init_file_inode(struct inode *inode);
void vineyard_fs_init_dir_inode(struct inode *inode);

struct inode *vineyard_fs_iget(struct super_block *sb, u64 nodeid,
			int generation, struct vineyard_attr *attr);