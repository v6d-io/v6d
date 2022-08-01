#pragma once
#include "vineyard_fs.h"

uint64_t get_next_vineyard_ino(void);
void vineyard_fs_init_file_inode(struct inode *inode);
void vineyard_fs_init_dir_inode(struct inode *inode);

struct inode *vineyard_fs_build_inode(struct super_block *sb);
struct inode *vineyard_fs_iget(struct super_block *sb, struct vineyard_attr *attr);