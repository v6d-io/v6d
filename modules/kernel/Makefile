obj-m := vineyard_fs.o
vineyard_fs-objs := inode.o xattr.o file.o dir.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean