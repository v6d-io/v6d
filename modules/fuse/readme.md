# readme.md
run 
``` shell
cd build
cmake ..
make vineyard-fusermount
sudo -E python3 ../modules/fuse/ltest/fuse_test.py 
```
to test the fuse implemenetation on the fuse,if it is error in the middle, run
``` shell
 sudo pkill -f vineyard* 
```
warning: it will kill all the process has pattern vineyard*

explanation 
## folders:

### ./adaptors  
only arrow_ipc implemented,  deserializer_registry.h contains all the searilizer and desearilizer, note that to enable the searilization, array is stored in  1*n table and under the column "a". it also has some misc files for dev_use. 

### ./test
    it contains two python scripts, one is adaptor, another is adaptor
## files:
fuse_impl.cc and fuse_impl.h is simply the impemenation of fuse3 interface

fusermount.cc is the mount file. 





