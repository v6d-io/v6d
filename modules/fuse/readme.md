# readme.md
run 
``` shell
cd build
 cmake .. -DBUILD_VINEYARD_FUSE=ON -DBUILD_VINEYARD_GRAPH=OFF
make vineyard-fusermount
sudo -E python3 ../modules/fuse/test/fuse_test.py 
```
to test the fuse implemenetation on the fuse,if it is error in the middle, run
``` shell
 sudo pkill -f vineyard
```
warning: it will kill all the process has pattern vineyard*
## folders 
### ./adaptors  
only arrow_ipc implemented,  deserializer_registry.h contains all the desearilizer, note that to enable the searilization in this PR, array is stored in  1*n table and under the column "a", and serializer_registry.h is for future, now the write is disabled. 
### ./test

it contains 1 python file, and another two debugging assistanct files, they read and write the arrow_ipc format.
## files 

fuse_impl.cc and fuse_impl.h is simply the impemenation of fuse3 interface

## linter
```shell
python3 -m isort --profile black --python-version 38 .
```
fusermount.cc is the mount file. 