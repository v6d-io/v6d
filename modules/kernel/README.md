# Vineyard kernel module
## **Introduction**
Vineyard kernel module is a file system module for **Linux**. It can provide a file view of objects in vineyard storage. A blob is treated as a binary file, a tensor is treated as a npy file and a dataframe is treated as a parquet/arrow file(TBD). User can use POSIX interface (e.g. open(), read(), close()) to access vineyard objects. This project makes it possible for other user programs to be ported to vineyard storage engine with very little modification.

**version： v0.1**

|  type      | view          | open(ro) | open(rw) | read  | write  | fsync |
|  :----     | :----         | :---:    | :---:    | :---: | :---:  | :---: |
| Blob       | binary        | O        | -        | O     | -      | -     |
| Tensor     | npy           | O        | -        | O     | -      | -     |
| Dataframe  | arrow/parquet | -        | -        | -     | -      | -     |

## **Example**

**Compile and run**

**Go to v6d/modules/kernel file directory and compile kernel module**

```shell
cd modules/kernel
## now at v6d/modules/kernel/
make -j$(nproc)
```

**Add the kernel module into Linux kernel**

```shell
sudo insmod vineyard_fs.ko
```

**Go to root directory of vineyard and add example program to add objects into vineyard**
```shell
cd ../..
## now at v6d/
mkdir example
cd example
vim test.cc
```
code:
``` C
#include <memory>
#include <string>
#include <thread>

#include "arrow/api.h"
#include "arrow/io/api.h"

#include "basic/ds/tensor.h"
#include "client/client.h"
#include "client/ds/object_meta.h"
#include "common/util/logging.h"

using namespace vineyard;

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage ./list_object_test <ipc_socket>");
    return 1;
  }
  std::string ipc_socket = std::string(argv[1]);

  Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));
  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  TensorBuilder<double> builder(client, {2, 3, 4});
  double* data = builder.data();
  for (int i = 0; i < 24; ++i) {
    data[i] = i;
  }
  auto sealed = std::dynamic_pointer_cast<Tensor<double>>(builder.Seal(client));
  VINEYARD_CHECK_OK(client.Persist(sealed->id()));
  LOG(INFO) << "Finish building a tensor";

  client.Disconnect();

  return 0;
}
```

**Create CMakeLists.txt**

```
vim ./CMakeLists.txt
```

code:

```shell
## now at v6d/example/CMakeLists.txt

add_executable(test test.cc)
target_link_libraries(test PUBLIC vineyard_client vineyard_basic)
```

**Open CMakeLists.txt in root directory of v6d and config BUILD_NETLINK_SERVER as ON**

```Cmake
## Line31
option(BUILD_NETLINK_SERVER "Build netlink server" ON)
...
## End of file
add_subdirectory(example)
```

**Go to root directory and compile v6d**

```shell
cd ..
git submodule update --init
mkdir build
cd build
## now at v6d/build/
cmake ..
make -j$(nproc)
```
(Now at v6d/build directory)

**Run vineyard**

```shell
sudo ./bin/vineyardd /var/run/vineyard.sock --meta=local
```

If you can see

```
I0922 10:50:05.162508 276180 rpc_server.cc:54] Vineyard will listen on 0.0.0.0:9600 for RPC
I0922 10:50:05.162593 276190 netlink_server.cc:507] Net link server handler thread start!
```

This means that vineyard file system started successfully!

**Open another terminal and go to build directory and run test program**

```shell
## now at v6d/build/

sudo ./bin/test /var/run/vineyard.sock
```

If you can see

```
I0922 11:17:09.411404 280903 test.cc:24] Connected to IPCServer: /var/run/vineyard.sock
I0922 11:17:09.411988 280903 test.cc:33] Finish building a tensor
```

It means that we create a tensor and store it into vineyard.

**Mount vineyard to a directory(e.g. v6d/mount-point/) and show all files**

```shell
cd ..
## now at v6d/
mkdir mount-point
sudo mount -t vineyardfs test ./mount-point
ls ./mount-point
```

And you can see

```shell
➜  v6d git:(dev/kernel) ✗ ls ./mount-point 
1585311138050730.npy  9224957347992150643
## The actual file name may differ from the one shown above.
```

**Use a test program to open the file and read it!**

```shell
vim test.py
```

code:
```python
import numpy as np

arr = np.load('/path/of/your/v6d/mount-point/1585311138050730.npy')
print(arr)
print(arr.shape)
```

run it

```shell
python3 test.py
```

And you can see:

```shell
[[[ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]]

 [[12. 13. 14. 15.]
  [16. 17. 18. 19.]
  [20. 21. 22. 23.]]]
(2, 3, 4)
```

## **Close**

**Umount file system**

```shell
## now at v6d/
sudo umount ./mount-point
## Now we can not mount again after umount. You need to restart vineyardd to mount it again.
```

**Close vineyardd**

Use ctrl c at vineyard server terminal.

You can see:
```shell
I0922 11:30:07.368603 280601 netlink_server.cc:553] Bye! Handler thread exit!
^CI0922 11:33:02.047547 280591 vineyardd.cc:61] SIGTERM Signal received, stop vineyard server...
I0922 11:33:02.047596 280591 meta_service.cc:46] meta service is stopping ...
➜  build git:(dev/kernel) ✗
```

**Remove file system kernel module**
```shell
sudo rmmod vineyard_fs.ko
```

**Clean up file**
```shell
cd modules/kernel
## now at v6d/modules/kernel/
make clean
```
