vineyard-cli
============

**vineyard-ctl**: A command-line tool for **vineyard**.

Supported Commands
------------------

+ :code:`ls`
+ :code:`query`
+ :code:`head`
+ :code:`copy`
+ :code:`del`
+ :code:`stat`
+ :code:`put`
+ :code:`config`
+ :code:`migrate`

Connect to a vineyard server
----------------------------

+ Via command-line:
   Options:
   
   + :code:`ipc_socket`: Socket location of connected vineyard server.
   + :code:`rpc_host`: RPC HOST of the connected vineyard server.
   + :code:`rpc_port`: RPC PORT of the connected vineyard server.
   + :code:`rpc_endpoint`: RPC endpoint of the connected vineyard server.

Example:

.. code:: shell

    vineyard-ctl --ipc_socket /var/run/vineyard.sock

+ Via vineyard configuraion file:
This will pick IPC or RPC values from the vineyard configuration file.

Example:

.. code:: shell

    vineyard-ctl {command}

ls
---

List vineyard objects.

Options:

+ :code:`pattern`: The pattern string that will be matched against the object’s typename.
+ :code:`regex`: The pattern string will be considered as a regex expression.
+ :code:`limit`: The limit to list.

Example:

.. code:: shell

    vineyard-ctl ls --pattern * --regex --limit 8

query
-----

Query a vineyard object.

Options:

+ :code:`object_id`: ID of the object to be fetched.
+ :code:`meta`: Metadata of the object (**Simple** or **JSON**).
+ :code:`metric`: Metric data of the object (**nbytes** or **signature** or **typename**).
+ :code:`exists`: Check if the object exists or not.
+ :code:`stdout`: Get object to stdout.
+ :code:`output_file`: Get object to file.

Example:

.. code:: shell

    vineyard-ctl query --object_id 00002ec13bc81226 --meta json --metric typename

head
----

Print first n(limit) lines of a vineyard object. Currently supported for a pandas dataframe only.

Options:

+ :code:`object_id`: ID of the object to be printed.
+ :code:`limit`: Number of lines of the object to be printed.

Example:

.. code:: shell

    vineyard-ctl head --object_id 00002ec13bc81226 --limit 3

copy
----

Copy a vineyard object.

Options:

+ :code:`object_id`: ID of the object to be copied.
+ :code:`shallow`: Get a shallow copy of the object.
+ :code:`deep`: Get a deep copy of the object.

Example:

.. code:: shell

    vineyard-ctl copy --object_id 00002ec13bc81226 --shallow

del
---

Delete a vineyard object.

Options:

+ :code:`object_id`: ID of the object to be deleted.
+ :code:`regex_pattern`: Delete all the objects that match the regex pattern.
+ :code:`force`: Recursively delete even if the member object is also referred by others.
+ :code:`deep`: Deeply delete an object means we will deleting the members recursively.

Example:

.. code:: shell

    vineyard-ctl del --object_id 00002ec13bc81226 --force

stat
----

Get the status of connected vineyard server.

Options:

+ :code:`instance_id`: Instance ID of vineyardd that the client is connected to.
+ :code:`deployment`: The deployment mode of the connected vineyardd cluster.
+ :code:`memory_usage`: Memory usage (in bytes) of current vineyardd instance.
+ :code:`memory_limit`: Memory limit (in bytes) of current vineyardd instance.
+ :code:`deferred_requests`: Number of waiting requests of current vineyardd instance.
+ :code:`ipc_connections`: Number of alive IPC connections on the current vineyardd instance.
+ :code:`rpc_connections`: Number of alive RPC connections on the current vineyardd instance.

Example:

.. code:: shell

    vineyard-ctl stat

put
---

Put a python value to vineyard.

Options:

+ :code:`value`: The python value you want to put to the vineyard server.
+ :code:`file`: The file you want to put to the vineyard server as a pandas dataframe.
+ :code:`sep`: Delimiter used in the file.
+ :code:`delimiter`: Delimiter used in the file.
+ :code:`header`: Row number to use as the column names.

Example:

.. code:: shell

    vineyard-ctl put --file example_csv_file.csv --sep ,

config
------

Edit configuration file.

Options:

+ :code:`ipc_socket_value`: The ipc_socket value to enter in the config file.
+ :code:`rpc_host_value`: The rpc_host value to enter in the config file.
+ :code:`rpc_port_value`: The rpc_port value to enter in the config file.
+ :code:`rpc_endpoint_value`: The rpc_endpoint value to enter in the config file.

Example:

.. code:: shell

    vineyard-ctl config --ipc_socket_value /var/run/vineyard.sock

migrate
-------

Migrate a vineyard object.

Options:

+ :code:`ipc_socket_value`: The ipc_socket value for the second client.
+ :code:`rpc_host_value`: The rpc_host value for the second client.
+ :code:`rpc_port_value`: The rpc_port value for the second client.
+ :code:`rpc_endpoint_value`: The rpc_endpoint value for the second client.
+ :code:`object_id`: ID of the object to be migrated.
+ :code:`local`: Migrate the vineyard object local to local.
+ :code:`remote`: Migrate the vineyard object remote to local.

Example:

.. code:: shell

    vineyard-ctl migrate --ipc_socket_value /tmp/vineyard.sock --object_id 00002ec13bc81226 --remote
