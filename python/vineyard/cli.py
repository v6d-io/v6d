#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# PYTHON_ARGCOMPLETE_OK
#
# Copyright 2020-2021 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""vineyard-ctl: A command line tool for vineyard."""

import argparse
import sys
import os
import json
import pandas as pd
import treelib
import argcomplete

import vineyard

EXAMPLES = """
Some examples on how to use vineyard-ctl:

1. Connect to a vineyard server
    >>> vineyard-ctl --ipc_socket /var/run/vineyard.sock

2. List vineyard objects
    >>> vineyard-ctl ls --limit 6

3. Query a vineyard object
    >>> vineyard-ctl query --object_id 00002ec13bc81226

4. Print first n lines of a vineyard object
    >>> vineyard-ctl head --object_id 00002ec13bc81226 --limit 3

5. Copy a vineyard object
    >>> vineyard-ctl copy --object_id 00002ec13bc81226 --shallow

6. Delete a vineyard object
    >>> vineyard-ctl del --object_id 00002ec13bc81226

7. Get the status of connected vineyard server
    >>> vineyard-ctl stat

8. Put a python value to vineyard
    >>> vineyard-ctl put --file example_csv_file.csv

9. Edit configuration file
    >>> vineyard-ctl config --ipc_socket_value /var/run/vineyard.sock

10. Migrate a vineyard object
    >>> vineyard-ctl migrate --ipc_socket_value /tmp/vineyard.sock --object_id 00002ec13bc81226 --local

11. Issue a debug request
    >>> vineyard-ctl debug --payload '{"instance_status":[], "memory_size":[]}'

12. Start vineyardd
    >>> vineyard-ctl start --local
"""


def vineyard_argument_parser():
    """Utility to create a command line Argument Parser."""
    parser = argparse.ArgumentParser(prog='vineyard-ctl',
                                     usage='%(prog)s [options]',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='vineyard-ctl: A command line tool for vineyard',
                                     allow_abbrev=False,
                                     epilog=EXAMPLES)
    parser.add_argument('--version', action='version', version=f'{parser.prog} v{vineyard.__version__}')
    parser.add_argument('--ipc_socket', help='Socket location of connected vineyard server')
    parser.add_argument('--rpc_host', help='RPC HOST of the connected vineyard server')
    parser.add_argument('--rpc_port', type=int, help='RPC PORT of the connected vineyard server')
    parser.add_argument('--rpc_endpoint', help='RPC endpoint of the connected vineyard server')

    cmd_parser = parser.add_subparsers(title='commands', dest='cmd')

    ls_opt = cmd_parser.add_parser('ls',
                                   formatter_class=argparse.RawDescriptionHelpFormatter,
                                   description='Description: List vineyard objects',
                                   epilog='Example:\n\n>>> vineyard-ctl ls --pattern * --regex --limit 8')
    ls_opt.add_argument('--pattern',
                        default='*',
                        type=str,
                        help='The pattern string that will be matched against the object’s typename')
    ls_opt.add_argument('--regex',
                        action='store_true',
                        help='The pattern string will be considered as a regex expression')
    ls_opt.add_argument('--limit', default=5, type=int, help='The limit to list')

    query_opt = cmd_parser.add_parser('query',
                                      formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='Description: Query a vineyard object',
                                      epilog=('Example:\n\n>>> vineyard-ctl query --object_id ' +
                                              '00002ec13bc81226 --meta json --metric typename'))
    query_opt.add_argument('--object_id', required=True, help='ID of the object to be fetched')
    query_opt.add_argument('--meta',
                           choices=['simple', 'json'],
                           help='Metadata of the object').completer=\
        argcomplete.completers.ChoicesCompleter(('simple', 'json'))
    query_opt.add_argument('--metric',
                           choices=['nbytes', 'signature', 'typename'],
                           help='Metric data of the object').completer=\
        argcomplete.completers.ChoicesCompleter(('nbytes', 'signature', 'typename'))
    query_opt.add_argument('--exists', action='store_true', help='Check if the object exists or not')
    query_opt.add_argument('--stdout', action='store_true', help='Get object to stdout')
    query_opt.add_argument('--output_file', type=str, help='Get object to file')
    query_opt.add_argument('--tree', action='store_true', help='Get object lineage in tree-like style')
    query_opt.add_argument('--memory_status', action='store_true', help='Get the memory used by the vineyard object')
    query_opt.add_argument('--detail', action='store_true', help='Get detailed memory used by the vineyard object')

    head_opt = cmd_parser.add_parser('head',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Description: Print first n(limit) lines of a vineyard object',
                                     epilog='Example:\n\n>>> vineyard-ctl head --object_id 00002ec13bc81226 --limit 3')
    head_opt.add_argument('--object_id', required=True, help='ID of the object to be printed')
    head_opt.add_argument('--limit', type=int, default=5, help='Number of lines of the object to be printed')

    copy_opt = cmd_parser.add_parser('copy',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Description: Copy a vineyard object',
                                     epilog='Example:\n\n>>> vineyard-ctl copy --object_id 00002ec13bc81226 --shallow')
    copy_opt.add_argument('--object_id', required=True, help='ID of the object to be copied')
    copy_opt.add_argument('--shallow', action='store_true', help='Get a shallow copy of the object')
    copy_opt.add_argument('--deep', action='store_true', help='Get a deep copy of the object')

    del_opt = cmd_parser.add_parser('del',
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    description='Description: Delete a vineyard object',
                                    epilog='Example:\n\n>>> vineyard-ctl del --object_id 00002ec13bc81226 --force')
    del_opt_group = del_opt.add_mutually_exclusive_group(required=True)
    del_opt_group.add_argument('--object_id', help='ID of the object to be deleted')
    del_opt_group.add_argument('--regex_pattern', help='Delete all the objects that match the regex pattern')

    del_opt.add_argument('--force',
                         action='store_true',
                         help='Recursively delete even if the member object is also referred by others')
    del_opt.add_argument('--deep',
                         action='store_true',
                         help='Deeply delete an object means we will deleting the members recursively')

    stat_opt = cmd_parser.add_parser('stat',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Description: Get the status of connected vineyard server',
                                     epilog='Example:\n\n>>> vineyard-ctl stat')
    stat_opt.add_argument('--instance_id',
                          dest='properties',
                          action='append_const',
                          const='instance_id',
                          help='Instance ID of vineyardd that the client is connected to')
    stat_opt.add_argument('--deployment',
                          dest='properties',
                          action='append_const',
                          const='deployment',
                          help='The deployment mode of the connected vineyardd cluster')
    stat_opt.add_argument('--memory_usage',
                          dest='properties',
                          action='append_const',
                          const='memory_usage',
                          help='Memory usage (in bytes) of current vineyardd instance')
    stat_opt.add_argument('--memory_limit',
                          dest='properties',
                          action='append_const',
                          const='memory_limit',
                          help='Memory limit (in bytes) of current vineyardd instance')
    stat_opt.add_argument('--deferred_requests',
                          dest='properties',
                          action='append_const',
                          const='deferred_requests',
                          help='Number of waiting requests of current vineyardd instance')
    stat_opt.add_argument('--ipc_connections',
                          dest='properties',
                          action='append_const',
                          const='ipc_connections',
                          help='Number of alive IPC connections on the current vineyardd instance')
    stat_opt.add_argument('--rpc_connections',
                          dest='properties',
                          action='append_const',
                          const='rpc_connections',
                          help='Number of alive RPC connections on the current vineyardd instance')

    put_opt = cmd_parser.add_parser('put',
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    description='Description: Put a python value to vineyard',
                                    epilog='Example:\n\n>>> vineyard-ctl put --file example_csv_file.csv --sep ,')

    put_opt_group = put_opt.add_mutually_exclusive_group(required=True)
    put_opt_group.add_argument('--value', help='The python value you want to put to the vineyard server')
    put_opt_group.add_argument('--file', help='The file you want to put to the vineyard server as a pandas dataframe')

    put_opt.add_argument('--sep', default=',', help='Delimiter used in the file')
    put_opt.add_argument('--delimiter', default=',', help='Delimiter used in the file')
    put_opt.add_argument('--header', type=int, default=0, help='Row number to use as the column names')

    config_opt = cmd_parser.add_parser('config',
                                       formatter_class=argparse.RawDescriptionHelpFormatter,
                                       description='Description: Edit configuration file',
                                       epilog=('Example:\n\n>>> vineyard-ctl config --ipc_socket_value ' +
                                               '/var/run/vineyard.sock'))
    config_opt.add_argument('--ipc_socket_value', help='The ipc_socket value to enter in the config file')
    config_opt.add_argument('--rpc_host_value', help='The rpc_host value to enter in the config file')
    config_opt.add_argument('--rpc_port_value', help='The rpc_port value to enter in the config file')
    config_opt.add_argument('--rpc_endpoint_value', help='The rpc_endpoint value to enter in the config file')

    migrate_opt = cmd_parser.add_parser('migrate',
                                        formatter_class=argparse.RawDescriptionHelpFormatter,
                                        description='Description: Migrate a vineyard object',
                                        epilog=('Example:\n\n>>> vineyard-ctl migrate --ipc_socket_value ' +
                                                '/tmp/vineyard.sock --object_id 00002ec13bc81226 --remote'))
    migrate_opt.add_argument('--ipc_socket_value', help='The ipc_socket value for the second client')
    migrate_opt.add_argument('--rpc_host_value', help='The rpc_host value for the second client')
    migrate_opt.add_argument('--rpc_port_value', help='The rpc_port value for the second client')
    migrate_opt.add_argument('--rpc_endpoint_value', help='The rpc_endpoint value for the second client')
    migrate_opt.add_argument('--object_id', required=True, help='ID of the object to be migrated')

    migration_choice_group = migrate_opt.add_mutually_exclusive_group(required=True)
    migration_choice_group.add_argument('--local',
                                        action='store_true',
                                        help='Migrate the vineyard object local to local')
    migration_choice_group.add_argument('--remote',
                                        action='store_true',
                                        help='Migrate the vineyard object remote to local')

    debug_opt = cmd_parser.add_parser('debug',
                                      formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='Description: Issue a debug request',
                                      epilog=('Example:\n\n>>> vineyard-ctl debug --payload ' +
                                              '\'{"instance_status":[], "memory_size":[]}\''))
    debug_opt.add_argument('--payload', type=json.loads, help='The payload that will be sent to the debug handler')

    start_opt = cmd_parser.add_parser('start',
                                      formatter_class=argparse.RawDescriptionHelpFormatter,
                                      description='Description: Start vineyardd',
                                      epilog='Example:\n\n>>> vineyard-ctl start --local')

    start_opt_group = start_opt.add_mutually_exclusive_group(required=True)
    start_opt_group.add_argument('--local', action='store_true', help='start a local vineyard cluster')
    start_opt_group.add_argument('--distributed',
                                 action='store_true',
                                 help='start a local vineyard cluster in a distributed fashion')

    start_opt.add_argument('--hosts', nargs='+', default=None, help='A list of machines to launch vineyard server')
    start_opt.add_argument('--etcd_endpoints',
                           type=str,
                           default=None,
                           help=('Launching vineyard using specified etcd endpoints. If not specified, vineyard ' +
                                 'will launch its own etcd instance'))
    start_opt.add_argument('--vineyardd_path',
                           type=str,
                           default=None,
                           help=('Location of vineyard server program. If not specified, vineyard will ' +
                                 'use its own bundled vineyardd binary'))
    start_opt.add_argument('--size',
                           type=str,
                           default='256M',
                           help=('The memory size limit for vineyard’s shared memory. The memory size can ' +
                                 'be a plain integer or as a fixed-point number using one of these ' +
                                 'suffixes: E, P, T, G, M, K. You can also use the power-of-two ' +
                                 'equivalents: Ei, Pi, Ti, Gi, Mi, Ki.'))
    start_opt.add_argument('--socket',
                           type=str,
                           default='/var/run/vineyard.sock',
                           help=('The UNIX domain socket socket path that vineyard server will listen on. ' +
                                 'When the socket parameter is None, a random path under temporary directory ' +
                                 'will be generated and used.'))
    start_opt.add_argument('--rpc_socket_port',
                           type=int,
                           default=9600,
                           help='The port that vineyard will use to privode RPC service')
    start_opt.add_argument('--debug', type=bool, default=False, help='Whether print debug logs')

    argcomplete.autocomplete(parser)
    return parser


optparser = vineyard_argument_parser()


def exit_with_help():
    """Utility to exit the program with help message in case of any error."""
    optparser.print_help(sys.stderr)
    sys.exit(-1)


def connect_vineyard(args):
    """Utility to create a vineyard client using an IPC or RPC socket."""
    if args.ipc_socket is not None:
        client = vineyard.connect(args.ipc_socket)
        # force use rpc client in cli tools
        client = vineyard.connect(*client.rpc_endpoint.split(':'))
    elif args.rpc_endpoint is not None:
        client = vineyard.connect(*args.rpc_endpoint.split(':'))
    elif args.rpc_host is not None and args.rpc_port is not None:
        client = vineyard.connect(args.rpc_host, args.rpc_port)
    else:
        client = connect_via_config_file()

    return client


def connect_via_config_file():
    """Utility to create a vineyard client using an IPC or RPC socket from config file."""
    try:
        with open(os.path.expanduser('~/.vineyard/config'), encoding='UTF-8') as config_file:
            sockets = config_file.readlines()
        ipc_socket = sockets[0].split(':')[1][:-1]
        rpc_host = sockets[1].split(':')[1][:-1]
        try:
            rpc_port = int(sockets[2].split(':')[1][:-1])
        except ValueError:
            rpc_port = None
        rpc_endpoint = sockets[3].split(':')[1][:-1]
    except BaseException as exc:
        raise Exception('The config file is either not present or not formatted correctly.') from exc
    if ipc_socket:
        client = vineyard.connect(ipc_socket)
        # force use rpc client in cli tools
        client = vineyard.connect(*client.rpc_endpoint.split(':'))
    elif rpc_endpoint:
        client = vineyard.connect(*rpc_endpoint.split(':'))
    elif rpc_host and rpc_port:
        client = vineyard.connect(rpc_host, rpc_port)
    else:
        exit_with_help()

    return client


def as_object_id(object_id):
    """Utility to convert object_id to an integer if possible else convert to a vineyard.ObjectID."""
    try:
        return int(object_id)
    except ValueError:
        return vineyard.ObjectID.wrap(object_id)


def list_object(client, args):
    """Utility to list vineyard objects."""
    try:
        objects = client.list_objects(pattern=args.pattern, regex=args.regex, limit=args.limit)
        print(f"List of your vineyard objects:\n{objects}")
    except BaseException as exc:
        raise Exception('The following error was encountered while listing vineyard objects:') from exc


def query(client, args):
    """Utility to fetch a vineyard object based on object_id."""
    try:
        value = client.get_object(as_object_id(args.object_id))
        print(f"The vineyard object you requested:\n{value}")
    except BaseException as exc:
        if args.exists:
            print(f"The object with object_id({args.object_id}) doesn't exists")
        raise Exception(('The following error was encountered while fetching ' +
                         f'the vineyard object({args.object_id}):')) from exc

    if args.exists:
        print(f'The object with object_id({args.object_id}) exists')
    if args.stdout:
        sys.stdout.write(str(value) + '\n')
    if args.output_file is not None:
        with open(args.output_file, 'w', encoding='UTF-8') as output_file:
            output_file.write(str(value))
    if args.meta is not None:
        if args.meta == 'simple':
            print(f'Meta data of the object:\n{value.meta}')
        elif args.meta == 'json':
            json_meta = json.dumps(value.meta, indent=4)
            print(f'Meta data of the object in JSON format:\n{json_meta}')
    if args.metric is not None:
        print(f'{args.metric}: {getattr(value, args.metric)}')
    if args.tree:
        meta = client.get_meta(as_object_id(args.object_id))
        tree = treelib.Tree()
        get_tree(meta, tree)
        tree.show(line_type="ascii-exr")
    if args.memory_status:
        meta = client.get_meta(as_object_id(args.object_id))
        if args.detail:
            tree = treelib.Tree()
            memory_dict = {}
            get_tree(meta, tree, True, memory_dict)
            tree.show(line_type="ascii-exr")
            print(f'The object taking the maximum memory is:\n{max(memory_dict, key=lambda x: memory_dict[x])}')
        print(f'The total memory used: {pretty_format_memory(get_memory_used(meta))}')


def delete_object(client, args):
    """Utility to delete a vineyard object based on object_id."""
    if args.object_id is not None:
        try:
            client.delete(object_id=as_object_id(args.object_id), force=args.force, deep=args.deep)
            print(f'The vineyard object({args.object_id}) was deleted successfully')
        except BaseException as exc:
            raise Exception(('The following error was encountered while deleting ' +
                             f'the vineyard object({args.object_id}):')) from exc
    elif args.regex_pattern is not None:
        try:
            objects = client.list_objects(pattern=args.regex_pattern, regex=True, limit=5)
        except BaseException as exc:
            raise Exception('The following error was encountered while listing vineyard objects:') from exc
        for obj in objects:
            try:
                client.delete(object_id=as_object_id(obj.id), force=args.force, deep=args.deep)
                print(f'The vineyard object({obj.id}) was deleted successfully')
            except BaseException as exc:
                raise Exception(('The following error was encountered while deleting ' +
                                 f'the vineyard object({obj.id}):')) from exc
    else:
        exit_with_help()


def status(client, args):
    """Utility to print the status of connected vineyard server."""
    stat = client.status
    if args.properties is None:
        print(stat)
    else:
        print('InstanceStatus:')
        for prop in args.properties:
            print(f'    {prop}: {getattr(stat, prop)}')


def put_object(client, args):
    """Utility to put python value to vineyard server."""
    if args.value is not None:
        try:
            value = args.value
            client.put(value)
            print(f'{value} was successfully put to vineyard server')
        except BaseException as exc:
            raise Exception(
                (f'The following error was encountered while putting {args.value} to vineyard server:')) from exc
    elif args.file is not None:
        try:
            value = pd.read_csv(args.file, sep=args.sep, delimiter=args.delimiter, header=args.header)
            client.put(value)
            print(f'{value} was successfully put to vineyard server')
        except BaseException as exc:
            raise Exception(('The following error was encountered while putting ' +
                             f'{args.file} as pandas dataframe to vineyard server:')) from exc


def head(client, args):
    """Utility to print the first n lines of a vineyard object."""
    try:
        value = client.get(as_object_id(args.object_id))
        if isinstance(value, pd.DataFrame):
            print(value.head(args.limit))
        else:
            print("'head' is currently supported for a pandas dataframe only.")
    except BaseException as exc:
        raise Exception(('The following error was encountered while fetching ' +
                         f'the vineyard object({args.object_id}):')) from exc


def copy(client, args):
    """Utility to copy a vineyard object."""
    if args.shallow:
        object_id = client.shallow_copy(as_object_id(args.object_id))
        print(f'The object({args.object_id}) was succesfully copied to {object_id}')
    elif args.deep:
        object_id = client.deep_copy(as_object_id(args.object_id))
        print(f'The object({args.object_id}) was succesfully copied to {object_id}')
    else:
        exit_with_help()


def migrate_object(client, args):
    """Utility to migrate a vineyard object."""
    client1 = client

    if args.ipc_socket_value is not None:
        client2 = vineyard.connect(args.ipc_socket_value)
        # force use rpc client in cli tools
        client2 = vineyard.connect(*client.rpc_endpoint.split(':'))
    elif args.rpc_endpoint_value is not None:
        client2 = vineyard.connect(*args.rpc_endpoint_value.split(':'))
    elif args.rpc_host_value is not None and args.rpc_port_value is not None:
        client2 = vineyard.connect(args.rpc_host_value, args.rpc_port_value)
    else:
        raise Exception("You neither provided an IPC value nor a RPC value for the second client")

    object_id = as_object_id(args.object_id)

    try:
        if args.local:
            return_object_id = client1.migrate(object_id)
        if args.remote:
            return_object_id = client2.migrate(object_id)
        print(f'The vineyard object({args.object_id}) was migrateed successfully')
        print(f'Returned object ID - {return_object_id}')
    except BaseException as exc:
        raise Exception(('The following error was encountered while migrating ' +
                         f'the vineyard object({args.object_id}):')) from exc


def debug(client, args):
    """Utility to issue a debug request."""
    try:
        result = client.debug(args.payload)
    except BaseException as exc:
        raise Exception(('The following error was encountered during the debug' +
                         f' request with payload, {args.payload}:')) from exc
    print(f'The result returned by the debug handler:\n{result}')


def get_tree(meta, tree, memory=False, memory_dict=None, parent=None):
    """Utility to display object lineage in a tree like form."""
    node = f'{meta["typename"]} <{meta["id"]}>'
    if memory:
        memory_used = pretty_format_memory(meta["nbytes"])
        node += f': {memory_used}'
        memory_dict[node] = memory_used
    tree.create_node(node, node, parent=parent)
    parent = node
    for key in meta:
        if isinstance(meta[key], vineyard._C.ObjectMeta):
            get_tree(meta[key], tree, memory, memory_dict, parent)


def get_memory_used(meta):
    """Utility to get the memory used by the vineyard object."""
    total_bytes = 0
    for key in meta:
        if isinstance(meta[key], vineyard._C.ObjectMeta):
            if str(meta[key]['typename']) == 'vineyard::Blob':
                size = meta[key]['length']
                total_bytes += size
            else:
                total_bytes += get_memory_used(meta[key])
    return total_bytes


def pretty_format_memory(nbytes):
    """Utility to return memory with appropriate unit."""
    if nbytes < (1 << 10):
        return f'{nbytes} bytes'
    if (1 << 20) > nbytes > (1 << 10):
        return f'{nbytes / (1 << 10)} KB'
    if (1 << 30) > nbytes > (1 << 20):
        return f'{nbytes / (1 << 20)} MB'
    if nbytes > (1 << 30):
        return f'{nbytes / (1 << 30)} GB'


def config(args):
    """Utility to edit the config file."""
    with open(os.path.expanduser('~/.vineyard/config'), encoding='UTF-8') as config_file:
        sockets = config_file.readlines()
    with open(os.path.expanduser('~/.vineyard/config'), 'w', encoding='UTF-8') as config_file:
        if args.ipc_socket_value is not None:
            sockets[0] = f'ipc_socket:{args.ipc_socket_value}\n'
        if args.rpc_host_value is not None:
            sockets[1] = f'rpc_host:{args.rpc_host_value}\n'
        if args.rpc_port_value is not None:
            sockets[2] = f'rpc_port:{args.rpc_port_value}\n'
        if args.rpc_endpoint_value is not None:
            sockets[3] = f'rpc_endpoint:{args.rpc_endpoint_value}'
        config_file.writelines(sockets)


def start_vineyardd(args):
    """Utility to start vineyardd."""
    if args.local:
        vineyard.deploy.local.start_vineyardd(etcd_endpoints=args.etcd_endpoints,
                                              vineyardd_path=args.vineyardd_path,
                                              size=args.size,
                                              socket=args.socket,
                                              rpc_socket_port=args.rpc_socket_port,
                                              debug=args.debug)
    elif args.distributed:
        vineyard.deploy.distributed.start_vineyardd(hosts=args.hosts,
                                                    etcd_endpoints=args.etcd_endpoints,
                                                    vineyardd_path=args.vineyardd_path,
                                                    size=args.size,
                                                    socket=args.socket,
                                                    rpc_socket_port=args.rpc_socket_port,
                                                    debug=args.debug)


def main():
    """Main function for vineyard-ctl."""
    args = optparser.parse_args()
    if args.cmd is None:
        return exit_with_help()

    if args.cmd == 'config':
        return config(args)
    if args.cmd == 'start':
        return start_vineyardd(args)

    client = connect_vineyard(args)

    if args.cmd == 'ls':
        return list_object(client, args)
    if args.cmd == 'query':
        return query(client, args)
    if args.cmd == 'del':
        return delete_object(client, args)
    if args.cmd == 'stat':
        return status(client, args)
    if args.cmd == 'put':
        return put_object(client, args)
    if args.cmd == 'head':
        return head(client, args)
    if args.cmd == 'copy':
        return copy(client, args)
    if args.cmd == 'migrate':
        return migrate_object(client, args)
    if args.cmd == 'debug':
        return debug(client, args)

    return exit_with_help()


if __name__ == "__main__":
    main()
