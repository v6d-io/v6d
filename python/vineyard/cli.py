#! /usr/bin/env python
# -*- coding: utf-8 -*-
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

from argparse import ArgumentParser
import sys

import vineyard


def vineyard_argument_parser():
    """Utility to create a command line Argument Parser."""
    parser = ArgumentParser(prog='vineyard-ctl',
                            usage='%(prog)s [options]',
                            description='vineyard-ctl: A command line tool for vineyard',
                            allow_abbrev=False)

    connection_group = parser.add_mutually_exclusive_group(required=True)
    connection_group.add_argument('--ipc_socket', help='Socket location of connected vineyard server')
    connection_group.add_argument('--rpc_host', help='RPC HOST of the connected vineyard server')
    connection_group.add_argument('--rpc_port', type=int, help='RPC PORT of the connected vineyard server')
    connection_group.add_argument('--rpc_endpoint', help='RPC endpoint of the connected vineyard server')

    cmd_parser = parser.add_subparsers(title='commands', dest='cmd')

    ls_opt = cmd_parser.add_parser('ls', add_help=False, help='List vineyard objects')
    ls_opt.add_argument('--pattern',
                        default='*',
                        type=str,
                        help='The pattern string that will be matched against the object’s typename')
    ls_opt.add_argument('--regex',
                        action='store_true',
                        help='The pattern string will be considered as a regex expression')
    ls_opt.add_argument('--limit', default=5, type=int, help='The limit to list')

    get_opt = cmd_parser.add_parser('get', add_help=False, help='Get a vineyard object')
    get_opt.add_argument('--object_id', required=True, help='ID of the object to be fetched')

    del_opt = cmd_parser.add_parser('del', add_help=False, help='Delete a vineyard object')
    del_opt.add_argument('--object_id', required=True, help='ID of the object to be deleted')
    del_opt.add_argument('--force',
                         action='store_true',
                         help='Recursively delete even if the member object is also referred by others')
    del_opt.add_argument('--deep',
                         action='store_true',
                         help='Deeply delete an object means we will deleting the members recursively')

    stat_opt = cmd_parser.add_parser('stat', add_help=False, help='Get the status of connected vineyard server')
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
        exit_with_help()

    return client


def as_object_id(object_id):
    """Utility to convert object_id to an integer if possible else convert to a vineyard.ObjectID."""
    try:
        return int(object_id)
    except ValueError:
        return vineyard.ObjectID.wrap(object_id)


def list_obj(client, pattern, regex, limit):
    """Utility to list vineyard objects."""
    try:
        objects = client.list_objects(pattern=pattern, regex=regex, limit=limit)
        print(f"List of your vineyard objects:\n{objects}")
    except BaseException as exc:
        raise Exception(f'The following error was encountered while listing vineyard objects:\n{exc}') from exc


def get(client, object_id):
    """Utility to fetch a vineyard object based on object_id."""
    try:
        value = client.get_object(as_object_id(object_id))
        print(f"The vineyard object you requested:\n{value}")
    except BaseException as exc:
        raise Exception(('The following error was encountered while fetching ' +
                         f'the vineyard object({object_id}):\n{exc}')) from exc


def delete_obj(client, object_id, force, deep):
    """Utility to delete a vineyard object based on object_id."""
    try:
        client.delete(object_id=as_object_id(object_id), force=force, deep=deep)
        print(f'The vineyard object({object_id}) was deleted successfully')
    except BaseException as exc:
        raise Exception(('The following error was encountered while deleting ' +
                         f'the vineyard object({object_id}):\n{exc}')) from exc


def status(client, properties):
    """Utility to print the status of connected vineyard server."""
    stat = client.status
    if properties is None:
        print(stat)
    else:
        print('InstanceStatus:')
        for prop in properties:
            print(f'    {prop}: {getattr(stat, prop)}')


def main():
    """Main function for vineyard-ctl."""
    args = optparser.parse_args()
    if args.cmd is None:
        return exit_with_help()

    client = connect_vineyard(args)

    if args.cmd == 'ls':
        return list_obj(client, args.pattern, args.regex, args.limit)
    if args.cmd == 'get':
        return get(client, args.object_id)
    if args.cmd == 'del':
        return delete_obj(client, args.object_id, args.force, args.deep)
    if args.cmd == 'stat':
        return status(client, args.properties)

    return exit_with_help()


if __name__ == "__main__":
    main()
