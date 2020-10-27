#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020 Alibaba Group Holding Limited.
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

from argparse import ArgumentParser
import sys

import vineyard


def vineyard_argument_parser():
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--ipc_socket')
    parser.add_argument('--rpc_host')
    parser.add_argument('--rpc_port', type=int)
    parser.add_argument('--rpc_endpoint')
    cmd_parsers = parser.add_subparsers(title='commands', dest='cmd')

    ls_opt = cmd_parsers.add_parser('ls', parents=[parser], add_help=False, help='List objects')
    ls_opt.add_argument('--limit', default=5, type=int)

    get_opt = cmd_parsers.add_parser('get', parents=[parser], add_help=False, help='Get object')
    get_opt.add_argument('--object_id')

    del_opt = cmd_parsers.add_parser('del', parents=[parser], add_help=False, help='Delete object')
    del_opt.add_argument('--object_id')
    del_opt.add_argument('--recursive', default=False, type=bool)
    del_opt.add_argument('--force', default=False, type=bool)

    return parser


optparser = vineyard_argument_parser()


def exit_with_help():
    optparser.print_help(sys.stderr)
    sys.exit(-1)


__vineyard_client = None


def connect_vineyard(args):
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

    global __vineyard_client
    __vineyard_client = client
    return client


def as_object_id(object_id):
    try:
        return int(object_id)
    except ValueError:
        return vineyard.ObjectID.wrap(object_id)


def ls(client, limit):
    objects = client.list(limit=limit)
    print(objects)


def get(client, object_id):
    if object_id is None:
        exit_with_help()
    value = client.get(as_object_id(object_id))
    print(value)


def delete(client, object_id, recursive):
    if object_id is None:
        exit_with_help()
    client.delete(as_object_id(object_id), deep=recursive)


def main():
    args = optparser.parse_args()
    if args.cmd is None:
        exit_with_help()
    client = connect_vineyard(args)
    if args.cmd == 'ls':
        return ls(client, args.limit)
    elif args.cmd == 'get':
        return get(client, args.object_id)
    elif args.cmd == 'del':
        return delete(client, args.object_id, args.recursive)


if __name__ == "__main__":
    main()
