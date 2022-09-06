#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020-2022 Alibaba Group Holding Limited.
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

import argparse
import os
import subprocess
import sys


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Installing vineyard as a systemd service.'
    )

    parsers = parser.add_subparsers(
        title='subcommands',
        dest="command",
        description='valid subcommands',
        help='additional help',
    )
    install = parsers.add_parser('install', help='Install the systemd service.')
    install.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='Force install the systemd service.',
    )
    uninstall = parsers.add_parser('uninstall', help='Uninstall the systemd service.')
    uninstall.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='Force uninstall the systemd service.',
    )

    parsers.add_parser('enable', help='Enable the systemd service.')
    parsers.add_parser('disable', help='Disable the systemd service.')

    parsers.add_parser('start', help='Start the systemd service.')
    parsers.add_parser('stop', help='Stop the systemd service.')
    parsers.add_parser('reload', help='Reload the systemd service.')
    parsers.add_parser('restart', help='Restart the systemd service.')

    parsers.add_parser('status', help='Show the status of the systemd service.')

    parser.add_argument(
        '-u',
        '--user',
        action='store_true',
        help='Interactive with the systemd user service.',
    )
    return parser.parse_args(sys.argv[1:])


service_file_template = """# Vineyard as a systemd service
[Unit]
Description=Vineyard - an in-memory immutable data manager

[Service]
Type=idle
Restart=no
ExecStart=/usr/bin/env python3 -m vineyard -flagfile {config}
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"""

config_file_template = """--allocator=dlmalloc
--size=4Gi
--socket={default_socket}
"""

root_service_file = '/etc/systemd/system/vineyard.service'
root_config_file = '/etc/vineyard.conf'

user_service_file = os.path.expanduser('~/.config/systemd/user/vineyard.service')
user_config_file = os.path.expanduser('~/.config/vineyard.conf')


def systemd_install_service(args, user_args):
    if args.user:
        service_file = user_service_file
        config_file = user_config_file
        service_file_cotent = service_file_template.format(config=config_file)
        config_file_content = config_file_template.format(
            default_socket=os.path.expanduser('~/.vineyard.sock')
        )
    else:
        service_file = root_service_file
        config_file = root_config_file
        service_file_cotent = service_file_template.format(config=config_file)
        config_file_content = config_file_template.format(
            default_socket=os.path.expanduser('~/.vineyard.sock')
        )

    if args.force or not os.path.exists(service_file):
        os.makedirs(os.path.dirname(service_file), exist_ok=True)
        with open(service_file, 'w') as f:
            f.write(service_file_cotent)
    print('vineyard: systemd service file installed at {}'.format(service_file))

    if args.force or not os.path.exists(config_file):
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            f.write(config_file_content)
    print('vineyard: systemd service config file installed at {}'.format(service_file))

    # if args.user:
    #     subprocess.call(['journalctl', '--user-unit', 'vineyard'])
    # else:
    #     subprocess.call(['journalctl', 'vineyard'])

    # reload
    subprocess.call(['systemctl', *user_args, 'daemon-reload'])


def systemd_uninstall_service(args, user_args):
    if args.user:
        service_file = user_service_file
        config_file = user_config_file
    else:
        service_file = root_service_file
        config_file = root_config_file

    if os.path.exists(service_file):
        os.remove(service_file)
    if os.path.exists(config_file):
        os.remove(config_file)

    # reload
    subprocess.call(['systemctl', *user_args, 'daemon-reload'])


def systemd_enable_service(args, user_args):
    subprocess.call(['systemctl', *user_args, 'enable', 'vineyard'])


def systemd_disable_service(args, user_args):
    subprocess.call(['systemctl', *user_args, 'disable', 'vineyard'])


def systemd_start_service(args, user_args):
    subprocess.call(['systemctl', *user_args, 'start', 'vineyard'])


def systemd_stop_service(args, user_args):
    subprocess.call(['systemctl', *user_args, 'stop', 'vineyard'])


def systemd_reload_service(args, user_args):
    subprocess.call(['systemctl', *user_args, 'reload', 'vineyard'])


def systemd_restart_service(args, user_args):
    subprocess.call(['systemctl', *user_args, 'restart', 'vineyard'])


def systemd_status_service(args, user_args):
    subprocess.call(['systemctl', *user_args, 'status', 'vineyard'])


def main():
    args = parse_args()
    user_args = []
    if args.user:
        user_args.append('--user')
    if args.command == 'install':
        systemd_install_service(args, user_args)
    elif args.command == 'uninstall':
        systemd_uninstall_service(args, user_args)
    elif args.command == 'enable':
        systemd_enable_service(args, user_args)
    elif args.command == 'disable':
        systemd_disable_service(args, user_args)
    elif args.command == 'start':
        systemd_start_service(args, user_args)
    elif args.command == 'stop':
        systemd_stop_service(args, user_args)
    elif args.command == 'reload':
        systemd_reload_service(args, user_args)
    elif args.command == 'restart':
        systemd_restart_service(args, user_args)
    elif args.command == 'status':
        systemd_status_service(args, user_args)
    else:
        raise ValueError('Invalid command: {}'.format(args.command))


if __name__ == '__main__':
    try:
        subprocess.check_output(['systemctl', '--version'])
    except Exception:  # pylint: disable=broad-except
        print('Vineyard requires systemd to be installed as a systemd service.')
        sys.exit(1)
    main()
