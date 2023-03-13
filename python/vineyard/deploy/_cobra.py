#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2020-2023 Alibaba Group Holding Limited.
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

import json
import logging
import subprocess
import textwrap

from makefun import with_signature

logger = logging.getLogger('vineyard')


def infer_name(name):
    cs = []
    for c in name:
        if not c.isalnum():
            cs.append("_")
        elif c.isupper():
            cs.append(c.lower())
        else:
            cs.append(c)
    return "".join(cs)


def infer_argument(flag):
    arg, default_value = infer_name(flag["Name"]), flag['Default']
    if default_value:
        # see: pflag's `Flag::Type()` methods.
        if flag["Type"] != "string":
            default_value = json.loads(default_value)
    return arg, default_value, repr(default_value)


def infer_signature(name, usage, exclude_args):
    kwargs, arguments, defaults, argument_docs = [], {}, {}, []
    for flag in usage["GlobalFlags"] + usage["Flags"]:
        argument_name, default_value, default_value_rep = infer_argument(flag)
        if exclude_args and (
            argument_name in exclude_args or flag["Name"] in exclude_args
        ):
            continue
        arguments[argument_name] = flag['Name']
        kwargs.append(argument_name + "=" + default_value_rep)
        defaults[argument_name] = default_value

        argument_docs.append(argument_name + ":")
        argument_help = flag["Help"]
        if argument_help:
            if argument_help[-1] != ".":
                argument_help += "."
        if default_value_rep:
            argument_help += " Defaults to " + default_value_rep + "."
        argument_docs.append(textwrap.indent(argument_help, "    "))
    kwargs.append('capture=False')  # allow capturing the stdout
    return name + "(" + ", ".join(kwargs) + ")", arguments, defaults, argument_docs


def infer_docstring(name, usage, argument_docs):
    blocks = []
    if usage["Brief"]:
        blocks.append(usage["Brief"])
    if usage["Description"]:
        blocks.append(usage["Description"])
    if usage["Usage"]:
        blocks.append("Usage:\n" + textwrap.indent(usage["Usage"], "    "))
    if usage["Example"]:
        blocks.append("Example:\n" + textwrap.indent(usage["Example"], "    "))
    if not usage["Runnable"]:
        blocks.append("This command is not runnable, use subroutines instead.")
    if usage["Children"]:
        children = []
        for child in usage["Children"]:
            child_doc = name + "." + infer_name(child["Name"])
            if child["Brief"]:
                child_doc += ": " + child["Brief"]
            children.append(child_doc)
        blocks.append("Subroutines:\n" + textwrap.indent("\n".join(children), "    - "))
    if argument_docs:
        blocks.append(
            "Parameters:\n" + textwrap.indent("\n".join(argument_docs), "    ")
        )
    return "\n\n".join(blocks)


def make_command(executable, usage, exclude_args, scope=None):
    name = infer_name(usage["Name"])
    signature, arguments, defaults, argument_docs = infer_signature(
        name, usage, exclude_args
    )
    doc = infer_docstring(name, usage, argument_docs)

    @with_signature(signature, func_name=name, doc=doc)
    def cmd(*args, capture=False, **kwargs):
        if usage["Runnable"]:
            command_and_args = [executable]
            command_and_args.extend(scope)
            # FIXME: how to proceed with non-parameter arguments?
            # for arg in args:
            #     command_and_args.append("--" + arg)
            for key, value in kwargs.items():
                # get(key, default...) won't work, as we cannot diff not exists and None
                if key in defaults and value == defaults[key]:
                    continue
                if not isinstance(value, str):
                    value = json.dumps(value)
                command_and_args.append("--%s=%s" % (arguments.get(key, key), value))
            logger.debug('Executing: %s', ' '.join(command_and_args))
            parameters = {
                'args': command_and_args,
                'universal_newlines': True,
                'encoding': "utf-8",
                'errors': "ignore",
                'shell': False,
            }
            if capture:
                output = subprocess.check_output(
                    **parameters,
                )
                if output:
                    output = output.strip()
                return output
            else:
                return subprocess.check_call(
                    **parameters,
                    bufsize=1,
                    stderr=subprocess.STDOUT,
                )

    # generate subcommands
    if scope is None:
        scope = []
    for child in usage["Children"]:
        child_scope = scope + [child['Name']]
        child_name, child_command = make_command(
            executable, child, exclude_args, child_scope
        )
        setattr(cmd, child_name, child_command)

    return name, cmd


def click(executable, usage_args=None, exclude_args=None):
    if usage_args is None:
        usage_args = ["-j"]
    if not isinstance(usage_args, (list, tuple)):
        usage_args = [usage_args]
    usage = json.loads(
        subprocess.check_output(
            [executable] + usage_args,
            universal_newlines=True,
            encoding="utf-8",
            errors="ignore",
            shell=False,
            stderr=subprocess.STDOUT,
        )
    )
    _, cmd = make_command(executable, usage, exclude_args)
    return cmd
