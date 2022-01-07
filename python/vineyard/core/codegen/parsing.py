#!/usr/bin/env python3
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

import copy
import itertools
import logging
import os

DEP_MISSING_ERROR = '''
    Dependencies {dep} cannot be found, please try again after:

        pip3 install {dep}

'''

try:
    import clang.cindex
    from clang.cindex import CursorKind
    from clang.cindex import TranslationUnit
except ImportError:
    raise RuntimeError(DEP_MISSING_ERROR.format(dep='libclang'))

try:
    import parsec
except ImportError:
    raise RuntimeError(DEP_MISSING_ERROR.format(dep='parsec'))

###############################################################################
#
# parse codegen spec
#
#   __attribute__((annotate("codegen"))):
#       meta codegen
#
#   __attribute__((annotate("codegen:Type"))):
#       member type: Type member_
#
#   __attribute__((annotate("codegen:Type*"))):
#       member type: std::shared_ptr<Type> member_
#
#   __attribute__((annotate("codegen:[Type*]"))):
#       list member type: std::vector<Type> member_
#
#   __attribute__((annotate("codegen:{Type}"))):
#       set member type: std::set<Type> member_
#
#   __attribute__((annotate("codegen:{Type*}"))):
#       set member type: std::set<std::shared_ptr<Type>> member_
#
#   __attribute__((annotate("codegen:{int32_t: Type}"))):
#       dict member type: std::map<int32_t, Type> member_
#
#   __attribute__((annotate("codegen:{int32_t: Type*}"))):
#       dict member type: std::map<int32_t, std::shared_ptr<Type>> member_
#
# FIXME(hetao): parse the codegen spec directly from the type signature of
# the member variable
#


class CodeGenKind:
    def __init__(self, kind='meta', element_type=None):
        self.kind = kind
        if element_type is None:
            self.element_type = None
            self.star = ''
        else:
            if isinstance(element_type[0], tuple):
                self.element_type = (element_type[0][0], element_type[1][0])
                self.star = element_type[1][1]
            else:
                self.element_type = element_type[0]
                self.star = element_type[1]
        if self.star:
            self.deref = ''
        else:
            self.deref = '*'

    @property
    def is_meta(self):
        return self.kind == 'meta'

    @property
    def is_plain(self):
        return self.kind == 'plain'

    @property
    def is_set(self):
        return self.kind == 'set'

    @property
    def is_list(self):
        return self.kind == 'list'

    @property
    def is_dlist(self):
        return self.kind == 'dlist'

    @property
    def is_dict(self):
        return self.kind == 'dict'

    def __repr__(self):
        star_str = '*' if self.star else ''
        if self.is_meta:
            return 'meta'
        if self.is_plain:
            return '%s%s' % (self.element_type, star_str)
        if self.is_list:
            return '[%s%s]' % (self.element_type, star_str)
        if self.is_dlist:
            return '[[%s%s]]' % (self.element_type, star_str)
        if self.is_set:
            return '{%s%s}' % (self.element_type, star_str)
        if self.is_dict:
            return '{%s: %s%s}' % (self.element_type[0], self.element_type[1], star_str)
        raise RuntimeError('Invalid codegen kind: %s' % self.kind)


name_pattern = (
    parsec.spaces()
    >> parsec.regex(r'[_a-zA-Z][_a-zA-Z0-9<>, ]*(::[_a-zA-Z][_a-zA-Z0-9<>, ]*)*')
    << parsec.spaces()
)

star_pattern = (
    parsec.spaces() >> parsec.optional(parsec.string('*'), '') << parsec.spaces()
)

parse_meta = parsec.spaces().parsecmap(lambda _: CodeGenKind('meta'))

parse_plain = (
    parsec.spaces() >> (name_pattern + star_pattern) << parsec.spaces()
).parsecmap(lambda value: CodeGenKind('plain', value))
parse_list = (
    parsec.string('[') >> (name_pattern + star_pattern) << parsec.string(']')
).parsecmap(lambda value: CodeGenKind('list', value))
parse_dlist = (
    parsec.string('[[') >> (name_pattern + star_pattern) << parsec.string(']]')
).parsecmap(lambda value: CodeGenKind('dlist', value))
parse_set = (
    parsec.string('{') >> (name_pattern + star_pattern) << parsec.string('}')
).parsecmap(lambda value: CodeGenKind('set', value))
parse_dict = (
    parsec.string('{')
    >> parsec.separated((name_pattern + star_pattern), parsec.string(':'), 2, 2)
    << parsec.string('}')
).parsecmap(lambda values: CodeGenKind('dict', tuple(values)))

codegen_spec_parser = (
    parse_dict ^ parse_set ^ parse_dlist ^ parse_list ^ parse_plain ^ parse_meta
)


def parse_codegen_spec(kind):
    if kind.startswith('vineyard'):
        kind = kind[len('vineyard') :]
    if kind.startswith('codegen'):
        kind = kind[len('codegen') :]
    if kind.startswith(':'):
        kind = kind[1:]
    return codegen_spec_parser.parse(kind)


###############################################################################
#
# dump the AST for debugging
#


def dump_ast(node, indent, saw, base_indent=4, include_refs=False):
    def is_std_ns(node):
        return node.kind == CursorKind.NAMESPACE and node.spelling == 'std'

    k = node.kind  # type: clang.cindex.CursorKind
    # skip printting UNEXPOSED_*
    if not k.is_unexposed():
        tpl = '{indent}{kind}{name}{type_name}'
        if node.spelling:
            name = ' s: %s' % node.spelling
        else:
            name = ''
        if node.type and node.type.spelling:
            type_name = ', t: %s' % node.type.spelling
        else:
            type_name = ''

        # FIXME: print opcode or literal

        print(
            tpl.format(indent=' ' * indent, kind=k.name, name=name, type_name=type_name)
        )

    saw.add(node.hash)
    if include_refs:
        if node.referenced is not None and node.referenced.hash not in saw:
            dump_ast(
                node.referenced, indent + base_indent, saw, base_indent, include_refs
            )

    # FIXME: skip auto generated decls
    skip = len([c for c in node.get_children() if indent == 0 and is_std_ns(c)])
    for c in node.get_children():
        if not skip:
            dump_ast(c, indent + base_indent, saw, base_indent, include_refs)
        if indent == 0 and is_std_ns(c):
            skip -= 1
    saw.remove(node.hash)


class ParseOption:
    Default = 0x0
    DetailedPreprocessingRecord = 0x01
    Incomplete = 0x02
    PrecompiledPreamble = 0x04
    CacheCompletionResults = 0x08
    ForSerialization = 0x10
    CXXChainedPCH = 0x20
    SkipFunctionBodies = 0x40
    IncludeBriefCommentsInCodeCompletion = 0x80
    CreatePreambleOnFirstParse = 0x100
    KeepGoing = 0x200
    SingleFileParse = 0x400
    LimitSkipFunctionBodiesToPreamble = 0x800
    IncludeAttributedTypes = 0x1000
    VisitImplicitAttributes = 0x2000
    IgnoreNonErrorsFromIncludedFiles = 0x4000
    RetainExcludedConditionalBlocks = 0x8000


###############################################################################
#
# AST utils
#


def check_serialize_attribute(node):
    for child in node.get_children():
        if child.kind == CursorKind.ANNOTATE_ATTR:
            for attr_kind in ['vineyard', 'no-vineyard', 'codegen']:
                if child.spelling.startswith(attr_kind):
                    return child.spelling
    return None


def check_if_class_definition(node):
    for child in node.get_children():
        if child.kind in [
            CursorKind.CXX_BASE_SPECIFIER,
            CursorKind.CXX_ACCESS_SPEC_DECL,
            CursorKind.CXX_METHOD,
            CursorKind.FIELD_DECL,
        ]:
            return True
    return False


def filter_the_module(root, filepath):
    children = []
    for child in root.get_children():
        if (
            child.location
            and child.location.file
            and child.location.file.name == filepath
        ):
            children.append(child)
    return children


def traverse(node, to_reflect, to_include, namespaces=None):
    '''Traverse the AST tree.'''
    if node.kind in [
        CursorKind.CLASS_DECL,
        CursorKind.CLASS_TEMPLATE,
        CursorKind.STRUCT_DECL,
    ]:
        # codegen for all top-level classes (definitions, not declarations) in
        # the given file.
        if check_if_class_definition(node):
            attr = check_serialize_attribute(node)
            if attr is None or 'no-vineyard' not in attr:
                to_reflect.append(('vineyard', namespaces, node))

    if node.kind == CursorKind.INCLUSION_DIRECTIVE:
        to_include.append(node)

    if node.kind in [CursorKind.TRANSLATION_UNIT, CursorKind.NAMESPACE]:
        if node.kind == CursorKind.NAMESPACE:
            if namespaces is None:
                namespaces = []
            else:
                namespaces = copy.copy(namespaces)
            namespaces.append(node.spelling)
        for child in node.get_children():
            traverse(child, to_reflect, to_include, namespaces=namespaces)


def find_fields(definition):
    fields, using_alias, first_mmeber_offset, has_post_construct = [], [], -1, False
    for child in definition.get_children():
        if first_mmeber_offset == -1:
            if child.kind not in [
                CursorKind.TEMPLATE_TYPE_PARAMETER,
                CursorKind.CXX_BASE_SPECIFIER,
                CursorKind.ANNOTATE_ATTR,
            ]:
                first_mmeber_offset = child.extent.start.offset

        if child.kind == CursorKind.FIELD_DECL:
            attr = check_serialize_attribute(child)
            if attr:
                fields.append((attr, child))
            continue

        if child.kind == CursorKind.TYPE_ALIAS_DECL:
            using_alias.append((child.spelling, child.extent))
            continue

        if (
            not has_post_construct
            and child.kind == CursorKind.CXX_METHOD
            and child.spelling == 'PostConstruct'
        ):
            for body in child.get_children():
                if body.kind == CursorKind.CXX_OVERRIDE_ATTR:
                    has_post_construct = True
    return fields, using_alias, first_mmeber_offset, has_post_construct


def check_class(node):
    template_parameters = []
    for child in node.get_children():
        if child.kind == CursorKind.TEMPLATE_TYPE_PARAMETER:
            template_parameters.append((child.spelling, child.extent))
    return node.spelling, template_parameters


def generate_template_header(ts):
    if not ts:
        return ''
    ps = []
    for t in ts:
        if t.startswith('typename'):
            ps.append(t)
        else:
            ps.append('typename %s' % t)
    return 'template<{ps}>'.format(ps=', '.join(ps))


def generate_template_type(name, ts):
    if not ts:
        return name
    return '{name}<{ps}>'.format(name=name, ps=', '.join(ts))


def parse_compilation_database(build_directory):
    # check if the file exists first to suppress the clang warning.
    compile_commands_json = os.path.join(build_directory, 'compile_commands.json')
    if not os.path.isfile(compile_commands_json) or not os.access(
        compile_commands_json, os.R_OK
    ):
        return None
    try:
        return clang.cindex.CompilationDatabase.fromDirectory(build_directory)
    except clang.cindex.CompilationDatabaseError:
        return None


def validate_and_strip_input_file(source):
    if not os.path.isfile(source) or not os.access(source, os.R_OK):
        return None, 'File not exists'
    with open(source, 'r') as fp:
        content = fp.read().splitlines(keepends=False)
    # TODO: valid and remove the first line
    return '\n'.join(content), ''


def strip_flags(flags):
    stripped_flags = []
    for flag in flags:
        if flag == '-c' or flag.startswith('-O') or flags == '-Werror':
            continue
        stripped_flags.append(flag)
    return stripped_flags


def resolve_include(inc_node, system_includes, includes):
    inc_name = inc_node.spelling
    if not inc_name.endswith('.vineyard.h'):  # os.path.splitext won't work
        return None
    mod_name = inc_name[: -len(".vineyard.h")] + ".vineyard-mod"
    for inc in itertools.chain(system_includes, includes):
        target = os.path.join(inc, mod_name)
        if os.path.isfile(target) and os.access(target, os.R_OK):
            return os.path.join(inc, inc_name)
    return None


def parse_module(  # noqa: C901
    root_directory,
    source,
    target=None,
    system_includes=None,
    includes=None,
    extra_flags=None,
    build_directory=None,
    delayed=True,
    parse_only=True,
    verbose=False,
):
    # prepare inputs
    content, message = validate_and_strip_input_file(source)
    if content is None:
        raise RuntimeError('Invalid input: %s' % message)
    unsaved_files = [(source, content)]

    # NB:
    #   `-nostdinc` and `-nostdinc++`: to avoid libclang find an incorrect
    #                                  gcc installation.
    #   `-Wunused-private-field`: we skip parsing the function bodies.
    base_flags = [
        '-x',
        'c++',
        '-std=c++14',
        '-nostdinc',
        '-nostdinc++',
        '-Wno-unused-private-field',
    ]

    # prepare flags
    flags = None
    compliation_db = parse_compilation_database(build_directory)
    if compliation_db is not None:
        commands = compliation_db.getCompileCommands(source)
        if commands is not None and len(commands) > 0:
            # strip flags
            flags = strip_flags(list(commands[0].arguments)[1:-1])

            # NB: even use compilation database we still needs to include the
            # system includes, since we `-nostdinc{++}`.
            if system_includes:
                for inc in system_includes.split(';'):
                    flags.append('-isystem')
                    flags.append(inc)

            if extra_flags:
                flags.extend(extra_flags)

    if flags is None:
        flags = []
        if system_includes:
            for inc in system_includes.split(';'):
                flags.append('-isystem')
                flags.append(inc)
        if includes:
            for inc in includes.split(';'):
                flags.append('-I%s' % inc)
        if extra_flags:
            flags.extend(extra_flags)

        if delayed:
            flags.append('-fdelayed-template-parsing')
        else:
            flags.append('-fno-delayed-template-parsing')

    # parse
    index = clang.cindex.Index.create()
    options = (
        ParseOption.Default
        | ParseOption.DetailedPreprocessingRecord
        | ParseOption.SkipFunctionBodies
        | ParseOption.IncludeAttributedTypes
        | ParseOption.KeepGoing
    )

    if parse_only:
        options |= ParseOption.SingleFileParse

    parse_flags = base_flags + flags
    unit = index.parse(
        source, unsaved_files=unsaved_files, args=parse_flags, options=options
    )

    if not parse_only:
        for diag in unit.diagnostics:
            if verbose or (
                diag.location
                and diag.location.file
                and diag.location.file.name == source
            ):
                logging.warning(diag)

    # traverse
    modules = filter_the_module(unit.cursor, source)
    to_reflect, to_include = [], []
    for module in modules:
        if verbose:
            dump_ast(module, 0, set())
        traverse(module, to_reflect, to_include)

    return content, to_reflect, to_include, parse_flags


def parse_deps(
    root_directory,
    source,
    target=None,
    system_includes=None,
    includes=None,
    extra_flags=None,
    build_directory=None,
    delayed=True,
    verbose=False,
):
    _, _, to_include, parse_flags = parse_module(
        root_directory=root_directory,
        source=source,
        target=target,
        system_includes=system_includes,
        includes=includes,
        extra_flags=extra_flags,
        build_directory=build_directory,
        delayed=delayed,
        parse_only=True,
        verbose=verbose,
    )

    logging.info('Generating for %s ...', os.path.basename(source))

    # analyze include directories from parse flags
    i, include_in_flags = 0, []
    while i < len(parse_flags):
        if parse_flags[i].startswith('-I'):
            if parse_flags[i][2:]:
                include_in_flags.append(parse_flags[i][2:])
            else:
                include_in_flags.append(parse_flags[i + 1])
                i += 1
        if parse_flags[i] == '-isystem':
            include_in_flags.append(parse_flags[i + 1])
            i += 1
        i += 1

    for inc in to_include:
        header = resolve_include(inc, [], include_in_flags)
        if header is not None:
            print('Depends:%s' % header.strip())
