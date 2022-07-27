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
from collections import Counter
from typing import List
from typing import Optional
from typing import Tuple

DEP_MISSING_ERROR = '''
    Dependencies {dep} cannot be found, please try again after:

        pip3 install {dep}

'''

try:
    from clang import cindex
    from clang.cindex import Cursor
    from clang.cindex import CursorKind
    from clang.cindex import Type
    from clang.cindex import TypeKind
except ImportError:
    raise RuntimeError(  # pylint: disable=raise-missing-from
        DEP_MISSING_ERROR.format(dep='libclang')
    )

###############################################################################
#
# parse codegen spec:
#
#   __attribute__((annotate("vineyard"))): vineyard classes
#   __attribute__((annotate("shared"))): shared member/method
#   __attribute__((annotate("streamable"))): shared member/method
#   __attribute__((annotate("distributed"))): shared member/method
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


def figure_out_namespace(node: Cursor) -> Optional[str]:
    while True:
        parent = node.semantic_parent
        if parent is None:
            return None
        if parent.kind == CursorKind.NAMESPACE:
            parent_ns = figure_out_namespace(parent)
            if parent_ns is None:
                return parent.spelling
            else:
                return '%s::%s' % (parent_ns, parent.spelling)
        node = parent


def unpack_pointer_type(node_type: Type) -> Tuple[Type, str, str]:
    if node_type.kind == TypeKind.POINTER:
        node_type = node_type.get_pointee()
        star = '*'
    else:
        basename = node_type.spelling.split('<')[0]
        namespace = figure_out_namespace(node_type.get_declaration())

        if (  # pylint: disable=too-many-boolean-expressions
            basename == 'std::shared_ptr'
            or namespace in ['std', 'std::__1']
            and basename == 'shared_ptr'
            or basename == 'std::unique_ptr'
            or namespace in ['std', 'std::__1']
            and basename == 'unique_ptr'
        ):
            star = '*'
            node_type = node_type.get_template_argument_type(0)
            namespace = figure_out_namespace(node_type.get_declaration())
        else:
            star = ''

    node_typename = node_type.spelling
    if namespace is not None and node_typename.startswith(namespace):
        node_typename = node_typename[len(namespace) + 2 :]

    return node_type, star, node_typename


def is_template_parameter(node: Cursor, typename: str) -> bool:
    parent = node.semantic_parent
    if parent is None:
        return False
    if parent.kind == CursorKind.CLASS_TEMPLATE:
        for ch in parent.get_children():
            if ch.kind == CursorKind.TEMPLATE_TYPE_PARAMETER:
                if typename == ch.spelling:
                    return True
    return False


def is_primitive_types(
    node: Cursor, node_type: "cindex.Type", typename: str, star: str
) -> bool:
    if star:
        return False
    if node_type.is_pod():
        return True
    if is_template_parameter(node, typename):
        # treat template parameter as meta, see `scalar.vineyard-mod`.
        return True
    return typename in [
        'std::string',
        'String',
        'vineyard::String',
        'json',
        'vineyard::json',
    ]


def is_list_type(namespace: str, basename: str) -> bool:
    return (
        basename in ['vineyard::Tuple', 'vineyard::List']
        or namespace == 'vineyard'
        and basename in ['Tuple', 'List']
    )


def is_dict_type(namespace: str, basename: str) -> bool:
    return (
        basename in ['vineyard::Map', 'vineyard::UnorderedMap']
        or namespace == 'vineyard'
        and basename in ['Map', 'UnorderedMap']
    )


def parse_codegen_spec_from_type(node: Cursor):
    node_type, star, typename = unpack_pointer_type(node.type)
    if star:
        _, star_inside, _ = unpack_pointer_type(node_type)
        if star_inside:
            raise ValueError(
                'Pointer of pointer %s is not supported' % node.type.spelling
            )

    basename = typename.split('<')[0]
    namespace = figure_out_namespace(node_type.get_declaration())

    if not star:
        if is_list_type(namespace, basename):
            element_type = node_type.get_template_argument_type(0)

            nested_base_name = element_type.spelling.split('<')[0]
            nested_namespace = figure_out_namespace(element_type.get_declaration())
            if is_list_type(nested_namespace, nested_base_name):
                element_type = element_type.get_template_argument_type(0)
                element_type, inside_star, element_typename = unpack_pointer_type(
                    element_type
                )
                typekind = 'dlist'
            else:
                element_type, inside_star, element_typename = unpack_pointer_type(
                    element_type
                )
                if is_primitive_types(
                    node, element_type, element_typename, inside_star
                ):
                    if inside_star:
                        raise ValueError(
                            'pointer of primitive types inside Tuple/List is not '
                            'supported: %s' % node.type.spelling
                        )
                    return CodeGenKind('meta')
                else:
                    typekind = 'list'
            return CodeGenKind(typekind, (element_typename, inside_star))

        if is_dict_type(namespace, basename):
            key_type = node_type.get_template_argument_type(0)
            key_typename = key_type.spelling
            value_type = node_type.get_template_argument_type(1)
            value_type, inside_star, value_typename = unpack_pointer_type(value_type)
            if is_primitive_types(node, value_type, value_typename, inside_star):
                if inside_star:
                    raise ValueError(
                        'pointer of primitive types inside Map is not supported: %s'
                        % node.type.spelling
                    )
                return CodeGenKind('meta')
            else:
                return CodeGenKind(
                    'dict', ((key_typename,), (value_typename, inside_star))
                )

    if is_primitive_types(node, node_type, typename, star):
        return CodeGenKind('meta')
    else:
        # directly return: generate data members, in pointer format
        return CodeGenKind('plain', (basename, star))


###############################################################################
#
# dump the AST for debugging
#


def is_std_ns(node: Cursor) -> bool:
    if node.kind == CursorKind.NAMESPACE:
        if node.spelling == 'std':
            return True
        if node.spelling == '__1':
            parent: Cursor = node.semantic_parent
            if (
                parent is not None
                and parent.kind == CursorKind.NAMESPACE
                and parent.spelling == 'std'
            ):
                return True
    return False


def is_reference_node(node):
    return node.kind in [
        CursorKind.TYPE_REF,
        CursorKind.TEMPLATE_REF,
        CursorKind.MEMBER_REF,
        CursorKind.OVERLOADED_DECL_REF,
        CursorKind.VARIABLE_REF,
    ]


def dump_ast(
    node, indent=0, saw=None, base_indent=4, include_refs=False, include_ref_depth=1
):
    if saw is None:
        saw = Counter()

    k: "CursorKind" = node.kind
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

    saw[str(node.hash)] += 1

    # FIXME: skip auto generated decls
    skip = len([c for c in node.get_children() if indent == 0 and is_std_ns(c)])
    for c in node.get_children():
        if indent == 0 and is_std_ns(c):
            skip -= 1
        if skip == 0:
            dump_ast(
                c,
                indent + base_indent,
                saw,
                base_indent,
                include_refs,
                include_ref_depth - 1,
            )

    if include_refs and include_ref_depth > 0 and is_reference_node(node):
        ch = node.get_definition()
        if ch is not None:
            dump_ast(
                ch,
                indent + base_indent,
                saw,
                base_indent,
                include_refs,
                include_ref_depth - 1,
            )

    saw[str(node.hash)] -= 1


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
            for attr_kind in [
                'vineyard',
                'vineyard(streamable)',
                'shared',
                'distributed',
            ]:
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
            attribute = check_serialize_attribute(node)
            if attribute in ['vineyard', 'vineyard(streamable)']:
                to_reflect.append((attribute, namespaces, node))

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
            attribute = check_serialize_attribute(child)
            if attribute in ['shared', 'distributed']:
                fields.append(child)
            continue

        if child.kind == CursorKind.CXX_METHOD:
            attribute = check_serialize_attribute(child)
            if attribute == 'distributed':
                raise ValueError(
                    'The annotation "[[distributed]]" is not allowed on methods'
                )
            if attribute == 'shared':
                fields.append(child)
            if not has_post_construct and child.spelling == 'PostConstruct':
                for body in child.get_children():
                    if body.kind == CursorKind.CXX_OVERRIDE_ATTR:
                        has_post_construct = True
                        break
            continue

        if child.kind == CursorKind.TYPE_ALIAS_DECL:
            using_alias.append((child.spelling, child.extent))
            continue

    return fields, using_alias, first_mmeber_offset, has_post_construct


def find_distributed_field(definitions: List["CursorKind"]) -> "CursorKind":
    fields = []
    for child in definitions:
        if child.kind == CursorKind.FIELD_DECL:
            attribute = check_serialize_attribute(child)
            if attribute in ['distributed']:
                fields.append(child)
    if len(fields) == 0:
        return None
    if len(fields) == 1:
        return fields[0]
    raise ValueError(
        'A distributed object can only have at most one distributed member '
        '(annotated with "[[distributed]]"'
    )


def split_members_and_methods(fields):
    members, methods = [], []
    for field in fields:
        if field.kind == CursorKind.FIELD_DECL:
            members.append(field)
        elif field.kind == CursorKind.CXX_METHOD:
            methods.append(field)
        else:
            raise ValueError('Unknown field kind: %s' % field)
    return members, methods


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
    if build_directory is None:
        return None
    # check if the file exists first to suppress the clang warning.
    compile_commands_json = os.path.join(build_directory, 'compile_commands.json')
    if not os.path.isfile(compile_commands_json) or not os.access(
        compile_commands_json, os.R_OK
    ):
        return None
    try:
        return cindex.CompilationDatabase.fromDirectory(build_directory)
    except cindex.CompilationDatabaseError:
        return None


def validate_and_strip_input_file(source):
    if not os.path.isfile(source) or not os.access(source, os.R_OK):
        return None, 'File not exists'
    with open(source, 'r', encoding='utf-8') as fp:
        content = fp.read().splitlines(keepends=False)
    # pass(TODO): valid and remove the first line
    content = '\n'.join(content)

    # pass: rewrite `[[...]]` with `__attribute__((annotate(...)))`
    attributes = ['vineyard', 'vineyard(streamable)', 'shared', 'distributed']
    for attr in attributes:
        content = content.replace(
            '[[%s]]' % attr, '__attribute__((annotate("%s")))' % attr
        )

    return content, ''


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


def generate_parsing_flags(
    source,
    system_includes=None,
    includes=None,
    extra_flags=None,
    build_directory=None,
    delayed=True,
):
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
        '-D__VPP=1',
    ]
    warning_flags = [
        '-Wno-unused-function',
        '-Wno-unused-parameter',
        '-Wno-unused-private-field',
        '-Wno-unknown-warning-option',
    ]

    # prepare flags
    flags = None
    compliation_db = parse_compilation_database(build_directory)
    if compliation_db is not None:
        commands = compliation_db.getCompileCommands(source)
        if commands is not None and len(commands) > 0:
            # strip flags
            flags = strip_flags(list(commands[0].arguments)[1:-1])

            # adapts to libclang v14.0.1
            if flags and flags[-1] == '--':
                flags.pop(-1)

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

    return base_flags + flags + warning_flags


def parse_module(  # noqa: C901
    root_directory,  # pylint: disable=unused-argument
    source,
    target=None,  # pylint: disable=unused-argument
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

    # parse
    index = cindex.Index.create()
    options = (
        ParseOption.Default
        | ParseOption.DetailedPreprocessingRecord
        | ParseOption.SkipFunctionBodies
        | ParseOption.IncludeAttributedTypes
        | ParseOption.KeepGoing
    )

    parse_flags = generate_parsing_flags(
        source,
        system_includes=system_includes,
        includes=includes,
        extra_flags=extra_flags,
        build_directory=build_directory,
        delayed=delayed,
    )

    if parse_only:
        options |= ParseOption.SingleFileParse
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
            dump_ast(module)
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
