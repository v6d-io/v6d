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
import textwrap
from argparse import ArgumentParser

from distutils.util import strtobool

DEP_MISSING_ERROR = '''
    Dependencies {dep} cannot be found, please try again after:

        pip3 install {dep}

'''

try:
    import clang.cindex
    from clang.cindex import CursorKind, TranslationUnit
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
#   __attribute__((annotate("codegen"))): meta codegen
#   __attribute__((annotate("codegen:Type"))): member type: Type member_
#   __attribute__((annotate("codegen:Type*"))): member type: std::shared_ptr<Type> member_
#   __attribute__((annotate("codegen:[Type*]"))): list member type: std::vector<Type> member_
#   __attribute__((annotate("codegen:{Type}"))): set member type: std::set<Type> member_
#   __attribute__((annotate("codegen:{Type*}"))): set member type: std::set<std::shared_ptr<Type>> member_
#   __attribute__((annotate("codegen:{int32_t: Type}"))): dict member type: std::map<int32_t, Type> member_
#   __attribute__((annotate("codegen:{int32_t: Type*}"))): dict member type: std::map<int32_t, std::shared_ptr<Type>> member_
#
# FIXME(hetao): parse the codegen spec directly from the type signature of the member variable
#


class CodeGenKind(object):
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


name_pattern = parsec.spaces() >> parsec.regex(
    r'[_a-zA-Z][_a-zA-Z0-9<>, ]*(::[_a-zA-Z][_a-zA-Z0-9<>, ]*)*') << parsec.spaces()

star_pattern = parsec.spaces() >> parsec.optional(parsec.string('*'), '') << parsec.spaces()

parse_meta = parsec.spaces().parsecmap(lambda _: CodeGenKind('meta'))

parse_plain = (parsec.spaces() >>
               (name_pattern + star_pattern) << parsec.spaces()).parsecmap(lambda value: CodeGenKind('plain', value))
parse_list = (parsec.string('[') >>
              (name_pattern + star_pattern) << parsec.string(']')).parsecmap(lambda value: CodeGenKind('list', value))
parse_dlist = (
    parsec.string('[[') >>
    (name_pattern + star_pattern) << parsec.string(']]')).parsecmap(lambda value: CodeGenKind('dlist', value))
parse_set = (parsec.string('{') >>
             (name_pattern + star_pattern) << parsec.string('}')).parsecmap(lambda value: CodeGenKind('set', value))
parse_dict = (parsec.string('{') >> parsec.separated((name_pattern + star_pattern), parsec.string(':'), 2, 2) <<
              parsec.string('}')).parsecmap(lambda values: CodeGenKind('dict', tuple(values)))

codegen_spec_parser = parse_dict ^ parse_set ^ parse_dlist ^ parse_list ^ parse_plain ^ parse_meta


def parse_codegen_spec(kind):
    if kind.startswith('vineyard'):
        kind = kind[len('vineyard'):]
    if kind.startswith('codegen'):
        kind = kind[len('codegen'):]
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

        print(tpl.format(indent=' ' * indent, kind=k.name, name=name, type_name=type_name))

    saw.add(node.hash)
    if include_refs:
        if node.referenced is not None and node.referenced.hash not in saw:
            dump_ast(node.referenced, indent + base_indent, saw, base_indent, include_refs)

    # FIXME: skip auto generated decls
    skip = len([c for c in node.get_children() if indent == 0 and is_std_ns(c)])
    for c in node.get_children():
        if not skip:
            dump_ast(c, indent + base_indent, saw, base_indent, include_refs)
        if indent == 0 and is_std_ns(c):
            skip -= 1
    saw.remove(node.hash)


class ParseOption(object):
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
                CursorKind.CXX_BASE_SPECIFIER, CursorKind.CXX_ACCESS_SPEC_DECL, CursorKind.CXX_METHOD,
                CursorKind.FIELD_DECL
        ]:
            return True
    return False


def filter_the_module(root, filepath):
    children = []
    for child in root.get_children():
        if child.location and child.location.file and \
                child.location.file.name == filepath:
            children.append(child)
    return children


def traverse(node, to_reflect, to_include, namespaces=None):
    ''' Traverse the AST tree.
    '''
    if node.kind in [CursorKind.CLASS_DECL, CursorKind.CLASS_TEMPLATE, CursorKind.STRUCT_DECL]:
        # codegen for all top-level classes (definitions, not declarations) in the given file.
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
                    CursorKind.TEMPLATE_TYPE_PARAMETER, CursorKind.CXX_BASE_SPECIFIER, CursorKind.ANNOTATE_ATTR
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

        if not has_post_construct and \
                child.kind == CursorKind.CXX_METHOD and child.spelling == 'PostConstruct':
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


###############################################################################
#
# codegen the `Create` static method
#

create_tpl = '''
{class_header}
std::unique_ptr<Object> {class_name_elaborated}::Create() {{
    return std::static_pointer_cast<Object>(
        std::unique_ptr<{class_name_elaborated}>{{
            new {class_name_elaborated}()}});
}}
'''

create_meth_tpl = '''
  public:
    static std::unique_ptr<Object> Create() __attribute__((used)) {{
        return std::static_pointer_cast<Object>(
            std::unique_ptr<{class_name_elaborated}>{{
                new {class_name_elaborated}()}});
    }}
'''


def codegen_create(class_header, class_name, class_name_elaborated, meth=False):
    if meth:
        function_tpl = create_meth_tpl
    else:
        function_tpl = create_tpl

    return function_tpl.format(class_header=class_header,
                               class_name=class_name,
                               class_name_elaborated=class_name_elaborated)


###############################################################################
#
# codegen the `Construct` method
#

construct_tpl = '''
{class_header}
void {class_name_elaborated}::Construct(const ObjectMeta &meta) {{
    std::string __type_name = type_name<{class_name_elaborated}>();
    VINEYARD_ASSERT(
        meta.GetTypeName() == __type_name,
        "Expect typename '" + __type_name + "', but got '" + meta.GetTypeName() + "'");
    this->meta_ = meta;
    this->id_ = meta.GetId();

    {construct_body}

    {post_construct}
}}
'''

post_construct_tpl = '''
    if (meta.IsLocal()) {
        this->PostConstruct(meta);
    }'''

construct_meth_tpl = '''
  public:
    void Construct(const ObjectMeta& meta) override {{
        std::string __type_name = type_name<{class_name_elaborated}>();
        VINEYARD_ASSERT(
            meta.GetTypeName() == __type_name,
            "Expect typename '" + __type_name + "', but got '" + meta.GetTypeName() + "'");
        this->meta_ = meta;
        this->id_ = meta.GetId();

        {construct_body}

        {post_construct}
    }}
'''

post_construct_meth_tpl = '''
        if (meta.IsLocal()) {
            this->PostConstruct(meta);
        }'''

construct_meta_tpl = '''
    meta.GetKeyValue("{name}", this->{name});'''

construct_plain_tpl = '''
    this->{name}.Construct(meta.GetMemberMeta("{name}"));'''

construct_plain_star_tpl = '''
    this->{name} = {deref}std::dynamic_pointer_cast<{element_type}>(meta.GetMember("{name}"));'''

construct_list_tpl = '''
    this->{name}.resize(meta.GetKeyValue<size_t>("__{name}-size"));
    for (size_t __idx = 0; __idx < this->{name}.size(); ++__idx) {{
        this->{name}[__idx].Construct(
            meta.GetMemberMeta("__{name}-" + std::to_string(__idx)));
    }}'''

construct_list_star_tpl = '''
    for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("__{name}-size"); ++__idx) {{
        this->{name}.emplace_back({deref}std::dynamic_pointer_cast<{element_type}>(
                meta.GetMember("__{name}-" + std::to_string(__idx))));
    }}'''

construct_dlist_tpl = '''
    this->{name}.resize(meta.GetKeyValue<size_t>("__{name}-size"));
    for (size_t __idx = 0; __idx < this->{name}.size(); ++__idx) {{
        this->{name}[__idx].resize(meta.GetKeyValue<size_t>(
            "__{name}-" + std::to_string(__idx) + "-size"));
        for (size_t __idy = 0; __idy < this->{name}[__idx].size(); ++__idy) {{
            this->{name}[__idx][__idy].Construct(
                meta.GetMemberMeta("__{name}-" + std::to_string(__idx) + "-" + std::to_string(__idy)));
        }}
    }}'''

construct_dlist_star_tpl = '''
    this->{name}.resize(meta.GetKeyValue<size_t>("__{name}-size"));
    for (size_t __idx = 0; __idx < this->{name}.size(); ++__idx) {{
        for (size_t __idy = 0; __idy < meta.GetKeyValue<size_t>(
                "__{name}-" + std::to_string(__idx) + "-size"); ++__idy) {{
            this->{name}[__idx].emplace_back({deref}std::dynamic_pointer_cast<{element_type}>(
                meta.GetMember("__{name}-" + std::to_string(__idx) + "-" + std::to_string(__idy))));
        }}
    }}'''

construct_set_tpl = '''
    for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("__{name}-size"); ++__idx) {{
        this->{name}.emplace({deref}std::dynamic_pointer_cast<{element_type}>(
                meta.GetMember("__{name}-" + std::to_string(__idx))));
    }}'''

construct_dict_tpl = '''
    for (size_t __idx = 0; __idx < meta.GetKeyValue<size_t>("__{name}-size"); ++__idx) {{
        this->{name}.emplace(meta.GetKeyValue<{key_type}>("__{name}-key-" + std::to_string(__idx)),
                {deref}std::dynamic_pointer_cast<{value_type}>(
                        meta.GetMember("__{name}-value-" + std::to_string(__idx))));
    }}'''


def codegen_construct(class_header, class_name, class_name_elaborated, fields, has_post_ctor, meth=False):
    body = []
    for kind, field in fields:
        name = field.spelling
        spec = parse_codegen_spec(kind)
        if spec.is_meta:
            tpl = construct_meta_tpl
        if spec.is_plain:
            if spec.star:
                tpl = construct_plain_star_tpl
            else:
                tpl = construct_plain_tpl
        if spec.is_list:
            if spec.star:
                tpl = construct_list_star_tpl
            else:
                tpl = construct_list_tpl
        if spec.is_dlist:
            if spec.star:
                tpl = construct_dlist_star_tpl
            else:
                tpl = construct_dlist_tpl
        if spec.is_set:
            tpl = construct_set_tpl
        if spec.is_dict:
            tpl = construct_dict_tpl

        if spec.is_dict:
            key_type = spec.element_type[0]
            value_type = spec.element_type[1]
        else:
            key_type = None
            value_type = None

        body.append(
            tpl.format(name=name,
                       element_type=spec.element_type,
                       key_type=key_type,
                       value_type=value_type,
                       deref=spec.deref))

    if meth:
        function_tpl = construct_meth_tpl
        function_body_indent = 4
        if has_post_ctor:
            post_ctor = post_construct_meth_tpl
        else:
            post_ctor = ''
    else:
        function_tpl = construct_tpl
        function_body_indent = 0
        if has_post_ctor:
            post_ctor = post_construct_tpl
        else:
            post_ctor = ''

    code = function_tpl.format(class_header=class_header,
                               class_name=class_name,
                               class_name_elaborated=class_name_elaborated,
                               construct_body=textwrap.indent(''.join(body), ' ' * function_body_indent).strip(),
                               post_construct=post_ctor)
    return code


###############################################################################
#
# codegen the base builder
#

base_builder_tpl = '''
{class_header}
class {class_name}BaseBuilder: public ObjectBuilder {{
  public:
    {using_alias}

    explicit {class_name}BaseBuilder(Client &client) {{}}

    explicit {class_name}BaseBuilder(
            {class_name_elaborated} const &__value) {{
        {get_and_assign}
    }}

    explicit {class_name}BaseBuilder(
            std::shared_ptr<{class_name_elaborated}> const & __value):
        {class_name}BaseBuilder(*__value) {{
    }}

    std::shared_ptr<Object> _Seal(Client &client) override {{
        // ensure the builder hasn't been sealed yet.
        ENSURE_NOT_SEALED(this);

        VINEYARD_CHECK_OK(this->Build(client));
        auto __value = std::make_shared<{class_name_elaborated}>();

        size_t __value_nbytes = 0;

        __value->meta_.SetTypeName(type_name<{class_name_elaborated}>());
        if (std::is_base_of<GlobalObject, {class_name_elaborated}>::value) {{
            __value->meta_.SetGlobal(true);
        }}

        {assignments}

        __value->meta_.SetNBytes(__value_nbytes);

        VINEYARD_CHECK_OK(client.CreateMetaData(__value->meta_, __value->id_));

        // mark the builder as sealed
        this->set_sealed(true);

        {post_construct}
        return std::static_pointer_cast<Object>(__value);
    }}

    Status Build(Client &client) override {{
        return Status::OK();
    }}

  protected:
    {fields_declares}

    {setters}
}};
'''

field_declare_tpl = '''
    {field_type_elaborated} {field_name};'''


def codegen_field_declare(field_name, field_type, spec):
    if spec.is_meta:
        field_type_elaborated = field_type
    if spec.is_plain:
        field_type_elaborated = 'std::shared_ptr<ObjectBase>'
    if spec.is_list:
        field_type_elaborated = 'std::vector<std::shared_ptr<ObjectBase>>'
    if spec.is_dlist:
        field_type_elaborated = 'std::vector<std::vector<std::shared_ptr<ObjectBase>>>'
    if spec.is_set:
        field_type_elaborated = 'std::set<std::shared_ptr<ObjectBase>>'
    if spec.is_dict:
        field_type_elaborated = 'std::map<{key_type}, std::shared_ptr<ObjectBase>>'.format(
            key_type='typename %s::key_type' % field_type)
    return field_declare_tpl.format(field_name=field_name, field_type_elaborated=field_type_elaborated)


field_assign_meta_tpl = '''
        __value->{field_name} = {field_name};
        __value->meta_.AddKeyValue("{field_name}", __value->{field_name});
'''

field_assign_plain_tpl = '''
        // using __{field_name}_value_type = typename {field_type}{element_type};
        using __{field_name}_value_type = {element_type_name}decltype(__value->{field_name}){element_type};
        auto __value_{field_name} = std::dynamic_pointer_cast<__{field_name}_value_type>(
            {field_name}->_Seal(client));
        __value->{field_name} = {deref}__value_{field_name};
        __value->meta_.AddMember("{field_name}", __value->{field_name});
        __value_nbytes += __value_{field_name}->nbytes();
'''

field_assign_list_tpl = '''
        // using __{field_name}_value_type = typename {field_type}::value_type{element_type};
        using __{field_name}_value_type = typename decltype(__value->{field_name})::value_type{element_type};

        size_t __{field_name}_idx = 0;
        for (auto &__{field_name}_value: {field_name}) {{
            auto __value_{field_name} = std::dynamic_pointer_cast<__{field_name}_value_type>(
                __{field_name}_value->_Seal(client));
            __value->{field_name}.emplace_back({deref}__value_{field_name});
            __value->meta_.AddMember("__{field_name}-" + std::to_string(__{field_name}_idx),
                                     __value_{field_name});
            __value_nbytes += __value_{field_name}->nbytes();
            __{field_name}_idx += 1;
        }}
        __value->meta_.AddKeyValue("__{field_name}-size", __value->{field_name}.size());
'''

field_assign_dlist_tpl = '''
        // using __{field_name}_value_type = typename {field_type}::value_type::value_type{element_type};
        using __{field_name}_value_type = typename decltype(__value->{field_name})::value_type::value_type{element_type};

        size_t __{field_name}_idx = 0;
        __value->{field_name}.resize({field_name}.size());
        for (auto &__{field_name}_value_vec: {field_name}) {{
            size_t __{field_name}_idy = 0;
            for (auto &__{field_name}_value: __{field_name}_value_vec) {{
                auto __value_{field_name} = std::dynamic_pointer_cast<__{field_name}_value_type>(
                    __{field_name}_value->_Seal(client));
                __value->{field_name}[__{field_name}_idx].emplace_back({deref}__value_{field_name});
                __value->meta_.AddMember("__{field_name}-" + std::to_string(__{field_name}_idx) + "-" + std::to_string(__{field_name}_idy),
                                         __value_{field_name});
                __value_nbytes += __value_{field_name}->nbytes();
                __{field_name}_idy += 1;
            }}
            __{field_name}_idx += 1;
        }}
        __value->meta_.AddKeyValue("__{field_name}-size", __value->{field_name}.size());
'''

field_assign_set_tpl = '''
        // using __{field_name}_value_type = typename {field_type}::value_type{element_type};
        using __{field_name}_value_type = typename decltype(__value->{field_name})::value_type{element_type};

        size_t __{field_name}_idx = 0;
        for (auto &__{field_name}_value: {field_name}) {{
            auto __value_{field_name} = std::dynamic_pointer_cast<__{field_name}_value_type>(
                __{field_name}_value->_Seal(client));
            __value->{field_name}.emplace({deref}__value_{field_name});
            __value->meta_.AddMember("__{field_name}-" + std::to_string(__{field_name}_idx),
                                      __value_{field_name});
            __value_nbytes += __value_{field_name}->nbytes();
            __{field_name}_idx += 1;
        }}
        __value->meta_.AddKeyValue("__{field_name}-size", __value->{field_name}.size());
'''

field_assign_dict_tpl = '''
        // using __{field_name}_value_type = typename {field_type}::mapped_type{element_type};
        using __{field_name}_value_type = typename decltype(__value->{field_name})::mapped_type{element_type};

        size_t __{field_name}_idx = 0;
        for (auto &__{field_name}_kv: {field_name}) {{
            auto __value_{field_name} = std::dynamic_pointer_cast<__{field_name}_value_type>(
                __{field_name}_kv.second->_Seal(client));
            __value->{field_name}.emplace(__{field_name}_kv.first, {deref}__value_{field_name});
            __value->meta_.AddKeyValue("__{field_name}-key-" + std::to_string(__{field_name}_idx),
                                        __{field_name}_kv.first);
            __value->meta_.AddMember("__{field_name}-value-" + std::to_string(__{field_name}_idx),
                                     __value_{field_name});
            __value_nbytes += __value_{field_name}->nbytes();
            __{field_name}_idx += 1;
        }}
        __value->meta_.AddKeyValue("__{field_name}-size", __value->{field_name}.size());
'''


def codegen_field_assign(field_name, field_type, spec):
    if spec.is_meta:
        tpl = field_assign_meta_tpl
    if spec.is_plain:
        tpl = field_assign_plain_tpl
    if spec.is_list:
        tpl = field_assign_list_tpl
    if spec.is_dlist:
        tpl = field_assign_dlist_tpl
    if spec.is_set:
        tpl = field_assign_set_tpl
    if spec.is_dict:
        tpl = field_assign_dict_tpl
    if spec.deref:
        element_type = ''
        element_type_name = ''
    else:
        element_type = '::element_type'
        element_type_name = 'typename '
    return tpl.format(field_name=field_name,
                      field_type=field_type,
                      deref=spec.deref,
                      element_type=element_type,
                      element_type_name=element_type_name)


field_setter_meta_tpl = '''
    void set_{field_name}({field_type} const &{field_name}_) {{
        this->{field_name} = {field_name}_;
    }}
'''

field_setter_plain_tpl = '''
    void set_{field_name}(std::shared_ptr<ObjectBase> const & {field_name}_) {{
        this->{field_name} = {field_name}_;
    }}
'''

field_setter_list_tpl = '''
    void set_{field_name}(std::vector<std::shared_ptr<ObjectBase>> const &{field_name}_) {{
        this->{field_name} = {field_name}_;
    }}
    void set_{field_name}(size_t const idx, std::shared_ptr<ObjectBase> const &{field_name}_) {{
        if (idx >= this->{field_name}.size()) {{
            this->{field_name}.resize(idx + 1);
        }}
        this->{field_name}[idx] = {field_name}_;
    }}
    void add_{field_name}(std::shared_ptr<ObjectBase> const &{field_name}_) {{
        this->{field_name}.emplace_back({field_name}_);
    }}
'''

field_setter_dlist_tpl = '''
    void set_{field_name}(std::vector<std::vector<std::shared_ptr<ObjectBase>>> const &{field_name}_) {{
        this->{field_name} = {field_name}_;
    }}
    void set_{field_name}(size_t const idx, std::vector<std::shared_ptr<ObjectBase>> const &{field_name}_) {{
        if (idx >= this->{field_name}.size()) {{
            this->{field_name}.resize(idx + 1);
        }}
        this->{field_name}[idx] = {field_name}_;
    }}
    void set_{field_name}(size_t const idx, size_t const idy,
                          std::shared_ptr<ObjectBase> const &{field_name}_) {{
        if (idx >= this->{field_name}.size()) {{
            this->{field_name}.resize(idx + 1);
        }}
        if (idy >= this->{field_name}[idx].size()) {{
            this->{field_name}[idx].resize(idy + 1);
        }}
        this->{field_name}[idx][idy] = {field_name}_;
    }}
    void add_{field_name}(std::vector<std::shared_ptr<ObjectBase>> const &{field_name}_) {{
        this->{field_name}.emplace_back({field_name}_);
    }}
'''

field_setter_set_tpl = '''
    void set_{field_name}(std::set<std::shared_ptr<ObjectBase>> const &{field_name}_) {{
        this->{field_name} = {field_name}_;
    }}
    void add_{field_name}(std::shared_ptr<ObjectBase> const &{field_name}_) {{
        this->{field_name}.emplace({field_name}_);
    }}
'''

field_setter_dict_tpl = '''
    void set_{field_name}(std::map<{field_key_type}, std::shared_ptr<ObjectBase>> const &{field_name}_) {{
        this->{field_name} = {field_name}_;
    }}
    void set_{field_name}({field_key_type} const &{field_name}_key_,
                           std::shared_ptr<ObjectBase> {field_name}_value_) {{
        this->{field_name}.emplace({field_name}_key_, {field_name}_value_);
    }}
'''


def codegen_field_setter(field_name, field_type, spec):
    if spec.is_meta:
        tpl = field_setter_meta_tpl
    if spec.is_plain:
        tpl = field_setter_plain_tpl
    if spec.is_list:
        tpl = field_setter_list_tpl
    if spec.is_dlist:
        tpl = field_setter_dlist_tpl
    if spec.is_set:
        tpl = field_setter_set_tpl
    if spec.is_dict:
        tpl = field_setter_dict_tpl
    if spec.is_dict:
        field_key_type = 'typename %s::key_type' % field_type
        field_value_type = 'typename %s::mapped_type' % field_type
    else:
        field_key_type = None
        field_value_type = None
    return tpl.format(field_name=field_name,
                      field_type=field_type,
                      field_key_type=field_key_type,
                      field_value_type=field_value_type)


get_assign_meta_tpl = '''
        this->set_{field_name}(__value.{field_name});'''

get_assign_plain_tpl = '''
        this->set_{field_name}(
            std::make_shared<typename std::decay<decltype(__value.{field_name})>::type>(
                __value.{field_name}));'''

get_assign_plain_star_tpl = '''
        this->set_{field_name}(__value.{field_name});'''

get_assign_sequence_tpl = '''
        for (auto const &__{field_name}_item: __value.{field_name}) {{
            this->add_{field_name}(
                std::make_shared<typename std::decay<decltype(__{field_name}_item)>::type>(
                    __{field_name}_item));
        }}'''

get_assign_sequence_star_tpl = '''
        for (auto const &__{field_name}_item: __value.{field_name}) {{
            this->add_{field_name}(__{field_name}_item);
        }}'''

get_assign_dlist_tpl = '''
        this->{field_name}.resize(__value.{field_name}.size());
        for (size_t __idx = 0; __idx < __value.{field_name}.size(); ++__idx) {{
            this->{field_name}[__idx].resize(__value.{field_name}[__idx].size());
            for (size_t __idy = 0; __idy < __value.{field_name}[__idx].size(); ++__idy) {{
                auto const &__{field_name}_item = __value.{field_name}[__idx][__idy];
                this->{field_name}[__idx][__idy] =
                    std::make_shared<typename std::decay<decltype(__{field_name}_item)>::type>(
                        __{field_name}_item));
            }}
        }}'''

get_assign_dlist_star_tpl = '''
        this->{field_name}.resize(__value.{field_name}.size());
        for (size_t __idx = 0; __idx < __value.{field_name}.size(); ++__idx) {{
            this->{field_name}[__idx].resize(__value.{field_name}[__idx].size());
            for (size_t __idy = 0; __idy < __value.{field_name}[__idx].size(); ++__idy) {{
                this->{field_name}[__idx][__idy] = __value.{field_name}[__idx][__idy];
            }}
        }}'''

get_assign_dict_tpl = '''
        for (auto const &__{field_name}_item_kv: __value.{field_name}) {{
            this->set_{field_name}(__{field_name}_item_kv.first,
                                   std::make_shared<typename std::decay<
                                        decltype(__{field_name}_item_kv.second)>::type>(
                                        __{field_name}_item_kv.second));
        }}'''

get_assign_dict_star_tpl = '''
        for (auto const &__{field_name}_item_kv: __value.{field_name}) {{
            this->set_{field_name}(__{field_name}_item_kv.first,
                                   __{field_name}_item_kv.second);
        }}'''


def codegen_field_get_assign(field_name, spec):
    if spec.is_meta:
        tpl = get_assign_meta_tpl
    if spec.is_plain:
        if spec.star:
            tpl = get_assign_plain_star_tpl
        else:
            tpl = get_assign_plain_tpl
    if spec.is_list or spec.is_set:
        if spec.star:
            tpl = get_assign_sequence_star_tpl
        else:
            tpl = get_assign_sequence_tpl
    if spec.is_dlist:
        if spec.star:
            tpl = get_assign_dlist_star_tpl
        else:
            tpl = get_assign_dlist_tpl
    if spec.is_dict:
        if spec.star:
            tpl = get_assign_dict_star_tpl
        else:
            tpl = get_assign_dict_tpl
    return tpl.format(field_name=field_name)


using_alia_tpl = '''
    // using {alia}
    {extent};'''


def codegen_using_alia(alia, extent):
    return using_alia_tpl.format(alia=alia, extent=extent)


post_construct_in_seal_tpl = '''
        // run `PostConstruct` to return a valid object
        __value->PostConstruct(__value->meta_);
'''


def codegen_base_builder(class_header, class_name, class_name_elaborated, fields, using_alias_values, has_post_ctor):
    declarations = []
    assignments = []
    get_and_assigns = []
    setters = []
    using_alias_statements = []

    # genreate using alias
    for alia, extent in using_alias_values:
        using_alias_statements.append(codegen_using_alia(alia, extent))

    # core field assignment
    for kind, field in fields:
        name = field.spelling
        spec = parse_codegen_spec(kind)
        field_type = field.type.spelling

        # generate field declarations
        declarations.append(codegen_field_declare(name, field_type, spec))

        # generate field assignment
        assignments.append(codegen_field_assign(name, field_type, spec))

        # generate get-and-assign statements
        get_and_assigns.append(codegen_field_get_assign(name, spec))

        # generate field setter method
        setters.append(codegen_field_setter(name, field_type, spec))

    if has_post_ctor:
        post_ctor = post_construct_in_seal_tpl
    else:
        post_ctor = ''

    code = base_builder_tpl.format(class_header=class_header,
                                   class_name=class_name,
                                   class_name_elaborated=class_name_elaborated,
                                   post_construct=post_ctor,
                                   setters=''.join(setters).strip(),
                                   assignments=''.join(assignments).strip(),
                                   get_and_assign=''.join(get_and_assigns).strip(),
                                   using_alias=''.join(using_alias_statements).strip(),
                                   fields_declares=''.join(declarations).strip())
    return code


def generate_inclusion(includes):
    code = []
    for inc in includes:
        code.append('#include <%s>' % inc.spelling)
    return '\n'.join(code)


def generate_create(header, name, name_elaborated):
    return codegen_create(header, name, name_elaborated)


def generate_create_meth(header, name, name_elaborated):
    return codegen_create(header, name, name_elaborated, meth=True)


def generate_construct(fields, header, name, name_elaborated, has_post_ctor):
    print('construct: ', name, [(kind, n.spelling) for kind, n in fields])
    return codegen_construct(header, name, name_elaborated, fields, has_post_ctor)


def generate_construct_meth(fields, header, name, name_elaborated, has_post_ctor):
    print('construct: ', name, [(kind, n.spelling) for kind, n in fields])
    return codegen_construct(header, name, name_elaborated, fields, has_post_ctor, meth=True)


def generate_base_builder(fields, using_alias_values, header, name, name_elaborated, has_post_ctor):
    print('base_builder: ', name, [(kind, n.spelling) for kind, n in fields])
    return codegen_base_builder(header, name, name_elaborated, fields, using_alias_values, has_post_ctor)


def parse_compilation_database(build_directory):
    # check if the file exists first to suppress the clang warning.
    compile_commands_json = os.path.join(build_directory, 'compile_commands.json')
    if not os.path.isfile(compile_commands_json) or not os.access(compile_commands_json, os.R_OK):
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
    mod_name = inc_name[:-len(".vineyard.h")] + ".vineyard-mod"
    for inc in itertools.chain(system_includes, includes):
        target = os.path.join(inc, mod_name)
        if os.path.isfile(target) and os.access(target, os.R_OK):
            return os.path.join(inc, inc_name)
    return None


def parse_module(root_directory,
                 source,
                 target=None,
                 system_includes=None,
                 includes=None,
                 extra_flags=None,
                 build_directory=None,
                 delayed=True,
                 parse_only=True,
                 verbose=False):
    # prepare inputs
    content, message = validate_and_strip_input_file(source)
    if content is None:
        raise RuntimeError('Invalid input: %s' % message)
    unsaved_files = [(source, content)]

    # NB:
    #   `-nostdinc` and `-nostdinc++`: to avoid libclang find incorrect gcc installation.
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
    options = ParseOption.Default \
        | ParseOption.DetailedPreprocessingRecord \
        | ParseOption.SkipFunctionBodies \
        | ParseOption.IncludeAttributedTypes \
        | ParseOption.KeepGoing

    if parse_only:
        options |= ParseOption.SingleFileParse

    parse_flags = base_flags + flags
    unit = index.parse(source, unsaved_files=unsaved_files, args=parse_flags, options=options)

    if not parse_only:
        for diag in unit.diagnostics:
            if verbose or (diag.location and diag.location.file and \
                    diag.location.file.name == source):
                logging.warning(diag)

    # traverse
    modules = filter_the_module(unit.cursor, source)
    to_reflect, to_include = [], []
    for module in modules:
        if verbose:
            dump_ast(module, 0, set())
        traverse(module, to_reflect, to_include)

    return content, to_reflect, to_include, parse_flags


def parse_deps(root_directory,
               source,
               target=None,
               system_includes=None,
               includes=None,
               extra_flags=None,
               build_directory=None,
               delayed=True,
               verbose=False):
    _, _, to_include, parse_flags = parse_module(root_directory=root_directory,
                                                 source=source,
                                                 target=target,
                                                 system_includes=system_includes,
                                                 includes=includes,
                                                 extra_flags=extra_flags,
                                                 build_directory=build_directory,
                                                 delayed=delayed,
                                                 parse_only=True,
                                                 verbose=verbose)

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


def codegen(root_directory,
            source,
            target=None,
            system_includes=None,
            includes=None,
            extra_flags=None,
            build_directory=None,
            delayed=True,
            verbose=False):
    content, to_reflect, _, _ = parse_module(root_directory=root_directory,
                                             source=source,
                                             target=target,
                                             system_includes=system_includes,
                                             includes=includes,
                                             extra_flags=extra_flags,
                                             build_directory=build_directory,
                                             delayed=delayed,
                                             parse_only=False,
                                             verbose=verbose)

    logging.info('Generating for %s ...', os.path.basename(source))

    filename, _ = os.path.splitext(source)

    if target is not None:
        generated_file_path = target
    else:
        generated_file_path = '%s%s' % (filename, '.vineyard.h')

    real_root_path = os.path.realpath(root_directory)
    real_generate_path = os.path.realpath(generated_file_path)
    if real_root_path in real_generate_path:
        macro_guard_base = real_generate_path[len(real_root_path) + 1:]
    else:
        macro_guard_base = generated_file_path
    macro_guard = macro_guard_base.upper().replace('.', '_').replace('/', '_').replace('-', '_')

    with open(generated_file_path, 'w') as fp:
        code_injections = []
        code_blocks = []

        for _kind, namespaces, node in to_reflect:
            fields, using_alias, first_mmeber_offset, has_post_ctor = find_fields(node)
            name, ts = check_class(node)

            # get extend of using A = B
            using_alias_values = [(n, content[t.start.offset:t.end.offset]) for (n, t) in using_alias]

            # get extent of type parameters, since default value may involved.
            ts_names = [t for (t, _) in ts]  # without `typename`
            ts_name_values = [content[t.start.offset:t.end.offset] for (_, t) in ts]  # with `typename`

            name_elaborated = generate_template_type(name, ts_names)

            header = generate_template_header(ts_names)
            header_elaborated = generate_template_header(ts_name_values)

            meth_create = generate_create_meth(header, name, name_elaborated)
            meth_construct = generate_construct_meth(fields, header, name, name_elaborated, has_post_ctor)
            to_inject = '%s\n%s\n private:\n' % (meth_create, meth_construct)
            code_injections.append((first_mmeber_offset, to_inject))

            base_builder = generate_base_builder(fields, using_alias_values, header_elaborated, name, name_elaborated,
                                                 has_post_ctor)
            code_blocks.append((namespaces, base_builder))

        fp.write('#ifndef %s\n' % macro_guard)
        fp.write('#define %s\n\n' % macro_guard)

        # FIXME print the content of the original file
        offset = 0
        for next_offset, injection in code_injections:
            fp.write(content[offset:next_offset])
            fp.write(injection)
            offset = next_offset
        fp.write(content[offset:])

        for namespaces, block in code_blocks:
            if namespaces is not None:
                fp.write('\n\n')
                for ns in namespaces:
                    fp.write('namespace %s {\n' % ns)

            # print the code block
            fp.write(block)

            if namespaces is not None:
                fp.write('\n\n')
                for ns in namespaces:
                    fp.write('}  // namespace %s\n' % ns)
                fp.write('\n\n')

        fp.write('#endif // %s\n' % macro_guard)


def parse_sys_args():
    arg_parser = ArgumentParser()

    arg_parser.add_argument('-d',
                            '--dump-dependencies',
                            type=lambda x: bool(strtobool(x)),
                            nargs='?',
                            const=True,
                            default=False,
                            help='Just dump module dependencies, without code generation')
    arg_parser.add_argument('-r', '--root-directory', type=str, default='.', help='Root directory for code generation.')
    arg_parser.add_argument('-isystem',
                            '--system-includes',
                            type=str,
                            default='',
                            help='Directories that will been included in CMakeLists.txt')
    arg_parser.add_argument('-I',
                            '--includes',
                            type=str,
                            default='',
                            help='Directories that will been included in CMakeLists.txt')
    arg_parser.add_argument('-s', '--source', type=str, required=True, help='Data structure source file to parse')
    arg_parser.add_argument('-t', '--target', type=str, default=None, help='Output path to be generated')
    arg_parser.add_argument('-b',
                            '--build-directory',
                            type=str,
                            default=None,
                            help='Build directory that contains compilation database (compile_commands.json)')
    arg_parser.add_argument('-f',
                            '--extra-flags',
                            type=str,
                            action='append',
                            default=list(),
                            help='Extra flags that will be passed to libclang')
    arg_parser.add_argument('-v',
                            '--verbose',
                            type=lambda x: bool(strtobool(x)),
                            nargs='?',
                            const=True,
                            default=False,
                            help='Run codegen script with verbose output')
    return arg_parser.parse_args()


def main():
    args = parse_sys_args()
    if args.dump_dependencies:
        parse_deps(args.root_directory, args.source, args.target, args.system_includes, args.includes, args.extra_flags,
                   args.build_directory, args.verbose)
    else:
        codegen(args.root_directory, args.source, args.target, args.system_includes, args.includes, args.extra_flags,
                args.build_directory, args.verbose)


if __name__ == '__main__':
    main()
