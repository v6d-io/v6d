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

import lazy_import
import pytest

from vineyard.contrib.ml.torcharrow import torcharrow_context

ta = lazy_import.lazy_module("torcharrow")


@pytest.fixture(scope="module", autouse=True)
def vineyard_for_torcharrow():
    with torcharrow_context():
        yield


def test_torch_arrow_column(vineyard_client):
    s = ta.column([1, 2, None, 4])
    assert s.sum() == 7

    s = vineyard_client.get(vineyard_client.put(s))
    assert isinstance(s, ta.Column)
    assert s.sum() == 7


def test_torch_arrow_dataframe(vineyard_client):
    s = ta.dataframe({"a": [1, 2, None, 4], "b": [5, 6, None, 8]})
    assert s.sum()['a'][0] == 7
    assert s.sum()['b'][0] == 19

    s = vineyard_client.get(vineyard_client.put(s))
    assert isinstance(s, ta.DataFrame)
    assert s.sum()['a'][0] == 7
    assert s.sum()['b'][0] == 19
