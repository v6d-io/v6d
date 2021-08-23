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


def find_most_precise_match(typename, candidates):
    ''' Find the most precise match for given typename inside a group of prefixes.

        Parameters
        ----------
        typename: str
            Given type name to be matched with prefixes.

        candidates: list of (prefix, candidate)
            List of candidates to match with, the first element is prefix, and the second
            entry is the candidate item, e.g., resolver or driver method.
    '''
    if candidates:
        for prefix, candidate in candidates.items():
            if typename.startswith(prefix):
                return prefix, candidate
    return None, None


__all__ = ['find_most_precise_match']
