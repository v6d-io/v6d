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

"""``VineyardDataSet`` is a data set implementation which handles in-memory data.
   stored in vineyard.
"""

from typing import Any
from typing import Dict
from typing import Optional

from kedro.io.core import AbstractDataSet

_EMPTY = object()


class VineyardDataSet(AbstractDataSet):
    """``Vineyard`` loads and saves Python objects from/to vineyard.

    Example:
    ::

        >>> from vineyard.contrib.kedro.io import VineyardDataSet
        >>> import pandas as pd
        >>>
        >>> data = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5],
        >>>                      'col3': [5, 6]})
        >>> data_set = VineyardDataSet(ds_name="data")
        >>>
        >>> data_set.save(data)
        >>> loaded_data = data_set.load()
        >>> assert loaded_data.equals(data)
        >>>
        >>> new_data = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5]})
        >>> data_set.save(new_data)
        >>> reloaded_data = data_set.load()
        >>> assert reloaded_data.equals(new_data)

    """

    def __init__(self, ds_name: str, vineyard_ipc_socket: Optional[str] = None):
        """Creates a new instance of ``MemoryDataSet`` pointing to the
        provided Python object.

        Args:
            data: Python object containing the data.
            copy_mode: The copy mode used to copy the data. Possible
                values are: "deepcopy", "copy" and "assign". If not
                provided, it is inferred based on the data type.
        """
        self.ds_name = ds_name
        self.vineyard_ipc_socket = vineyard_ipc_socket

        # vineyard client: lazy initialization
        self._client = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_client']
        return state

    def _init_client(self):
        if self._client is None:
            import vineyard

            self._client: vineyard.IPCClient = vineyard.connect(
                self.vineyard_ipc_socket
            )

    def _load(self) -> Any:
        self._init_client()
        return self._client.get(name=self.ds_name, fetch=True)

    def _save(self, data: Any):
        self._init_client()
        self._release()  # release the (possible) old data
        self._client.put(data, name=self.ds_name, persist=True)

    def _exists(self) -> bool:
        self._init_client()
        try:
            return self._client.exists(self._client.get_name(self.ds_name))
        except:  # noqa: E722, pylint: disable=bare-except
            return False

    def _release(self) -> None:
        self._init_client()
        try:
            object_id = self._client.get_name(self.ds_name)
        except:  # noqa: E722, pylint: disable=bare-except
            return
        self._client.delete(object_id)
        self._client.drop_name(self.ds_name)

    def _describe(self) -> Dict[str, Any]:
        return {"name": self.ds_name, "vineyard_ipc_socket": self.vineyard_ipc_socket}
