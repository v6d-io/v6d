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
# The NDArrayDtype and NDArrayArray is partially derived from the tensorpandas project,
#
#   https://github.com/ig248/tensorpandas/blob/master/tensorpandas/base.py
#


import numbers
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd
from numpy.lib.mixins import NDArrayOperatorsMixin
from pandas.core.arrays import PandasArray
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import PandasExtensionDtype
from pandas.core.indexers import check_array_indexer
from pandas.core.internals.managers import BlockManager

try:
    from pandas.core.internals.blocks import BlockPlacement
    from pandas.core.internals.blocks import NumpyBlock as Block
except ImportError:
    BlockPlacement = None
    from pandas.core.internals.blocks import Block
try:
    from pandas.core.indexes.base import ensure_index
except ImportError:
    try:
        from pandas.core.indexes.base import _ensure_index as ensure_index
    except ImportError:
        from pandas.indexes.base import _ensure_index as ensure_index

try:
    from pandas.core.internals.blocks import DatetimeLikeBlock
except ImportError:
    try:
        from pandas.core.internals.blocks import DatetimeBlock as DatetimeLikeBlock
    except ImportError:
        from pandas.core.internals import DatetimeBlock as DatetimeLikeBlock

try:
    from pandas.core.arrays.datetimes import DatetimeArray
except ImportError:
    try:
        from pandas.core.arrays import DatetimeArray
    except ImportError:
        DatetimeArray = None


from vineyard._C import Object
from vineyard._C import ObjectID
from vineyard._C import ObjectMeta
from vineyard.data.tensor import ndarray
from vineyard.data.utils import expand_slice
from vineyard.data.utils import from_json
from vineyard.data.utils import normalize_dtype
from vineyard.data.utils import to_json


class registry_type(type):
    """Fix registry lookup for extension types.

    It appears that parquet stores `str(TensorDtype)`, yet the
    lookup tries to match it to `TensorDtype.name`.
    """

    def __str__(self):
        try:
            return self.name
        except AttributeError:
            return self.__name__


def _infer_na_value(dtype):
    dtype = np.dtype(dtype)
    na_value = None
    if np.issubdtype(dtype, np.datetime64):
        na_value = dtype.type("NaT")
    return na_value


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.extensions.ExtensionDtype.html
@pd.api.extensions.register_extension_dtype
class NDArrayDtype(PandasExtensionDtype, metaclass=registry_type):
    # kind = "O"
    type = np.ndarray
    _metadata = ("shape", "_dtype")
    _cache: Dict[tuple, "NDArrayDtype"] = {}

    def __new__(cls, shape=(), dtype=None):
        if not isinstance(shape, tuple):
            raise TypeError("Shape must be a tuple")
        try:
            dtype = np.dtype(dtype)
        except TypeError as err:
            raise ValueError(f"{dtype} is not a valid dtype") from err
        if (shape, dtype) not in cls._cache:
            cls._cache[(shape, dtype)] = super().__new__(cls)
        return cls._cache[(shape, dtype)]

    def __init__(self, shape=(), dtype=None):
        self.shape = shape
        # we can not use .dtype as it is leads to conflicts in e.g.
        # is_extension_array_dtype
        self._dtype = np.dtype(dtype)

    @classmethod
    def construct_from_string(cls, string):
        if string == "NDArray":
            return cls()
        try:
            tdtype = eval(string, {}, {"NDArray": cls, "dtype": np.dtype})
            if not isinstance(tdtype, cls):
                raise TypeError(
                    f"Type expression evaluated to {tdtype} of type {type(tdtype)}"
                    " - expected a NDArrayDType instance."
                )
            return tdtype
        except Exception as err:
            raise TypeError(
                f"Cannot construct a '{cls.__name__}' from '{string}'"
            ) from err

    @property
    def na_value(self):
        na_value = _infer_na_value(self._dtype)
        if na_value is not None:
            na = np.full(self.shape, na_value, dtype=self._dtype)
        else:
            na = np.nan + np.empty(self.shape, dtype=self._dtype)
        return na

    def __str__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return f"NDArray(shape={self.shape!r}, dtype={self._dtype!r})"

    def __hash__(self) -> int:
        # make myself hashable
        return hash(str(self))

    @classmethod
    def construct_array_type(cls):
        return NDArrayArray

    def __from_numpy__(self, array: np.ndarray) -> pd.api.extensions.ExtensionArray:
        """Construct NDArrayArray from numpy ndarray."""
        return NDArrayArray(array)


class NDArrayArray(
    pd.api.extensions.ExtensionArray,
    pd._libs.arrays.NDArrayBacked,
    NDArrayOperatorsMixin,
):
    def __init__(self, data, dim=1):
        """Initialize from an nd-array or list of arrays."""
        if isinstance(data, self.__class__):
            ndarray = data._ndarray
        elif (
            isinstance(data, (np.ndarray, DatetimeArray)) and data.dtype != object
        ):  # i.e. not array of arrays
            # NB: in pd1.3, DatetimeArray is returned by some operations
            # where previously an ndarray was returned
            ndarray = np.array(data)
        else:
            ndarray = np.stack(data)
        self._dim = dim
        pd._libs.arrays.NDArrayBacked.__init__(self, ndarray, ndarray.dtype)

    # Attributes
    @property
    def dtype(self):
        return NDArrayDtype(shape=self.ndarray_shape, dtype=self.ndarray_dtype)

    @property
    def size(self):
        return len(self)

    @property
    def shape(self):
        return self._ndarray.shape[0 : self._dim]

    @property
    def ndim(self):
        return self._dim

    @property
    def nbytes(self) -> int:
        """Return number of bytes needed to store this object in memory."""
        return self._ndarray.nbytes

    def __len__(self):
        return self._ndarray.shape[0]

    @property
    def ndarray_shape(self):
        return self._ndarray.shape[self._dim :]

    @property
    def ndarray_ndim(self):
        return self._ndarray.ndim - self._dim

    @property
    def ndarray_dtype(self):
        return self._ndarray.dtype

    def __getitem__(self, item):
        if isinstance(item, type(self)):
            item = item._ndarray
        item = check_array_indexer(self, item)
        result = self._ndarray[item]
        if result.ndim + self._dim <= self._ndarray.ndim:
            return result
        else:
            return self.__class__(
                result, dim=self._dim - (self._ndarray.ndim - result.ndim)
            )

    def __setitem__(self, key: Union[int, np.ndarray], value: Any) -> None:
        """Set one or more values inplace.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.
        """
        self._ndarray[key] = value

    # Methods
    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(scalars)

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(np.concatenate([arr._ndarray for arr in to_concat]))

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, NDArrayDtype):
            return self.copy() if copy else self
        return np.array(self, dtype=dtype, copy=copy)

    def isna(self):
        return np.any(np.isnan(self._ndarray), axis=tuple(range(1, self._ndarray.ndim)))

    def take(
        self, indices: Sequence[int], allow_fill: bool = False, fill_value: Any = None
    ) -> "NDArrayArray":
        if fill_value is None:
            fill_value = self.dtype.na_value
        _indices = np.array(indices)
        _result = np.full(
            (len(_indices), *self.ndarray_shape), fill_value, dtype=self.dtype._dtype
        )
        if allow_fill:
            if np.any((_indices < 0) & (_indices != -1)):
                raise ValueError("Fill points must be indicated by -1")
            destination = _indices >= 0  # boolean
            _indices = _indices[_indices >= 0]
        else:
            destination = slice(None, None, None)
        if len(_indices) > 0 and not self._ndarray.shape[0]:
            raise IndexError("cannot do a non-empty take")
        _result[destination] = self._ndarray[_indices]

        return self.__class__(_result)

    def copy(self):
        return self.__class__(self._ndarray.copy(), self._dim)

    def view(self):
        return self.__class__(self._ndarray, self._dim)

    def __array__(self, dtype=None):
        if dtype == np.dtype(object):
            # Return a 1D array for pd.array() compatibility
            return np.array([*self._ndarray, None], dtype=object)[:-1]
        return self._ndarray

    # adopted from PandasArray
    _HANDLED_TYPES = (np.ndarray, numbers.Number, PandasArray)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (self.__class__,)):
                return NotImplemented

        # Ther doesn't seem to be a way of knowing if another array came from
        # a PandasArray.
        #
        # This creates a huge confusion between column and row arrays.
        def _as_array(x):
            if isinstance(x, self.__class__):
                x = x._ndarray
            return x

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(_as_array(x) for x in inputs)
        if out:
            kwargs["out"] = tuple(_as_array(x) for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple and len(result):
            # multiple return values
            if not pd._libs.is_scalar(result[0]):
                # re-box array-like results
                return tuple(type(self)(x) for x in result)
            else:
                # but not scalar reductions
                return result
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            if not pd._libs.is_scalar(result):
                # re-box array-like results, but not scalar reductions
                result = type(self)(result)
            return result


@pd.api.extensions.register_series_accessor("ndarray")
class NDArrayAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if not isinstance(obj.dtype, NDArrayDtype):
            raise AttributeError("Can only use .ndarray accessor with ndarray values")

    @property
    def ndarray(self):
        return self._obj.values

    @property
    def values(self):
        return self.ndarray._ndarray

    @values.setter
    def values(self, new_values):
        pd._libs.arrays.NDArrayBacked.__init__(self, new_values, new_values.dtype)

    @property
    def dtype(self):
        return self.ndarray.ndarray_dtype

    @property
    def ndim(self):
        return self.ndarray.ndarray_ndim

    @property
    def shape(self):
        return self.ndarray.ndarray_shape


class NDArrayBlock(pd.core.internals.blocks.NDArrayBackedExtensionBlock):
    """Block for multi-dimensional ndarray."""

    __slots__ = ()
    is_numeric = False
    values: NDArrayArray

    def values_for_json(self) -> np.ndarray:
        return self.values._ndarray


def pandas_dataframe_builder(client, value, builder, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::DataFrame'
    meta['columns_'] = to_json(value.columns.values.tolist())
    meta.add_member('index_', builder.run(client, value.index))

    # accumulate columns
    value_columns = [None] * len(value.columns)
    for block in value._mgr.blocks:
        slices = list(expand_slice(block.mgr_locs.indexer))
        if isinstance(block.values, pd.arrays.SparseArray):
            assert len(slices) == 1
            value_columns[slices[0]] = block.values
        elif len(slices) == 1 and isinstance(block.values, NDArrayArray):
            value_columns[slices[0]] = np.array(block.values)
            vineyard_ref = getattr(block.values, '__vineyard_ref', None)
            # the block comes from vineyard
            if vineyard_ref is not None:
                setattr(value_columns[slices[0]], '__vineyard_ref', vineyard_ref)
        elif len(slices) == 1:
            value_columns[slices[0]] = block.values[0]
            vineyard_ref = getattr(block.values, '__vineyard_ref', None)
            # the block comes from vineyard
            if vineyard_ref is not None:
                setattr(value_columns[slices[0]], '__vineyard_ref', vineyard_ref)
        else:
            for index, column_index in enumerate(slices):
                value_columns[column_index] = block.values[index]

    for index, name in enumerate(value.columns):
        meta['__values_-key-%d' % index] = to_json(name)
        meta.add_member(
            '__values_-value-%d' % index, builder.run(client, value_columns[index])
        )
    meta['nbytes'] = 0  # FIXME
    meta['__values_-size'] = len(value.columns)
    meta['partition_index_row_'] = kw.get('partition_index', [-1, -1])[0]
    meta['partition_index_column_'] = kw.get('partition_index', [-1, -1])[1]
    meta['row_batch_index_'] = kw.get('row_batch_index', 0)
    return client.create_metadata(meta)


def pandas_dataframe_resolver(obj, resolver):
    meta = obj.meta
    columns = from_json(meta['columns_'])
    if not columns:
        return pd.DataFrame()

    names = []
    # ensure zero-copy
    blocks = []
    index_size = 0
    for idx, _ in enumerate(columns):
        names.append(from_json(meta['__values_-key-%d' % idx]))
        np_value = resolver.run(obj.member('__values_-value-%d' % idx))
        index_size = len(np_value)
        # ndim: 1 for SingleBlockManager/Series, 2 for BlockManager/DataFrame
        if BlockPlacement:
            placement = BlockPlacement(slice(idx, idx + 1, 1))
        else:
            placement = slice(idx, idx + 1, 1)
        if DatetimeArray is not None and isinstance(np_value, DatetimeArray):
            values = np_value.reshape(1, -1)
            setattr(values, '__vineyard_ref', getattr(np_value, '__vineyard_ref', None))
            block = DatetimeLikeBlock(values, placement, ndim=2)
        elif len(np_value.shape) == 1:
            values = np.expand_dims(np_value, 0).view(ndarray)
            setattr(values, '__vineyard_ref', getattr(np_value, '__vineyard_ref', None))
            block = Block(values, placement, ndim=2)
        else:
            values = NDArrayArray(np.expand_dims(np_value, 0).view(ndarray), dim=2)
            setattr(values, '__vineyard_ref', getattr(np_value, '__vineyard_ref', None))
            block = NDArrayBlock(values, placement, ndim=2)
        blocks.append(block)
    if 'index_' in meta:
        index = resolver.run(obj.member('index_'))
    else:
        index = pd.RangeIndex(index_size)
    return pd.DataFrame(BlockManager(blocks, [ensure_index(names), index]))


def pandas_sparse_array_builder(client, value, builder, **kw):
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::SparseArray<%s>' % value.dtype.name
    meta['value_type_'] = value.dtype.name
    sp_index_type, (sp_index_size, sp_index_array) = value.sp_index.__reduce__()
    meta['sp_index_name'] = sp_index_type.__name__
    meta['sp_index_size'] = sp_index_size
    meta.add_member('sp_index', builder.run(client, sp_index_array, **kw))
    meta.add_member('sp_values', builder.run(client, value.sp_values, **kw))
    return client.create_metadata(meta)


def pandas_sparse_array_resolver(obj, resolver):
    meta = obj.meta
    value_type = normalize_dtype(meta['value_type_'])
    sp_index_type = getattr(pd._libs.sparse, meta['sp_index_name'])
    sp_index_size = meta['sp_index_size']
    sp_index_array = resolver.run(obj.member('sp_index'))
    sp_index = sp_index_type(sp_index_size, sp_index_array)
    sp_values = resolver.run(obj.member('sp_values'))
    return pd.arrays.SparseArray(sp_values, sparse_index=sp_index, dtype=value_type)


def make_global_dataframe(client, blocks, extra_meta=None) -> ObjectMeta:
    meta = ObjectMeta()
    meta['typename'] = 'vineyard::GlobalDataFrame'
    meta.set_global(True)
    meta['partitions_-size'] = len(blocks)
    if extra_meta:
        for k, v in extra_meta.items():
            meta[k] = v

    # assume chunks are split over the row axis
    if 'partition_shape_row_' not in meta:
        meta['partition_shape_row_'] = len(blocks)
    if 'partition_shape_column_' not in meta:
        meta['partition_shape_column_'] = 1

    for idx, block in enumerate(blocks):
        if not isinstance(block, (ObjectMeta, ObjectID, Object)):
            block = ObjectID(block)
        meta.add_member('partitions_-%d' % idx, block)

    gtensor_meta = client.create_metadata(meta)
    client.persist(gtensor_meta)
    return gtensor_meta


def global_dataframe_resolver(obj, resolver):
    """Return a list of dataframes."""
    meta = obj.meta
    num = int(meta['partitions_-size'])

    dataframes = []
    orders = []
    for i in range(num):
        df = meta.get_member('partitions_-%d' % i)
        if df.meta.islocal:
            dataframes.append(resolver.run(df))
            if 'row_batch_index_' in df.meta:
                orders.append(df.meta["row_batch_index_"])
    if orders != sorted(orders):
        raise ValueError("Bad dataframe orders:", orders)
    return dataframes


def register_dataframe_types(builder_ctx, resolver_ctx):
    if builder_ctx is not None:
        builder_ctx.register(pd.DataFrame, pandas_dataframe_builder)
        builder_ctx.register(pd.arrays.SparseArray, pandas_sparse_array_builder)

    if resolver_ctx is not None:
        resolver_ctx.register('vineyard::DataFrame', pandas_dataframe_resolver)
        resolver_ctx.register('vineyard::SparseArray', pandas_sparse_array_resolver)
        resolver_ctx.register("vineyard::GlobalDataFrame", global_dataframe_resolver)
