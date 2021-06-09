import numpy as np
import pytest
import vineyard

from torch.utils.data import Dataset, DataLoader
from vineyard.core import default_builder_context, default_resolver_context
from vineyard.data import register_builtin_types

register_builtin_types(default_builder_context, default_resolver_context)


class RandomDataset(Dataset):
    def __init__(self, num):
        self.num = num
        self.ds = [(np.random.rand(4, 5, 6), np.random.rand(2, 3)) for i in range(num)]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.ds[idx]
        
def test_dataset(vineyard_client):
    ds = RandomDataset(10)
    object_id = vineyard_client.put(ds)
    new_ds = vineyard_client.get(object_id)

    for i in range(len(ds)):
        np.testing.assert_allclose(ds[i][0], new_ds[i][0])
        np.testing.assert_allclose(ds[i][1], new_ds[i][1])


