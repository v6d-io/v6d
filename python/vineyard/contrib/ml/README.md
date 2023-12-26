vineyard-ml: Accelerating Data Science Pipelines
================================================

Vineyard has been tightly integrated with the data preprocessing pipelines in
widely-adopted machine learning frameworks like PyTorch, TensorFlow, and MXNet.
Shared objects in vineyard, e.g., `vineyard::Tensor`, `vineyard::DataFrame`,
`vineyard::Table`, etc., can be directly used as the inputs of the training
and inference tasks in these frameworks.

Examples
--------

### Datasets

The following examples shows how `DataFrame` in vineyard can be used as the input
of Dataset for PyTorch:

```python
import os

import numpy as np
import pandas as pd

import torch
import vineyard

# connected to vineyard, see also: https://v6d.io/notes/getting-started.html
client = vineyard.connect(os.environ['VINEYARD_IPC_SOCKET'])

# generate a dummy dataframe in vineyard
df = pd.DataFrame({
    # multi-dimensional array as a column
    'data': vineyard.data.dataframe.NDArrayArray(np.random.rand(1000, 10)),
    'label': np.random.rand(1000)
})
object_id = client.put(df)

# take it as a torch dataset
from vineyard.contrib.ml.torch import torch_context
with torch_context():
    # ds is a `torch.utils.data.TensorDataset`
    ds = client.get(object_id)

# or, you can use datapipes from torchdata
from vineyard.contrib.ml.torch import datapipe
pipe = datapipe(ds)

# use the datapipes in your training loop
for data, label in pipe:
    # do something
    pass
```

### Pytorch Modules

The following example shows how to use vineyard to share pytorch modules between processes:

```python
import torch
import vineyard

# connected to vineyard, see also: https://v6d.io/notes/getting-started.html
client = vineyard.connect(os.environ['VINEYARD_IPC_SOCKET'])

# generate a dummy model in vineyard
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

model = Model()

# put the model into vineyard
from vineyard.contrib.ml.torch import torch_context
with torch_context():
    object_id = client.put(model)

# get the module state dict from vineyard and load it into a new model
model = Model()
with torch_context():
    state_dict = client.get(object_id)
model.load_state_dict(state_dict, assign=True)
```

Reference and Implementation
----------------------------

- [torch](https://github.com/v6d-io/v6d/blob/main/python/vineyard/contrib/ml/torch.py): including PyTorch datasets, torcharrow and torchdata.
- [tensorflow](https://github.com/v6d-io/v6d/blob/main/python/vineyard/contrib/ml/tensorflow.py)
- [mxnet](https://github.com/v6d-io/v6d/blob/main/python/vineyard/contrib/ml/mxnet.py)

For more details about vineyard itself, please refer to the [Vineyard](https://v6d.io) project.
