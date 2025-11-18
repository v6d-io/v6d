import vineyard
import torch
from vineyard.contrib.ml.torch import torch_context

client = vineyard.connect('/var/run/vineyard.sock1')

#create a random tensor
#x = torch.rand(100, 100, 100)
#client.delete(name="test_lazy_tensor")
#with torch_context(client):
#    client.put(x, name="test_lazy_tensor", persist=True)

test_kwargs = {'test': 'test'}
with torch_context(client):
    lazy_tensor = client.lazy_get(name="test_lazy_tensor111", **test_kwargs)
"""
try:
    for i in range(8):
        name = "test_lazy_tensor" + str(i)
        print("Creating tensor", name)
        lazy_tensor = client.lazy_get("test_lazy_tensor")
except Exception as e:
    print(e)
"""

# At a later point, attempt to access the tensor

try:
    if lazy_tensor.is_ready():
        print("Tensor is ready")
        #try:
        #    tensor = lazy_tensor.get()
        #    print(tensor)
        #except RuntimeError as e:
        #    pass
    else:
        print("Tensor is not ready")
        #import time
        #time.sleep(5)
        # print("Waiting for 5 seconds")
        #if lazy_tensor.is_ready():
        #    print("Tensor is ready")
        #    tensor = lazy_tensor.get()
        #    print(tensor)
except Exception as e:
    print(e)

print("Done")