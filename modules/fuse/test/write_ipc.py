import vineyard as vy
import pyarrow as pa
import numpy as np
import pandas as pd
import sys
# default_socket = "/var/run/vineyard.sock"

# client = vy.connect(default_socket);
filename = sys.argv[1]
height,width = 5,4
df = pd.DataFrame(np.random.randint(0,100,size=(height, width)), columns=list('ABCD'))
pdf = pa.Table.from_pandas(df)

with open(filename, 'wb') as source:
    with pa.ipc.RecordBatchStreamWriter(source,pdf.schema) as writer:
        data = writer.write(pdf)
