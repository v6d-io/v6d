import sys
import os
import pyarrow as pa
filename = sys.argv[1]
# file_dir = os.path.abspath()
with open(filename, 'rb') as source:
    with pa.ipc.open_stream(source) as reader:
        data = reader.read_all()
        print(data)