import pyarrow as pa
import sys
with open(sys.argv[1], 'rb') as source:
   with pa.ipc.open_stream(source) as reader:
      df = reader.read_all()
      print(df)
