import vineyard
import pyorc
import pyarrow as pa
import sys
import json

from vineyard.io.dataframe import DataframeStreamBuilder

def orc_type(field):
    if pa.is_int(field):
        return 'int'
    elif pa.is_string(field):
        return 'string'
    raise ValueError('Cannot Convert %s' % field)

def write_local_orc(stream_id, path, vineyard_socket):
    client = vineyard.connect(vineyard_socket)
    stream = client.get(stream_id)
    reader = stream.open_reader(client)

    writer = None
    with open(path, 'wb') as f:
        while buf := reader.next():
            buf_reader = pa.ipc.open_stream(buf)
            if writer is None:
                #get schema
                schema = []
                for field in buf_reader.schema():
                    colname = field.name
                    typename = orc_type(field.type)
                    schema.append(f'{colname}:{typename}')
                schema = ','.join(schema)
                writer = pyorc.Writer(f, f'struct<{schema}>')
            for batch in buf_reader:
                df = batch.to_pandas()
                writer.writerows(df.itertuples())


if __name__ == '__main__':
    write_local_orc(sys.argv[1], sys.argv[2])