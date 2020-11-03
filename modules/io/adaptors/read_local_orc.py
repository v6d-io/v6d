import vineyard
import pyorc
import pyarrow as pa
import sys
import json

from vineyard.io.dataframe import DataframeStreamBuilder

def arrow_type(field):
    if field.name == 'decimal':
        return pa.decimal128(field.precision)
    elif field.name == 'uniontype':
        return pa.union(field.cont_types)
    elif field.name == 'array':
        return pa.list_(field.type)
    elif field.name == 'map':
        return pa.map_(field.key, field.value)
    elif field.name == 'struct':
        return pa.struct(field.fields)
    else:
        types = {
            'boolean': pa.bool_(),
            'tinyint': pa.int8(),
            'smallint': pa.int16(),
            'int': pa.int32(),
            'bigint': pa.int64(),
            'float': pa.float32(),
            'double': pa.float64(),
            'string': pa.string(),
            'char': pa.string(),
            'varchar': pa.string(),
            'binary': pa.binary(),
            'timestamp': pa.timestamp('ms'),
            'date': pa.date32(),
        }
        if field.name not in types:
            raise ValueError('Cannot convert to arrow type: ' + field.name)
        return types[field.name]

def read_local_orc(path, vineyard_socket):
    client = vineyard.connect(vineyard_socket)
    builder = DataframeStreamBuilder(client)
    stream = builder.seal(client)
    ret = {'type': 'return'}
    ret['content'] = repr(stream.id)
    print(json.dumps(ret))

    writer = stream.open_writer(client)

    with open(path, 'rb') as f:
        reader = pyorc.Reader(f)
        fields = reader.schema.fields
        schema = []
        for c in fields:
            schema.append((c, arrow_type(fields[c])))
        pa_struct = pa.struct(schema)
        while rows := reader.read(num=1024):
            rb = pa.RecordBatch.from_struct_array(pa.array(rows, type=pa_struct))
            sink = pa.BufferOutputStream()
            rb_writer = pa.ipc.new_stream(sink, rb.schema)
            rb_writer.write_batch(rb)
            rb_writer.close()
            buf = sink.getvalue()
            chunk = writer.next(buf.size)
            buf_writer = pa.FixedSizeBufferWriter(chunk)
            buf_writer.write(buf)
            buf_writer.close()

    writer.finish()
    return stream

if __name__ == '__main__':
    read_local_orc(sys.argv[1], sys.argv[2])