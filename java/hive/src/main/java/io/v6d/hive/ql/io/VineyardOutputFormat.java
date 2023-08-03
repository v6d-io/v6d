/** Copyright 2020-2023 Alibaba Group Holding Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.v6d.hive.ql.io;

import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.modules.basic.arrow.TableBuilder;
import io.v6d.modules.basic.columnar.ColumnarDataBuilder;
import io.v6d.modules.basic.arrow.SchemaBuilder;
import io.v6d.modules.basic.arrow.Table;
import io.v6d.modules.basic.arrow.Arrow;
import io.v6d.modules.basic.arrow.RecordBatchBuilder;

import java.io.IOException;
import java.util.Properties;
import java.util.ArrayList;
import java.util.List;

import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.Types;
import org.apache.arrow.vector.types.pojo.*;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.hive.ql.exec.FileSinkOperator;
import org.apache.hadoop.hive.ql.io.HiveOutputFormat;
import org.apache.hadoop.hive.ql.io.arrow.ArrowWrapperWritable;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.typeinfo.DecimalTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.PrimitiveTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Progressable;
import org.apache.hadoop.hive.common.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VineyardOutputFormat<K extends NullWritable, V extends VineyardRowWritable>
        implements HiveOutputFormat<K, V> {
    private static Logger logger = LoggerFactory.getLogger(VineyardOutputFormat.class);

    @Override
    public FileSinkOperator.RecordWriter getHiveRecordWriter(
            JobConf jc,
            Path finalOutPath,
            Class<? extends Writable> valueClass,
            boolean isCompressed,
            Properties tableProperties,
            Progressable progress)
            throws IOException {
        return new SinkRecordWriter(
                jc, finalOutPath, valueClass, isCompressed, tableProperties, progress);
    }

    @Override
    public RecordWriter<K, V> getRecordWriter(
            FileSystem fileSystem, JobConf jobConf, String s, Progressable progressable)
            throws IOException {
        return new MapredRecordWriter<K, V>();
    }

    @Override
    public void checkOutputSpecs(FileSystem fileSystem, JobConf jobConf) throws IOException {}
}

class SinkRecordWriter implements FileSinkOperator.RecordWriter {
    private JobConf jc;
    private Path finalOutPath;
    private FileSystem fs;
    private List<VineyardRowWritable> objects;
    private Schema schema;
    private Properties tableProperties;
    private Progressable progress;

    // vineyard
    private IPCClient client;
    private TableBuilder tableBuilder;
    private SchemaBuilder schemaBuilder;
    RecordBatchBuilder recordBatchBuilder;
    private String tableName;

    public static final PathFilter VINEYARD_FILES_PATH_FILTER = new PathFilter() {
        @Override
        public boolean accept(Path p) {
            String name = p.getName();
            return name.startsWith("_task_tmp.") && (!name.substring(10, name.length()).startsWith("-"));
        }
    };

    private void getTableName() {
        String location = tableProperties.getProperty("location");
        // System.out.println("finalOutPath : "+ finalOutPath.toString());

        // Get partition count
        // String partition = tableProperties.getProperty("partition_columns.types");
        // int index = -1;
        // int partitionCount= 0;
        // if (partition != null) {
        //     do {
        //         partitionCount++;
        //         index = partition.indexOf(":", index + 1);
        //     } while(index != -1);
        // }
        // System.out.println("Partition count:" + partitionCount);

        // Construct table name
        String path = finalOutPath.toString();
        path = path.substring(location.length(), path.length());
        String pathSplits[] = path.split("/");
        PathFilter vineyardPathFilter = VINEYARD_FILES_PATH_FILTER;
        PathFilter hiddernPathFilter = FileUtils.HIDDEN_FILES_PATH_FILTER;
        if (pathSplits.length == 0) {
            return;
        }
        tableName = location;
        for (int i = 1; i < pathSplits.length; i++) {
            if (pathSplits[i].length() > 0 && hiddernPathFilter.accept(new Path(pathSplits[i]))) {
                System.out.println("path split:" + pathSplits[i]);
                tableName += "/" + pathSplits[i];
            } else if (pathSplits[i].length() > 0 && vineyardPathFilter.accept(new Path(pathSplits[i]))) {
                System.out.println("path split:" + pathSplits[i].substring(10, pathSplits[i].length()));
                tableName += "/" + pathSplits[i].substring(10, pathSplits[i].length());
            }
        }
        tableName = tableName.replaceAll("/", "#");
        // System.out.println("Table name:" + tableName);

        // Create temp file
        String tmpFilePath = finalOutPath.toString();
        tmpFilePath = tmpFilePath.substring(0, tmpFilePath.lastIndexOf("/"));
        // System.out.println("out path:" + tmpFilePath);
        tmpFilePath = tmpFilePath.replaceAll("_task", "");
        Path tmpPath = new Path(tmpFilePath, "vineyard.tmp");
        try {
            fs = finalOutPath.getFileSystem(jc);
            System.out.println("tmp path:" + tmpPath.toString());
            FSDataOutputStream output = FileSystem.create(fs, tmpPath, new FsPermission("777"));
            if (output != null) {
                System.out.println("Create succeed!");
                // output.write("test".getBytes(), 0, 4);
                output.close();
            }
        } catch (Exception e) {
            System.out.println("Create failed!");
        }
    }

    @lombok.SneakyThrows
    public SinkRecordWriter(
            JobConf jc,
            Path finalOutPath,
            Class<? extends Writable> valueClass,
            boolean isCompressed,
            Properties tableProperties,
            Progressable progress) {
        this.jc = jc;
        if (!ArrowWrapperWritable.class.isAssignableFrom(valueClass)) {
            throw new VineyardException.Invalid("value class must be ArrowWrapperWritable");
        }
        if (isCompressed) {
            throw new VineyardException.Invalid("compressed output is not supported");
        }
        this.finalOutPath = finalOutPath;
        this.tableProperties = tableProperties;
        this.progress = progress;

        for (Object key : tableProperties.keySet()) {
            System.out.printf("table property: %s, %s\n", key, tableProperties.getProperty((String) key));
        }
        getTableName();

        // connect to vineyard
        if (client == null) {
            // TBD: get vineyard socket path from table properties
            client = new IPCClient("/tmp/vineyard/vineyard.sock");
        }
        if (client == null || !client.connected()) {
            throw new VineyardException.Invalid("failed to connect to vineyard");
        } else {
            System.out.printf("Connected to vineyard succeed!\n");
            System.out.printf("Hello vineyard!\n");
        }

        objects = new ArrayList<VineyardRowWritable>();
        Arrow.instantiate();
    }

    @Override
    public void write(Writable w) throws IOException {   
        if (w == null) {
            System.out.println("w is null");
            return;
        }

        VineyardRowWritable rowWritable = (VineyardRowWritable) w;
        objects.add(rowWritable);
    }
    
    // check if the table is already created.
    // if not, create a new table.
    // if yes, append the data to the table.(Get from vineyard, and seal it in a new table) 
    @Override
    public void close(boolean abort) throws IOException {
        Table oldTable = null;
        if (objects.size() == 0) {
            System.out.println("No data to write.");
            client.disconnect();
            System.out.println("Bye, vineyard!");
            return;
        }

        // construct record batch
        constructRecordBatch();
        tableBuilder = new TableBuilder(client, schemaBuilder);
        try {
            ObjectID objectID = client.getName(tableName, false);
            if (objectID == null) {
                System.out.println("Table not exist.");
            } else {
                oldTable = (Table) ObjectFactory.getFactory().resolve(client.getMetaData(objectID));
            }
        } catch (Exception e) {
            System.out.println("Get table id failed");
        }

        if (oldTable != null) {
            for (int i = 0; i < oldTable.getBatches().size(); i++) {
                tableBuilder.addBatch(oldTable.getBatches().get(i));
            }
        }

        tableBuilder.addBatch(recordBatchBuilder);
        try {
            ObjectMeta meta = tableBuilder.seal(client);
            System.out.println("Table id in vineyard:" + meta.getId().value());
            client.persist(meta.getId());
            System.out.println("Table persisted, name:" + tableName);
            client.putName(meta.getId(), tableName);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            throw new IOException("Seal TableBuilder failed");
        }
        client.disconnect();

        System.out.println("Bye, vineyard!");
    }

    private static ArrowType toArrowType(TypeInfo typeInfo) {
        switch (typeInfo.getCategory()) {
            case PRIMITIVE:
                switch (((PrimitiveTypeInfo) typeInfo).getPrimitiveCategory()) {
                    case BOOLEAN:
                        return Types.MinorType.BIT.getType();
                    case BYTE:
                        return Types.MinorType.TINYINT.getType();
                    case SHORT:
                        return Types.MinorType.SMALLINT.getType();
                    case INT:
                        return Types.MinorType.INT.getType();
                    case LONG:
                        return Types.MinorType.BIGINT.getType();
                    case FLOAT:
                        return Types.MinorType.FLOAT4.getType();
                    case DOUBLE:
                        return Types.MinorType.FLOAT8.getType();
                    case STRING:
                    case VARCHAR:
                    case CHAR:
                        return Types.MinorType.VARCHAR.getType();
                    case DATE:
                        return Types.MinorType.DATEDAY.getType();
                    case TIMESTAMP:
                        return new ArrowType.Timestamp(TimeUnit.MICROSECOND, "UTC");
                    case BINARY:
                        return Types.MinorType.VARBINARY.getType();
                    // case DECIMAL:
                    //     final DecimalTypeInfo decimalTypeInfo = (DecimalTypeInfo) typeInfo;
                    //     return new ArrowType.Decimal(decimalTypeInfo.precision(), decimalTypeInfo.scale(), 128);
                    case INTERVAL_YEAR_MONTH:
                        return Types.MinorType.INTERVALYEAR.getType();
                    case INTERVAL_DAY_TIME:
                        return Types.MinorType.INTERVALDAY.getType();
                    case VOID:
                    case TIMESTAMPLOCALTZ:
                    case UNKNOWN:
                    default:
                        throw new IllegalArgumentException();
                }
            case LIST:
                return ArrowType.List.INSTANCE;
            case STRUCT:
                return ArrowType.Struct.INSTANCE;
            // case MAP:
            //     return new ArrowType.Map(false);
            case UNION:
                default:
                throw new IllegalArgumentException();
        }
    }

    private void constructRecordBatch() {
        // construct schema
        // StructVector rootVector = StructVector.empty(null, allocator);
        List<Field> fields = new ArrayList<Field>();
        int columns = objects.get(0).getTargetTypeInfos().length;
        for (int i = 0; i < columns; i++) {
            Field field = Field.nullable(objects.get(0).getTargetTypeInfos()[i].getTypeName(), toArrowType(objects.get(0).getTargetTypeInfos()[i]));
            fields.add(field);
        }
        schema = new Schema((Iterable<Field>)fields);
        schemaBuilder = SchemaBuilder.fromSchema(schema);

        // construct recordBatch
        try {
            recordBatchBuilder = new RecordBatchBuilder(client, schema, objects.size());
            System.out.println("Create done!");
            fillRecordBatchBuilder(schema);
            System.out.println("Fill done!");
        } catch (Exception e) {
            System.out.println(e.getMessage());
            System.out.println("Create record batch builder failed");
        }
    }

    private void fillRecordBatchBuilder(Schema schema) {
        int rowCount = objects.size();
        for (int i = 0; i < schema.getFields().size(); i++) {
            ColumnarDataBuilder column;
            try {
                column = recordBatchBuilder.getColumnBuilder(i);
            } catch (Exception e) {
                System.out.println("Create column builder failed");
                return;
            }
            Field field = schema.getFields().get(i);
            if (field.getType().equals(Arrow.Type.Boolean)) {
                for (int j = 0; j < rowCount; j++) {
                    boolean value;
                    if (objects.get(j).getValues().get(i) instanceof Boolean) {
                        value = (Boolean) objects.get(j).getValues().get(i);
                    } else {
                        value = ((BooleanWritable) objects.get(j).getValues().get(i)).get();
                    }
                    column.setBoolean(j, value);
                }
            } else if (field.getType().equals(Arrow.Type.Int)) {
                for (int j = 0; j < rowCount; j++) {
                    int value;
                    if (objects.get(j).getValues().get(i) instanceof Integer) {
                        value = (Integer) objects.get(j).getValues().get(i);
                    } else {
                        value = ((IntWritable) objects.get(j).getValues().get(i)).get();
                    }
                    column.setInt(j, value);
                }
            } else if (field.getType().equals(Arrow.Type.Int64)) {
                for (int j = 0; j < rowCount; j++) {
                    long value;
                    if (objects.get(j).getValues().get(i) instanceof Long) {
                        value = (Long) objects.get(j).getValues().get(i);
                    } else {
                        value = ((LongWritable) objects.get(j).getValues().get(i)).get();
                    }
                    column.setLong(j, value);
                }
            } else if (field.getType().equals(Arrow.Type.Float)) {
                for (int j = 0; j < rowCount; j++) {
                    float value;
                    if (objects.get(j).getValues().get(i) instanceof Float) {
                        value = (Float) objects.get(j).getValues().get(i);
                    } else {
                        value = ((FloatWritable) objects.get(j).getValues().get(i)).get();
                    }
                    column.setFloat(j, value);
                }
            } else if (field.getType().equals(Arrow.Type.Double)) {
                for (int j = 0; j < rowCount; j++) {
                    double value;
                    if (objects.get(j).getValues().get(i) instanceof Double) {
                        value = (Double) objects.get(j).getValues().get(i);
                    } else {
                        value = ((DoubleWritable) objects.get(j).getValues().get(i)).get();
                    }
                    column.setDouble(j, value);
                }
            } else if (field.getType().equals(Arrow.Type.VarChar)) {
                System.out.println("var char");
                // may be not correct
                for (int j = 0; j < rowCount; j++) {
                    String value;
                    if (objects.get(j).getValues().get(i) instanceof String) {
                        value = (String) objects.get(j).getValues().get(i);
                    } else {
                        value = ((Text) objects.get(j).getValues().get(i)).toString();
                    }
                    column.setUTF8String(j, new org.apache.arrow.vector.util.Text(value));
                }
            } else {
                System.out.println("Type:" + field.getType() + " is not supported");
                // throw new VineyardException.NotImplemented(
                //         "array builder for type " + field.getType() + " is not supported");
            }
        }
    }
}

class MapredRecordWriter<K extends NullWritable, V extends VineyardRowWritable>
        implements RecordWriter<K, V> {
    MapredRecordWriter() throws IOException {
        System.out.printf("creating vineyard record writer\n");
        throw new RuntimeException("mapred record writter: unimplemented");
    }

    @Override
    public void write(K k, V v) throws IOException {
        System.out.printf("write: k = %s, v = %s\n", k, v);
    }

    @Override
    public void close(Reporter reporter) throws IOException {
        System.out.printf("closing\n");
    }
}
