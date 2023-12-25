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

import com.google.common.base.Stopwatch;
import com.google.common.base.StopwatchContext;
import io.v6d.core.client.Context;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.arrow.*;
import io.v6d.modules.basic.arrow.util.ObjectTransformer;
import io.v6d.modules.basic.columnar.ColumnarDataBuilder;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import lombok.*;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.Types;
import org.apache.arrow.vector.types.pojo.*;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.commons.lang.NotImplementedException;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.hive.ql.exec.FileSinkOperator;
import org.apache.hadoop.hive.ql.io.HiveOutputFormat;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.typeinfo.DecimalTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.ListTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.MapTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.PrimitiveTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.StructTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Progressable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VineyardOutputFormat<K extends NullWritable, V extends RecordWrapperWritable>
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
    private static Logger logger = LoggerFactory.getLogger(SinkRecordWriter.class);

    static {
        Arrow.instantiate();
    }

    // FIXME: read from configuration
    public static final int RECORD_BATCH_SIZE = 256000;

    private JobConf jc;
    private Path finalOutPath;
    private FileSystem fs;
    private Progressable progress;
    private final ObjectTransformer transformer;

    // vineyard
    private FSDataOutputStream output;
    private static CloseableReentrantLock lock = new CloseableReentrantLock();
    private Schema schema;
    private TypeInfo[] infos;

    List<RecordBatchBuilder> chunks = new ArrayList<>();
    List<ColumnarDataBuilder> current;
    private int currentLoc = RECORD_BATCH_SIZE;

    // profiling
    private Stopwatch writeTimer = StopwatchContext.createUnstarted();

    public static final PathFilter VINEYARD_FILES_PATH_FILTER =
            new PathFilter() {
                @Override
                public boolean accept(Path p) {
                    String name = p.getName();
                    return name.startsWith("_task_tmp.")
                            && (!name.substring(10, name.length()).startsWith("-"));
                }
            };

    public SinkRecordWriter(
            JobConf jc,
            Path finalOutPath,
            Class<? extends Writable> valueClass,
            boolean isCompressed,
            Properties tableProperties,
            Progressable progress)
            throws IOException {
        val watch = StopwatchContext.create();
        this.jc = jc;
        if (!RecordWrapperWritable.class.isAssignableFrom(valueClass)) {
            throw new VineyardException.Invalid(
                    "value class must be RecordWrapperWritable, while it is " + valueClass);
        }
        if (isCompressed) {
            throw new VineyardException.Invalid("compressed output is not supported");
        }
        this.finalOutPath = finalOutPath;
        this.progress = progress;

        // initialize the schema
        this.initializeTableFile();
        this.schema = this.initializeTableSchema(tableProperties);

        this.transformer = new HiveTypeTransformer();
        Context.println("creating a sink record writer uses: " + watch.stop());
    }

    @Override
    public void write(Writable w) throws IOException {
        if (w == null) {
            Context.println("warning: null writable, skipping ...");
            return;
        }

        writeTimer.start();
        if (currentLoc == RECORD_BATCH_SIZE) {
            val builder =
                    new RecordBatchBuilder(
                            Context.getClient(), schema, RECORD_BATCH_SIZE, this.transformer);
            chunks.add(builder);
            current = builder.getColumnBuilders();
            currentLoc = 0;
        }

        val record = (RecordWrapperWritable) w;
        val values = record.getValues();
        VineyardException.asserts(
                values.length == current.size(),
                "The length of record doesn't match with the length of builders");
        for (int i = 0; i < values.length; ++i) {
            current.get(i).setObject(currentLoc, values[i]);
        }
        currentLoc += 1;
        writeTimer.stop();
    }

    // check if the table is already created.
    // if not, create a new table.
    // if yes, append the data to the table.(Get from vineyard, and seal it in a new table)
    @SneakyThrows(VineyardException.class)
    @Override
    public void close(boolean abort) throws IOException {
        Context.println(
                "closing SinkRecordWriter: chunks size:"
                        + chunks.size()
                        + ", write uses "
                        + writeTimer);

        // construct table from existing record batch builders
        val watch = StopwatchContext.create();

        val client = Context.getClient();
        val schemaBuilder = SchemaBuilder.fromSchema(schema);
        Context.println("create schema builder use " + watch);
        val tableBuilder = new TableBuilder(client, schemaBuilder);
        Context.println("create schema & table builder use " + watch);
        if (chunks.size() > 0) {
            chunks.get(chunks.size() - 1).shrink(client, currentLoc);
        }
        for (int i = 0; i < chunks.size(); i++) {
            val chunk = chunks.get(i);
            tableBuilder.addBatch(chunk);
        }
        ObjectMeta meta = tableBuilder.seal(client);
        Context.println("Table id in vineyard:" + meta.getId().value());
        client.persist(meta.getId());

        output.write((meta.getId().toString() + "\n").getBytes(StandardCharsets.US_ASCII));
        output.close();
    }

    private void initializeTableFile() throws IOException {
        fs = finalOutPath.getFileSystem(jc);
        this.output = FileSystem.create(fs, finalOutPath, new FsPermission((short) 0777));
        if (output == null) {
            throw new VineyardException.Invalid("Create table file failed.");
        }
    }

    private Field getField(TypeInfo typeInfo) {
        Field field = null;
        switch (typeInfo.getCategory()) {
            case LIST:
                List<Field> listChildren = new ArrayList<>();
                Field chField = getField(((ListTypeInfo) typeInfo).getListElementTypeInfo());
                listChildren.add(chField);
                field =
                        new Field(
                                typeInfo.getTypeName(),
                                FieldType.nullable(toArrowType(typeInfo)),
                                listChildren);
                break;
            case STRUCT:
                List<Field> structChildren = new ArrayList<>();
                for (val child : ((StructTypeInfo) typeInfo).getAllStructFieldTypeInfos()) {
                    structChildren.add(getField(child));
                }
                field =
                        new Field(
                                typeInfo.getTypeName(),
                                FieldType.nullable(toArrowType(typeInfo)),
                                structChildren);
                break;
            case MAP:
                listChildren = new ArrayList<>();
                structChildren = new ArrayList<>();
                List<Field> mapChildren = new ArrayList<>();
                Field keyField = getField(((MapTypeInfo) typeInfo).getMapKeyTypeInfo());
                Field valueField = getField(((MapTypeInfo) typeInfo).getMapValueTypeInfo());
                structChildren.add(keyField);
                structChildren.add(valueField);
                Field structField =
                        new Field(
                                typeInfo.getTypeName(),
                                FieldType.notNullable(ArrowType.Struct.INSTANCE),
                                structChildren);
                listChildren.add(structField);
                Field listField =
                        new Field(
                                typeInfo.getTypeName(),
                                FieldType.notNullable(ArrowType.List.INSTANCE),
                                listChildren);
                mapChildren.add(listField);
                field =
                        new Field(
                                typeInfo.getTypeName(),
                                FieldType.nullable(toArrowType(typeInfo)),
                                mapChildren);
                break;
            case UNION:
                throw new NotImplementedException();
            default:
                field = Field.nullable(typeInfo.getTypeName(), toArrowType(typeInfo));
                break;
        }
        return field;
    }

    private Schema initializeTableSchema(Properties tableProperties) {
        val structTypeInfo = TypeContext.computeStructTypeInfo(tableProperties);
        val targetTypeInfos =
                TypeContext.computeTargetTypeInfos(
                        structTypeInfo, ObjectInspectorUtils.ObjectInspectorCopyOption.WRITABLE);

        List<Field> fields = new ArrayList<>();
        for (val typeInfo : targetTypeInfos) {
            fields.add(getField(typeInfo));
        }
        infos = targetTypeInfos;
        return new Schema(fields);
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
                    case CHAR:
                    case VARCHAR:
                        return Types.MinorType.LARGEVARCHAR.getType();
                    case DATE:
                        return Types.MinorType.DATEDAY.getType();
                    case TIMESTAMP:
                        return new ArrowType.Timestamp(TimeUnit.NANOSECOND, "UTC");
                    case BINARY:
                        return Types.MinorType.VARBINARY.getType();
                    case DECIMAL:
                        final DecimalTypeInfo decimalTypeInfo = (DecimalTypeInfo) typeInfo;
                        return new ArrowType.Decimal(
                                decimalTypeInfo.precision(), decimalTypeInfo.scale(), 128);
                    case INTERVAL_YEAR_MONTH:
                        return Types.MinorType.INTERVALYEAR.getType();
                    case INTERVAL_DAY_TIME:
                        return Types.MinorType.INTERVALDAY.getType();
                    case VOID:
                    case UNKNOWN:
                    default:
                        throw new IllegalArgumentException();
                }
            case LIST:
                return ArrowType.List.INSTANCE;
            case STRUCT:
                return ArrowType.Struct.INSTANCE;
            case MAP:
                return new ArrowType.Map(true);
            case UNION:
            default:
                throw new IllegalArgumentException();
        }
    }
}

class MapredRecordWriter<K extends NullWritable, V extends RecordWrapperWritable>
        implements RecordWriter<K, V> {
    MapredRecordWriter() throws IOException {
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
