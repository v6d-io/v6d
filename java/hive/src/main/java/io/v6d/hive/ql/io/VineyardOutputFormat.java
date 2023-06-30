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

import io.v6d.core.common.util.VineyardException;
import io.v6d.core.client.IPCClient;
import io.v6d.modules.basic.arrow.*;
import io.v6d.modules.basic.dataframe.DataFrameBuilder;
import io.v6d.modules.basic.tensor.TensorBuilder;
import io.v6d.modules.basic.tensor.ITensor;
import io.v6d.modules.basic.arrow.BufferBuilder;
import io.v6d.core.client.ds.ObjectMeta;

import java.io.IOException;
import java.util.Properties;
import java.util.ArrayList;
import java.util.List;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.NonNullableStructVector;
import org.apache.arrow.vector.types.pojo.ArrowType.ArrowTypeID;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hive.ql.exec.FileSinkOperator;
import org.apache.hadoop.hive.ql.io.HiveOutputFormat;
import org.apache.hadoop.hive.ql.io.arrow.ArrowWrapperWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Progressable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VineyardOutputFormat<K extends NullWritable, V extends ArrowWrapperWritable>
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
    private Properties tableProperties;
    private Progressable progress;

    // vineyard
    private static IPCClient client;
    private DataFrameBuilder dataFrameBuilder;
    private TensorBuilder tensorBuilder;

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

        System.out.printf("final out path: %s\n", finalOutPath);
        // connect to vineyard
        if (client == null) {
            client = new IPCClient("/tmp/vineyard.sock");
        }
        if (client == null || !client.connected()) {
            throw new VineyardException.Invalid("failed to connect to vineyard");
        } else {
            System.out.printf("connected to vineyard succeed!\n");
        }
        
    }

    @Override
    public void write(Writable w) throws IOException {
        System.out.printf("vineard filesink record writer: %s, %s\n", w, w.getClass());
        VectorSchemaRoot root = ((ArrowWrapperWritable) w).getVectorSchemaRoot();
        org.apache.arrow.vector.types.pojo.Schema schema = root.getSchema();

        // why?
        // System.out.printf("row count: %d\n", root.getRowCount());
        // for (int i = 0; i < root.getRowCount(); ++i) {
        //     System.out.printf("row %d: ", i);
        //     for (int j = 0; j < structVector.getChildrenFromFields().size(); ++j) {
        //         System.out.printf("%s ", structVector.getChildrenFromFields().get(j).getObject(i));
        //     }
        //     System.out.printf("\n");
        // }
        // System.out.println("==============");
        // for (int i = 0; i < structVector.getChildrenFromFields().size(); ++i) {
        //     System.out.printf("field %d: %s\n", i, structVector.getChildrenFromFields().get(i).getName());
        // }
        System.out.println("==============");

        try{
            BufferBuilder builder = new BufferBuilder(client, 100);
            ObjectMeta meta =  builder.seal(client);
            System.out.printf("buffer id: %d\n", meta.getId());
        } catch (Exception e) {
            System.out.printf("failed to seal buffer: %s\n", e);
        }

        for (int i = 0; i < root.getRowCount(); i++) {
            System.out.printf("row %d: ", i);
            for (int j = 0; j < root.getFieldVectors().size(); ++j) {
                System.out.printf("%s ", root.getFieldVectors().get(j).getObject(i));
            }
            System.out.printf("\n");
        }
        System.out.println("==============");
        for (int j = 0; j < schema.getFields().size(); ++j) {
            System.out.printf(schema.getFields().get(j).getName() + " ");
        }
        for (int j = 0; j < schema.getFields().size(); ++j) {
            System.out.printf(schema.getFields().get(j).getName() + " ");
        }
        System.out.printf("\n");

        dataFrameBuilder = new DataFrameBuilder(client);
        // Create Tensors
        for (int i = 0; i < schema.getFields().size(); i++) {
            ArrowTypeID arrowTypeID = schema.getFields().get(i).getType().getTypeID();
            switch(arrowTypeID) {
                // TODO: other type
                case Int:
                    System.out.printf("int\n");
                    List<Integer> shape = new ArrayList<Integer>(1);
                    shape.add(root.getRowCount());
                    tensorBuilder = new TensorBuilder(client, shape, root.getFieldVectors().get(i));
                    // tensorBuilder.setValues(root.getFieldVectors().get(i));
                    // tensorBuilder = new TensorBuilder(root.getFieldVectors().get(i));
                    // column
                    dataFrameBuilder.addColumn(schema.getFields().get(i).getName(), tensorBuilder);
                    break;
                default:
                    System.out.printf("unsupported arrow type: %s\n", arrowTypeID);
                    break;
            }
        }
    }

    @Override
    public void close(boolean abort) throws IOException {
        System.out.println("vineyard filesink operator closing\n");
        dataFrameBuilder.seal(client);
    }
}

class MapredRecordWriter<K extends NullWritable, V extends ArrowWrapperWritable>
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
