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
import io.v6d.modules.basic.dataframe.DataFrameBuilder;
import io.v6d.modules.basic.tensor.TensorBuilder;
import io.v6d.core.client.ds.ObjectMeta;

import java.io.IOException;
import java.util.Properties;
import java.util.ArrayList;
import java.util.List;

import org.apache.arrow.vector.VectorSchemaRoot;
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
            // TBD: get vineyard socket path from table properties
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

        try {
            dataFrameBuilder = new DataFrameBuilder(client);
        } catch (Exception e) {
            throw new IOException("Create DataFrameBuilder failed");
        }
        // Create Tensors
        for (int i = 0; i < schema.getFields().size(); i++) {
            List<Integer> shape = new ArrayList<Integer>(1);
            shape.add(root.getRowCount());
            try {
                tensorBuilder = new TensorBuilder(client, shape, root.getFieldVectors().get(i));
                dataFrameBuilder.addColumn(schema.getFields().get(i).getName(), tensorBuilder);
            } catch (Exception e) {
                throw new IOException("Create TensorBuilder failed");
            }
        }
    }

    @Override
    public void close(boolean abort) throws IOException {
        System.out.println("vineyard filesink operator closing");
        try {
            ObjectMeta meta = dataFrameBuilder.seal(client);
            System.out.println("DataFrame id:" + meta.getId().value());
        } catch (Exception e) {
            throw new IOException("Seal DataFrame failed");
        }
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
