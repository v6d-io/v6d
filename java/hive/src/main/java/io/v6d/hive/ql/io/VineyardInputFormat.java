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

import org.apache.hadoop.hive.ql.io.HiveInputFormat;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.Reporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.LineRecordReader;
import io.v6d.modules.basic.dataframe.DataFrame;
import org.apache.hadoop.mapred.FileSplit;
import java.io.IOException;
// how to connect vineyard

// check phase:
// 1. check if vineyard is running
// 2. check if file exist in vineyard
// 3. if exist, read file from vineyard
// 4. if not exist, read file from hdfs and write to vineyard

public class VineyardInputFormat extends FileInputFormat<LongWritable, Text> {
    
    @Override
    public RecordReader<LongWritable, Text> getRecordReader(InputSplit genericSplit, JobConf job, Reporter reporter) throws IOException {
        reporter.setStatus(genericSplit.toString());
        System.out.printf("--------+creating vineyard record reader\n");
        return new MapredrecordReader<LongWritable, Text>(job, (FileSplit) genericSplit);
    }

}

class MapredrecordReader<K extends LongWritable, V extends Text>
        implements RecordReader<LongWritable, Text> {
    private static Logger logger = LoggerFactory.getLogger(MapredrecordReader.class);
    private LineRecordReader reader;
    private LongWritable lineKey = null;
	private Text lineValue = null;

    MapredrecordReader(JobConf job, FileSplit split) throws IOException {
        System.out.printf("--------+creating vineyard record reader\n");
        // throw new RuntimeException("mapred record reader: unimplemented");
        reader = new LineRecordReader(job, split);
        lineKey = reader.createKey();
		lineValue = reader.createValue();
    }

    @Override
    public void close() throws IOException {
        System.out.printf("--------closing\n");
        reader.close();
    }

    @Override
    public LongWritable createKey() {
        System.out.printf("--------creating key\n");
        return new LongWritable(0);
    }

    @Override
    public Text createValue() {
        System.out.printf("++++++++creating value\n");
        return new Text("");
    }

    @Override
    public long getPos() throws IOException {
        System.out.printf("+++++++get pos\n");
        return 0;
    }

    @Override
    public float getProgress() throws IOException {
        System.out.printf("++++++++get progress\n");
        return reader.getProgress();
    }

    @Override
    public boolean next(LongWritable key, Text value) throws IOException {
        System.out.printf("+++++++++next\n");
        while(true) {
            if(!reader.next(lineKey, lineValue)) {
                break;
            } else {
                // value.set(lineValue.toString());
                // key.set(key.get() + 1);

                String strReplace = lineValue.toString().replaceAll(",", "\001");
                Text txtReplace = new Text();
                txtReplace.set(strReplace);
                value.set(txtReplace.getBytes(), 0, txtReplace.getLength());

                return true;
            }
        }
        
        return false;
    }
}
// how to get data from vineyard ?????
