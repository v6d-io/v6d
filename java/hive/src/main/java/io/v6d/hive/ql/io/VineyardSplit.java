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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.*;

public class VineyardSplit extends FileSplit {
    String customPath;
    int batchStartIndex;
    int batchSize;

    protected VineyardSplit() {
        super();
    }

    public VineyardSplit(Path file, long start, long length, JobConf conf) {
        super(file, start, length, (String[]) null);
    }

    public VineyardSplit(Path file, long start, long length, String[] hosts) {
        super(file, start, length, hosts);
    }

    @Override
    public Path getPath() {
        System.out.println("getPath");
        return super.getPath();
    }

    @Override
    public long getStart() {
        return super.getStart();
    }

    @Override
    public long getLength() {
        return super.getLength();
    }

    @Override
    public String[] getLocations() throws IOException {
        System.out.println("getLocations");
        return new String[0];
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        System.out.println("readFields");
        super.readFields(in);
        batchStartIndex = in.readInt();
        batchSize = in.readInt();
    }

    @Override
    public void write(DataOutput out) throws IOException {
        System.out.println("write");
        super.write(out);
        out.writeInt(batchStartIndex);
        out.writeInt(batchSize);
    }

    public void setBatch(int batchStartIndex, int batchSize) {
        this.batchStartIndex = batchStartIndex;
        this.batchSize = batchSize;
    }

    public int getBatchStartIndex() {
        return batchStartIndex;
    }

    public int getBatchSize() {
        return batchSize;
    }
}
