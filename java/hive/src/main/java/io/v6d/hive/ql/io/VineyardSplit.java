package io.v6d.hive.ql.io;

import java.io.DataOutput;
import java.io.DataInput;
import java.io.IOException;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.fs.Path;

public class VineyardSplit extends FileSplit {
    String customPath;

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
        return super.getPath();
    }

    @Override
    public long getLength() {
        return 0;
    }

    @Override
    public String[] getLocations() throws IOException {
        return new String[0];
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        super.readFields(in);
    }

    @Override
    public void write(DataOutput out) throws IOException {
        super.write(out);
    }
}