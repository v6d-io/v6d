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
package io.v6d.hadoop.fs;

import com.google.common.base.StopwatchContext;
import io.v6d.core.client.Context;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException.ObjectNotExists;
import io.v6d.hive.ql.io.CloseableReentrantLock;
import io.v6d.modules.basic.arrow.SchemaBuilder;
import io.v6d.modules.basic.arrow.Table;
import io.v6d.modules.basic.arrow.TableBuilder;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import lombok.*;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.util.Progressable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class VineyardOutputStream extends FSDataOutputStream {
    private FileChannel channel;

    public VineyardOutputStream(FileChannel channel) throws IOException {
        super(new DataOutputBuffer(), null);
        this.channel = channel;
    }

    @Override
    public void close() throws IOException {
        this.channel.close();
    }

    @Override
    public String toString() {
        return "vineyard";
    }

    @Override
    public void write(int b) throws IOException {
        throw new UnsupportedOperationException("should not call this function.");
    }

    @Override
    public void write(byte b[], int off, int len) throws IOException {
        channel.write(java.nio.ByteBuffer.wrap(b, off, len));
    }
}

class VineyardInputStream extends FSInputStream {
    private FileChannel channel;

    public VineyardInputStream(FileChannel channel) throws IOException {
        this.channel = channel;
    }

    @Override
    public void seek(long offset) throws IOException {
        throw new UnsupportedOperationException("Vineyard input stream not support seek.");
    }

    @Override
    public long getPos() throws IOException {
        throw new UnsupportedOperationException("Vineyard input stream not support getPos.");
    }

    @Override
    public boolean seekToNewSource(long l) throws IOException {
        throw new UnsupportedOperationException(
                "Vineyard input stream not support seekToNewSource.");
    }

    @Override
    public int read() throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(1);
        int ret = channel.read(buffer);
        if (ret <= 0) {
            return -1;
        }
        return buffer.get(0);
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }
}

public class FileSystem extends org.apache.hadoop.fs.FileSystem {
    public static final String SCHEME = "vineyard";

    private URI uri = URI.create(SCHEME + ":/");
    private static Logger logger = LoggerFactory.getLogger(FileSystem.class);

    static final CloseableReentrantLock lock = new CloseableReentrantLock();
    private Configuration conf = null;

    static RawLocalFileSystem fs = null;
    static boolean enablePrintAllFiles = false;

    Path workingDir = new Path("vineyard:/");

    public FileSystem() {
        super();
    }

    public static void printAllFiles(java.nio.file.Path root, java.nio.file.FileSystem fs)
            throws IOException {
        DirectoryStream<java.nio.file.Path> stream = Files.newDirectoryStream(root);
        Queue<java.nio.file.Path> queue = new java.util.LinkedList<java.nio.file.Path>();
        for (java.nio.file.Path p : stream) {
            queue.add(p);
        }
        while (!queue.isEmpty()) {
            java.nio.file.Path p = queue.poll();
            Context.println(p.toString());
            if (Files.isDirectory(p)) {
                DirectoryStream<java.nio.file.Path> streamTemp = Files.newDirectoryStream(p);
                for (java.nio.file.Path p1 : streamTemp) {
                    queue.add(p1);
                }
                streamTemp.close();
            }
        }
        stream.close();
    }

    public static void printAllFiles(Path p) throws IOException {
        FileStatus[] status = fs.listStatus(p);
        Queue<FileStatus> queue = new java.util.LinkedList<FileStatus>();
        for (FileStatus s : status) {
            queue.add(s);
        }
        while (!queue.isEmpty()) {
            FileStatus p1 = queue.poll();
            Context.println(p1.getPath().toString());
            if (p1.isDirectory()) {
                FileStatus[] statusTemp = fs.listStatus(p1.getPath());
                for (FileStatus s : statusTemp) {
                    queue.add(s);
                }
            }
        }
    }

    private static void printAllFiles() throws IOException {
        if (enablePrintAllFiles) {
            Context.println("------------------");
            printAllFiles(new Path("/opt/hive/data/warehouse"));
        }
    }

    @Override
    public String getScheme() {
        return SCHEME;
    }

    @Override
    public URI getUri() {
        return uri;
    }

    @Override
    public void setXAttr(Path path, String name, byte[] value, EnumSet<XAttrSetFlag> flag)
            throws IOException {}

    @Override
    protected URI canonicalizeUri(URI uri) {
        return uri;
    }

    @Override
    public void initialize(URI name, Configuration conf) throws IOException {
        Context.println("initialize vineyard file system: " + name.toString());
        super.initialize(name, conf);
        this.conf = conf;
        this.uri = name;

        fs = new RawLocalFileSystem();
        fs.initialize(URI.create("file:///"), conf);
        mkdirs(new Path(uri.toString().replaceAll("/*", "/")));
    }

    @Override
    public FSDataInputStream open(Path path, int i) throws IOException {
        Path newPath = new Path(path.toString().replaceAll("vineyard", "file"));
        FSDataInputStream result = fs.open(newPath);
        return result;
    }

    @Override
    public FSDataOutputStream create(
            Path path,
            FsPermission fsPermission,
            boolean overwrite,
            int bufferSize,
            short replication,
            long blockSize,
            Progressable progressable)
            throws IOException {
        try (val lock = this.lock.open()) {
            return createInternal(
                    path,
                    fsPermission,
                    overwrite,
                    bufferSize,
                    replication,
                    blockSize,
                    progressable);
        }
    }

    private FSDataOutputStream createInternal(
            Path path,
            FsPermission fsPermission,
            boolean overwrite,
            int bufferSize,
            short replication,
            long blockSize,
            Progressable progressable)
            throws IOException {
        Path newPath = new Path(path.toString().replaceAll("vineyard", "file"));
        Path parentPath = newPath.getParent();
        try {
            FileStatus parentStatus = fs.getFileStatus(parentPath);
            if (!parentStatus.isDirectory()) {
                throw new IOException("Parent path is not a directory:" + parentPath.toString());
            }
        } catch (FileNotFoundException e) {
            // parent path not exist
            Context.println("Parent dir not exists. Create parent dir first!");
            fs.mkdirs(parentPath);
        }
        printAllFiles();
        FSDataOutputStream result =
                fs.create(
                        newPath,
                        fsPermission,
                        overwrite,
                        bufferSize,
                        replication,
                        blockSize,
                        progressable);
        return result;
    }

    @Override
    public FSDataOutputStream append(Path path, int i, Progressable progressable)
            throws IOException {
        return null;
    }

    @Override
    public boolean delete(Path path, boolean b) throws IOException {
        try (val lock = this.lock.open()) {
            Path newPath = new Path(path.toString().replaceAll("vineyard", "file"));
            return this.deleteInternal(newPath, b);
        }
    }

    private void printAllObjectsWithName() throws IOException {
        IPCClient client = Context.getClient();
        Context.println("print all objects with name");
        Context.println("====================================");
        try {
            Map<String, ObjectID> objects = client.listNames(".*", true, 255);
            for (val object : objects.entrySet()) {
                Context.println(
                        "object name:"
                                + object.getKey()
                                + ", object id:"
                                + object.getValue().value());
            }
        } catch (Exception e) {
            Context.println("Exception: " + e.getMessage());
        }
        Context.println("====================================");
    }

    public void cleanObjectInVineyard(Path filePath) throws IOException {
        IPCClient client = Context.getClient();
        Queue<Path> queue = new java.util.LinkedList<Path>();
        Collection<ObjectID> objectIDs = new ArrayList<ObjectID>();
        queue.add(filePath);
        while (!queue.isEmpty()) {
            try {
                Path path = queue.peek();
                FileStatus fileStatus = fs.getFileStatus(path);
                if (fileStatus.isDirectory()) {
                    FileStatus[] fileStatusArray = fs.listStatus(path);
                    for (FileStatus s : fileStatusArray) {
                        if (s.getPath().toString().compareTo(filePath.toString()) == 0) {
                            continue;
                        }
                        queue.add(s.getPath());
                    }
                }

                String objectName = path.toString().substring(path.toString().indexOf(":") + 1);
                ObjectID objectID = client.getName(objectName);
                objectIDs.add(objectID);
                client.dropName(objectName);
            } catch (FileNotFoundException e) {
                // file not exist
                Context.println("File not exist.");
                continue;
            } catch (ObjectNotExists e) {
                // object not exist
                Context.println("Object not exist.");
                continue;
            } finally {
                queue.poll();
            }
        }
        client.delete(objectIDs, false, false);
        printAllObjectsWithName();
    }

    private boolean deleteInternal(Path path, boolean b) throws IOException {
        cleanObjectInVineyard(path);
        return fs.delete(path, b);
    }

    @Override
    public boolean rename(Path src, Path dst) throws IOException {
        try (val lock = this.lock.open()) {
            val watch = StopwatchContext.create();
            val renamed = this.renameInternal(src, dst);
            Context.println("Filesystem rename uses: " + watch.stop());
            return renamed;
        }
    }

    private void mergeFile(Path src, Path dst) throws IOException {
        FSDataInputStream srcInput = fs.open(src);
        FSDataInputStream dstInput = fs.open(dst);
        byte[] objectIDByteArray = new byte[255];

        int len = srcInput.read(objectIDByteArray);
        String srcObjectIDStr =
                new String(objectIDByteArray, 0, len, StandardCharsets.UTF_8).replaceAll("\n", "");

        objectIDByteArray = new byte[255];
        len = dstInput.read(objectIDByteArray);
        String dstObjectIDStr =
                new String(objectIDByteArray, 0, len, StandardCharsets.UTF_8).replaceAll("\n", "");

        ObjectID mergedTableObjectID = null;
        try {
            IPCClient client = Context.getClient();
            ObjectID srcObjectID = ObjectID.fromString(srcObjectIDStr);
            ObjectID dstObjectID = ObjectID.fromString(dstObjectIDStr);
            Table srcTable =
                    (Table) ObjectFactory.getFactory().resolve(client.getMetaData(srcObjectID));
            Table dstTable =
                    (Table) ObjectFactory.getFactory().resolve(client.getMetaData(dstObjectID));

            // merge table
            Schema schema = srcTable.getSchema().getSchema();
            SchemaBuilder mergedSchemaBuilder = SchemaBuilder.fromSchema(schema);
            TableBuilder mergedTableBuilder = new TableBuilder(client, mergedSchemaBuilder);

            for (int i = 0; i < srcTable.getBatches().size(); i++) {
                mergedTableBuilder.addBatch(srcTable.getBatches().get(i));
            }

            for (int i = 0; i < dstTable.getBatches().size(); i++) {
                mergedTableBuilder.addBatch(dstTable.getBatches().get(i));
            }

            ObjectMeta meta = mergedTableBuilder.seal(client);
            Context.println("Table id in vineyard:" + meta.getId().value());
            client.persist(meta.getId());
            Context.println("Table persisted, name:" + dst);
            client.putName(meta.getId(), dst.toString().substring(dst.toString().indexOf(":") + 1));
            client.dropName(src.toString());
            mergedTableObjectID = meta.getId();

            // drop old table
            Collection<ObjectID> ids = new ArrayList<ObjectID>();
            ids.add(srcObjectID);
            ids.add(dstObjectID);
            client.delete(ids, false, false);
        } finally {
            srcInput.close();
            dstInput.close();
            if (mergedTableObjectID != null) {
                FSDataOutputStream out = fs.create(dst);
                String mergedTableIDStr = mergedTableObjectID.toString() + "\n";
                out.write((mergedTableIDStr + "\n").getBytes(StandardCharsets.UTF_8));
                out.close();
            }
        }
    }

    public boolean renameInternal(Path src, Path dst) throws IOException {

        Path newSrc = new Path(src.toString().replaceAll("vineyard", "file"));
        Path newDst = new Path(dst.toString().replaceAll("vineyard", "file"));
        String newTableName =
                dst.toString().substring(dst.toString().indexOf(":") + 1).replaceAll("/*", "/");
        String oldTableName =
                src.toString().substring(src.toString().indexOf(":") + 1).replaceAll("/*", "/");
        try {
            FileStatus srcStatus = fs.getFileStatus(newSrc);
            if (srcStatus.isDirectory()) {
                FileStatus[] status = fs.listStatus(newSrc);
                for (FileStatus s : status) {
                    renameInternal(
                            s.getPath(), new Path(newDst.toString() + "/" + s.getPath().getName()));
                }
                fs.delete(newSrc, true);
                return true;
            } else {
                try {
                    FileStatus dstStatus = fs.getFileStatus(newDst);
                } catch (FileNotFoundException e) {
                    // dst file not exist
                    fs.rename(newSrc, newDst);
                    FSDataInputStream in = fs.open(newDst);
                    byte[] objectIDByteArray = new byte[255];
                    int len = in.read(objectIDByteArray);
                    if (len > 0) {
                        String objectIDStr =
                                new String(objectIDByteArray, 0, len, StandardCharsets.UTF_8)
                                        .replaceAll("\n", "");
                        IPCClient client = Context.getClient();
                        try {
                            client.putName(ObjectID.fromString(objectIDStr), newTableName);
                            client.dropName(oldTableName);
                        } catch (Exception e1) {
                            // Skip some invalid file.
                            // File content may be not a valid object id.
                            Context.println("Failed to put name to vineyard: " + e1.getMessage());
                        }
                    }
                    printAllFiles();
                    return true;
                }
                // dst file exist
                mergeFile(newSrc, newDst);
                deleteInternal(newSrc, true);

                printAllFiles();
                return true;
            }
        } catch (FileNotFoundException e) {
            // src file not exist
            Context.println("src file not exist");
            return false;
        }
    }

    public void syncWithVineyard(String prefix) throws IOException {
        IPCClient client = Context.getClient();
        System.out.println("sync with vineyard: " + prefix);
        try {
            String reg = "^" + prefix + ".*";
            Map<String, ObjectID> objects = client.listNames(reg, true, 255);
            for (val object : objects.entrySet()) {
                try {
                    fs.getFileStatus(new Path("file://" + object.getKey()));
                } catch (FileNotFoundException e) {
                    // file not exist
                    Path path = new Path("file://" + object.getKey());
                    FSDataOutputStream out = fs.create(path);
                    ObjectID id = object.getValue();
                    out.write((id.toString() + "\n").getBytes(StandardCharsets.UTF_8));
                    out.close();
                }
            }
        } catch (Exception e) {
            Context.println("Exception: " + e.getMessage());
        }
    }

    @Override
    public FileStatus[] listStatus(Path path) throws FileNotFoundException, IOException {
        List<FileStatus> result = new ArrayList<FileStatus>();
        try (val lock = this.lock.open()) {
            String prefix =
                    path.toString()
                            .substring(path.toString().indexOf(":") + 1)
                            .replaceAll("/*", "/");
            syncWithVineyard(prefix);
            try {
                FileStatus status =
                        fs.getFileStatus(new Path(path.toString().replaceAll("vineyard", "file")));
                FileStatus[] statusArray =
                        fs.listStatus(new Path(path.toString().replaceAll("vineyard", "file")));
                for (FileStatus s : statusArray) {
                    FileStatus temp =
                            new FileStatus(
                                    s.getLen(),
                                    s.isDirectory(),
                                    s.getReplication(),
                                    s.getBlockSize(),
                                    s.getModificationTime(),
                                    s.getAccessTime(),
                                    new FsPermission((short) 777),
                                    s.getOwner(),
                                    s.getGroup(),
                                    new Path(
                                            SCHEME
                                                    + ":///"
                                                    + s.getPath()
                                                            .toString()
                                                            .substring(
                                                                    s.getPath()
                                                                                    .toString()
                                                                                    .indexOf(":")
                                                                            + 1)
                                                            .replaceAll("/*", "/")));
                    result.add(temp);
                }
            } catch (FileNotFoundException e) {
                // file not exist
                return new FileStatus[0];
            }
        }
        printAllFiles();
        return result.toArray(new FileStatus[result.size()]);
    }

    @Override
    public void setWorkingDirectory(Path path) {
        workingDir = path;
    }

    @Override
    public Path getWorkingDirectory() {
        return workingDir;
    }

    @Override
    public boolean mkdirs(Path path, FsPermission fsPermission) throws IOException {
        try (val lock = this.lock.open()) {
            return this.mkdirsInternal(path, fsPermission);
        }
    }

    private boolean mkdirsInternal(Path path, FsPermission fsPermission) throws IOException {

        Path newPath = new Path(path.toString().replaceAll("vineyard", "file"));
        try {
            fs.getFileStatus(newPath);
        } catch (FileNotFoundException e) {
            // file not exist
            boolean result = fs.mkdirs(newPath);
            printAllFiles();
            return result;
        }
        return false;
    }

    @Override
    public FileStatus getFileStatus(Path path) throws IOException {
        try (val lock = this.lock.open()) {
            return this.getFileStatusInternal(path);
        }
    }

    public FileStatus getFileStatusInternal(Path path) throws IOException {
        printAllFiles();
        FileStatus temp =
                fs.getFileStatus(new Path(path.toString().replaceAll("vineyard", "file")));
        FileStatus result =
                new FileStatus(
                        temp.getLen(),
                        temp.isDirectory(),
                        temp.getReplication(),
                        temp.getBlockSize(),
                        temp.getModificationTime(),
                        temp.getAccessTime(),
                        new FsPermission((short) 777),
                        temp.getOwner(),
                        temp.getGroup(),
                        new Path(
                                SCHEME
                                        + ":///"
                                        + temp.getPath()
                                                .toString()
                                                .substring(
                                                        temp.getPath().toString().indexOf(":") + 1)
                                                .replaceAll("/*", "/")));
        return result;
    }

    @Override
    public byte[] getXAttr(Path path, String name) throws IOException {
        throw new UnsupportedOperationException("Vineyard file system not support getXAttr.");
    }

    @Override
    public void moveFromLocalFile(Path src, Path dst) throws IOException {
        throw new UnsupportedOperationException(
                "Vineyard file system not support moveFromLocalFile.");
    }

    @Override
    public void copyFromLocalFile(boolean delSrc, Path src, Path dst) throws IOException {
        Context.println(
                "copyFromLocalFile: "
                        + src.toString()
                        + " to "
                        + dst.toString()
                        + " delSrc: "
                        + delSrc);
        throw new UnsupportedOperationException(
                "Vineyard file system not support copyFromLocalFile.");
    }

    @Override
    public void copyFromLocalFile(boolean delSrc, boolean overwrite, Path[] srcs, Path dst)
            throws IOException {
        throw new UnsupportedOperationException(
                "Vineyard file system not support copyFromLocalFile.");
    }

    @Override
    public void copyFromLocalFile(boolean delSrc, boolean overwrite, Path src, Path dst)
            throws IOException {
        throw new UnsupportedOperationException(
                "Vineyard file system not support copyFromLocalFile.");
    }

    @Override
    public void copyToLocalFile(Path src, Path dst) throws IOException {
        throw new UnsupportedOperationException(
                "Vineyard file system not support copyToLocalFile.");
    }

    @Override
    public void moveToLocalFile(Path src, Path dst) throws IOException {
        throw new UnsupportedOperationException(
                "Vineyard file system not support moveToLocalFile.");
    }

    @Override
    public void copyToLocalFile(boolean delSrc, Path src, Path dst) throws IOException {
        throw new UnsupportedOperationException(
                "Vineyard file system not support copyToLocalFile.");
    }

    @Override
    public void copyToLocalFile(boolean delSrc, Path src, Path dst, boolean useRawLocalFileSystem)
            throws IOException {
        throw new UnsupportedOperationException(
                "Vineyard file system not support copyToLocalFile.");
    }
}
