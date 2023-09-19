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
import com.google.common.jimfs.Jimfs;
import io.v6d.core.client.Context;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.ObjectID;
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
import java.nio.file.StandardOpenOption;
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
import org.apache.hadoop.hive.conf.HiveConf;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.util.Progressable;
import org.apache.hadoop.yarn.server.resourcemanager.scheduler.fair.FSParentQueue;
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

    // static java.nio.file.FileSystem jimfs = null;
    static RawLocalFileSystem fs = null;
    static boolean enablePrintAllFiles = true;

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

    public static void printAllFiles(Path p)
            throws IOException {
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
            // printAllFiles(jimfs.getPath("/"), jimfs);
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
        // try {
        //     if (jimfs == null) {
        //         jimfs = Jimfs.newFileSystem(com.google.common.jimfs.Configuration.unix());
        //     }
        // } catch (Exception e) {
        //     Context.println("Exception: " + e.getMessage());
        //     throw e;
        // }
        fs = new RawLocalFileSystem();
        fs.initialize(URI.create("file:///"), conf);
        mkdirs(new Path(uri.toString().replaceAll("///", "/")));
    }

    @Override
    public FSDataInputStream open(Path path, int i) throws IOException {
        Context.println("open file: " + path.toString());
        // FileChannel channel =
        //         FileChannel.open(
        //                 jimfs.getPath(path.toString().substring(path.toString().indexOf(":") + 1)),
        //                 StandardOpenOption.READ);
        Path newPath = new Path(path.toString().replaceAll("vineyard", "file"));
        FSDataInputStream result = fs.open(newPath);
        return result;
        // return new FSDataInputStream(new VineyardInputStream(channel));
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
        Context.println("create file: " + path.toString());
        // java.nio.file.Path nioFilePath =
        //         jimfs.getPath(path.toString().substring(path.toString().indexOf(":") + 1));
        // java.nio.file.Path nioParentDirPath = nioFilePath.getParent();
        // if (nioParentDirPath != null) {
        //     Files.createDirectories(nioParentDirPath);
        // }
        // Files.createFile(nioFilePath);
        // FileChannel channel = FileChannel.open(nioFilePath, StandardOpenOption.WRITE);
        // printAllFiles();
        // return new FSDataOutputStream(new VineyardOutputStream(channel), null);
        Path newPath = new Path(path.toString().replaceAll("vineyard", "file"));
        Context.println("new file:" + newPath.toString());
        Path parentPath = newPath.getParent();
        Context.println("parent file:" + parentPath.toString());
        try {
            FileStatus parentStatus = fs.getFileStatus(parentPath);
            if (!parentStatus.isDirectory()) {
                throw new IOException("Parent path is not a directory:" + parentPath.toString());
            }
        } catch (FileNotFoundException e) {
            // parent path not exist
            Context.println("create parent dir");
            fs.mkdirs(parentPath);
        }
        printAllFiles();
        FSDataOutputStream result = fs.create(newPath, fsPermission, overwrite, bufferSize, replication, blockSize, progressable);
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
            // return this.deleteInternal(
            //         jimfs.getPath(path.toString().substring(path.toString().indexOf(":") + 1)), b);
            Path newPath = new Path(path.toString().replaceAll("vineyard", "file"));
            return this.deleteInternal(newPath, b);
        }
    }

    private boolean deleteInternal(Path path, boolean b) throws IOException {
        return fs.delete(path, b);
    }

    // private boolean deleteInternal(java.nio.file.Path path, boolean b) throws IOException {
    //     Context.println("delete file: " + path.toString());
    //     java.nio.file.Path nioFilePath =
    //             jimfs.getPath(path.toString().substring(path.toString().indexOf(":") + 1));

    //     // check if the path is a directory
    //     if (Files.isDirectory(nioFilePath)) {
    //         DirectoryStream<java.nio.file.Path> stream = Files.newDirectoryStream(nioFilePath);
    //         for (java.nio.file.Path p : stream) {
    //             deleteInternal(p, b);
    //         }
    //         stream.close();
    //     } else {
    //         // drop name
    //         String name = nioFilePath.getFileName().toString();
    //         IPCClient client = Context.getClient();
    //         try {
    //             client.dropName(name);
    //         } catch (Exception e) {
    //             Context.println("Failed to drop name from vineyard: " + e.getMessage());
    //         }
    //     }
    //     Files.deleteIfExists(nioFilePath);

    //     printAllFiles();
    //     return true;
    // }

    @Override
    public boolean rename(Path src, Path dst) throws IOException {
        try (val lock = this.lock.open()) {
            val watch = StopwatchContext.create();
            val renamed = this.renameInternal(src, dst);
            Context.println("filesystem rename uses: " + watch.stop());
            return renamed;
        }
    }

    private void mergeFile(java.nio.file.Path src, java.nio.file.Path dst) throws IOException {
        FileChannel channelSrc = FileChannel.open(src, StandardOpenOption.READ);
        FileChannel channelDst = FileChannel.open(dst, StandardOpenOption.READ);

        ByteBuffer bytes = ByteBuffer.allocate(255);
        int len = channelSrc.read(bytes);
        String srcObjectIDStr =
                new String(bytes.array(), 0, len, StandardCharsets.UTF_8).replaceAll("\n", "");
        bytes = ByteBuffer.allocate(255);
        len = channelDst.read(bytes);
        String dstObjectIDStr =
                new String(bytes.array(), 0, len, StandardCharsets.UTF_8).replaceAll("\n", "");

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
            Context.println("record batch size:" + mergedTableBuilder.getBatchSize());
            Context.println("Table id in vineyard:" + meta.getId().value());
            client.persist(meta.getId());
            Context.println("Table persisted, name:" + dst);
            client.putName(meta.getId(), dst.toString());
            client.dropName(src.toString());
            mergedTableObjectID = meta.getId();

            // drop old table
            Collection<ObjectID> ids = new ArrayList<ObjectID>();
            ids.add(srcObjectID);
            ids.add(dstObjectID);
            client.delete(ids, false, false);
        } finally {
            channelSrc.close();
            channelDst.close();
            if (mergedTableObjectID != null) {
                channelDst = FileChannel.open(dst, StandardOpenOption.WRITE);
                String mergedTableIDStr = mergedTableObjectID.toString() + "\n";
                bytes =
                        ByteBuffer.allocate(
                                mergedTableIDStr.getBytes(StandardCharsets.UTF_8).length);
                channelDst.write(
                        ByteBuffer.wrap(mergedTableIDStr.getBytes(StandardCharsets.UTF_8)));
            }
        }
    }

    private void mergeFile(Path src, Path dst) throws IOException {
        Context.println("merge file: " + src.toString() + " to " + dst.toString());
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
            Context.println("src object id:" + srcObjectID.value());
            ObjectID dstObjectID = ObjectID.fromString(dstObjectIDStr);
            Context.println("dst object id:" + dstObjectID.value());
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
            Context.println("record batch size:" + mergedTableBuilder.getBatchSize());
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
        Context.println("rename file: " + src.toString() + " to " + dst.toString());
        // now we create new file and delete old file to simulate rename
        // java.nio.file.Path srcNioFilePath =
        //         jimfs.getPath(src.toString().substring(src.toString().indexOf(":") + 1));
        // java.nio.file.Path dstNioFilePath =
        //         jimfs.getPath(dst.toString().substring(dst.toString().indexOf(":") + 1));
        // java.nio.file.Path dstNioParentDirPath = dstNioFilePath.getParent();
        // Files.createDirectories(dstNioParentDirPath);

        // if (Files.isDirectory(srcNioFilePath)) {
        //     DirectoryStream<java.nio.file.Path> stream = Files.newDirectoryStream(srcNioFilePath);
        //     for (java.nio.file.Path p : stream) {
        //         renameInternal(
        //                 new Path(SCHEME + ":/" + p.toString()),
        //                 new Path(
        //                         SCHEME + ":/" + dstNioFilePath.toString() + "/" + p.getFileName()));
        //     }
        //     stream.close();
        //     Files.delete(srcNioFilePath);
        // } else {
        //     // TODO:
        //     // Next step: design a better way to sync at init function.
        //     syncWithVineyard(dstNioFilePath.toString());
        //     if (Files.exists(dstNioFilePath)) {
        //         printAllFiles();
        //         mergeFile(srcNioFilePath, dstNioFilePath);
        //         Files.delete(srcNioFilePath);
        //     } else {
        //         Files.move(srcNioFilePath, dstNioFilePath);
        //         ByteBuffer bytes = ByteBuffer.allocate(255);
        //         FileChannel channel = FileChannel.open(dstNioFilePath, StandardOpenOption.READ);
        //         int len = channel.read(bytes);
        //         if (len > 0) {
        //             String objectIDStr =
        //                     new String(bytes.array(), 0, len, StandardCharsets.UTF_8)
        //                             .replaceAll("\n", "");
        //             IPCClient client = Context.getClient();
        //             try {
        //                 client.putName(ObjectID.fromString(objectIDStr), dstNioFilePath.toString());
        //                 client.dropName(srcNioFilePath.toString());
        //             } catch (Exception e) {
        //                 // Skip some invalid file.
        //                 // File content may be not a valid object id.
        //                 Context.println("Failed to put name to vineyard: " + e.getMessage());
        //             }
        //         }
        //     }
        // }

        Path newSrc = new Path(src.toString().replaceAll("vineyard", "file"));
        Path newDst = new Path(dst.toString().replaceAll("vineyard", "file"));
        String newTableName = dst.toString().substring(dst.toString().indexOf(":") + 1).replaceAll("///", "/").replaceAll("//", "/");
        String oldTableName = src.toString().substring(src.toString().indexOf(":") + 1).replaceAll("///", "/").replaceAll("//", "/");
        Context.println("new table name:" + newTableName + ", old table name:" + oldTableName);
        try {
            FileStatus srcStatus = fs.getFileStatus(newSrc);
            if (srcStatus.isDirectory()) {
                FileStatus[] status = fs.listStatus(newSrc);
                for (FileStatus s : status) {
                    renameInternal(s.getPath(), new Path(newDst.toString() + "/" + s.getPath().getName()));
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
                        String objectIDStr = new String(objectIDByteArray, 0, len, StandardCharsets.UTF_8)
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
                // if (Files.exists(jimfs.getPath(object.getKey()))) {
                //     continue;
                // }
                // Files.createFile(jimfs.getPath(object.getKey()));
                // FileChannel channel =
                //         FileChannel.open(jimfs.getPath(object.getKey()), StandardOpenOption.WRITE);
                // ObjectID id = object.getValue();
                // channel.write(
                //         ByteBuffer.wrap((id.toString() + "\n").getBytes(StandardCharsets.UTF_8)));
                // channel.close();
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
        Context.println("listStatus: " + path.toString());
        List<FileStatus> result = new ArrayList<FileStatus>();
        try (val lock = this.lock.open()) {
            // java.nio.file.Path nioFilePath =
            //         jimfs.getPath(path.toString().substring(path.toString().indexOf(":") + 1));
            // syncWithVineyard(nioFilePath.toString());
            // if (Files.isDirectory(nioFilePath)) {
            //     DirectoryStream<java.nio.file.Path> stream = Files.newDirectoryStream(nioFilePath);
            //     for (java.nio.file.Path p : stream) {
            //         result.add(
            //                 new FileStatus(
            //                         Files.size(p),
            //                         Files.isDirectory(p),
            //                         1,
            //                         1,
            //                         0,
            //                         0,
            //                         new FsPermission((short) 777),
            //                         null,
            //                         null,
            //                         new Path(SCHEME + ":///" + p.toString())));
            //         System.out.println("path get name:" + new Path(SCHEME + ":///" + p.toString()).getName());
            //     }
            //     stream.close();
            // }
            String prefix = path.toString().substring(path.toString().indexOf(":") + 1).replaceAll("///", "/").replaceAll("//", "/");
            syncWithVineyard(prefix);
            try {
                FileStatus status = fs.getFileStatus(new Path(path.toString().replaceAll("vineyard", "file")));
                FileStatus[] statusArray = fs.listStatus(new Path(path.toString().replaceAll("vineyard", "file")));
                for (FileStatus s : statusArray) {
                    FileStatus temp = new FileStatus(
                        s.getLen(),
                        s.isDirectory(),
                        s.getReplication(),
                        s.getBlockSize(),
                        s.getModificationTime(),
                        s.getAccessTime(),
                        new FsPermission((short)777),
                        s.getOwner(),
                        s.getGroup(),
                        new Path(SCHEME + ":///" + s.getPath().toString().substring(s.getPath().toString().indexOf(":") + 1).replaceAll("///", "//").replaceAll("//", "/"))
                    );
                    Context.println("file:" + temp.getPath().toString());
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
        Context.println("mkdirs: " + path.toString());
        // java.nio.file.Path nioDirPath =
        //         jimfs.getPath(path.toString().substring(path.toString().indexOf(":") + 1));
        // if (Files.exists(nioDirPath)) {
        //     return false;
        // }
        // Files.createDirectories(nioDirPath);

        Path newPath = new Path(path.toString().replaceAll("vineyard", "file"));
        try{
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
        Context.println("getFileStatus: " + path.toString());
        // String pathStr = path.toString().substring(path.toString().indexOf(":") + 1);
        // java.nio.file.Path nioFilePath = jimfs.getPath(pathStr);
        // if (Files.exists(nioFilePath)) {
        //     printAllFiles();
        //     return new FileStatus(
        //             Files.size(nioFilePath),
        //             Files.isDirectory(nioFilePath),
        //             1,
        //             1,
        //             0,
        //             0,
        //             new FsPermission((short) 777),
        //             null,
        //             null,
        //             new Path(SCHEME + ":///" + pathStr));
        // }
        printAllFiles();
        FileStatus temp = fs.getFileStatus(new Path(path.toString().replaceAll("vineyard", "file")));
        FileStatus result = new FileStatus(
            temp.getLen(),
            temp.isDirectory(),
            temp.getReplication(),
            temp.getBlockSize(),
            temp.getModificationTime(),
            temp.getAccessTime(),
            new FsPermission((short)777),
            temp.getOwner(),
            temp.getGroup(),
            new Path(SCHEME + ":///" + temp.getPath().toString().substring(temp.getPath().toString().indexOf(":") + 1).replaceAll("///", "//").replaceAll("//", "/"))
        );
        return result;
        // throw new FileNotFoundException();
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
        // org.apache.hadoop.fs.FileSystem srcFS = src.getFileSystem(conf);
        // FSDataInputStream in = srcFS.open(src);
        // FSDataOutputStream out = create(dst, false);
        // byte[] buffer = new byte[1024];
        // do {
        //     int len = in.read(buffer);
        //     if (len <= 0) {
        //         break;
        //     }
        //     out.write(buffer, 0, len);
        // } while (true);
        // in.close();
        // out.close();
        // Context.println("copy done!");
        // printAllFiles();
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
