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

import io.netty.channel.Channel.Unsafe;
import io.v6d.core.client.Context;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.Buffer;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException.ObjectNotExists;
import io.v6d.hive.ql.io.CloseableReentrantLock;
import java.io.FileNotFoundException;
import java.io.FilePermission;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

import lombok.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.hbase.util.UnsafeAccess;
import org.apache.hadoop.io.DataInputBuffer;
import org.apache.hadoop.io.DataInputByteBuffer;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.util.Progressable;
import org.apache.hive.com.esotericsoftware.kryo.io.UnsafeMemoryInput;
import org.apache.hive.com.esotericsoftware.kryo.io.UnsafeMemoryOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// class VineyardDataOutputStream extends FSDataOutputStream {
//     // private String content = "";
//     private byte[] content;
//     private int length = 0;
//     private Path filePath;
//     private IPCClient client;
//     private UnsafeMemoryOutput output;
//     private static final CloseableReentrantLock lock = new CloseableReentrantLock();

//     public VineyardDataOutputStream(Path filePath, boolean overwrite) throws IOException {
//         super(new DataOutputBuffer(), null);
//         content = new byte[1];
//         client = Context.getClient();
//         ObjectID fileObjectID;
//         this.filePath = filePath;
//         Context.println("create path:" + filePath);
//         try {
//             fileObjectID = client.getName(filePath.toString());
//         } catch (ObjectNotExists e) {
//             // file not exist
//             client = Context.getClient();
//             return;
//         }
//         Context.println("path:" + filePath + " is already exist. Check if need to overwrite.");
//         if (overwrite) {
//             Set<ObjectID> objectIDs = new HashSet<ObjectID>();
//             objectIDs.add(fileObjectID);
//             client.delete(objectIDs, false, false);
//             client.dropName(filePath.toString());
//         } else {
//             throw new IOException("File already exist.");
//         }
//     }

//     @Override
//     public void close() throws IOException {
//         // Write to vineyard
//         ObjectMeta fileMeta = ObjectMeta.empty();
//         fileMeta.setTypename("vineyard::File");
//         fileMeta.setValue("is_dir_", false);

//         // Context.println("content_:" + content);
//         // byte[] contentBytes = content.getBytes(StandardCharsets.US_ASCII);

//         Context.println("length:" + length);
//         Buffer buffer = client.createBuffer(length);
//         output = new UnsafeMemoryOutput(buffer.getPointer(), (int)buffer.getSize());
//         output.write(content, 0, length);
//         output.flush();
//         Context.println("Objectid:" + buffer.getObjectId().value());
//         Context.println("String:" + content);
//         Context.println("Content:" + content);
//         Context.println("String:" + new String(content, StandardCharsets.US_ASCII));

//         ObjectMeta bufferMeta = ObjectMeta.empty();
//         bufferMeta.setId(buffer.getObjectId()); // blob's builder is a special case
//         bufferMeta.setInstanceId(client.getInstanceId());

//         bufferMeta.setTypename("vineyard::Blob");
//         bufferMeta.setNBytes(buffer.getSize());

//         // to make resolving the returned object metadata possible
//         bufferMeta.setBufferUnchecked(buffer.getObjectId(), buffer);
//         client.sealBuffer(buffer.getObjectId());

//         fileMeta.addMember("content_", bufferMeta);

//         fileMeta.setValue("length_", length);
//         fileMeta.setValue("modify_time_", System.currentTimeMillis());
//         fileMeta = client.createMetaData(fileMeta);
//         client.persist(fileMeta.getId());
//         client.putName(fileMeta.getId(), filePath.toString());
//         Context.println("put name:" + filePath.toString());
        
//         Context.println("bind path:" + filePath.toString() + " to file type:" + (fileMeta.getBooleanValue("is_dir_") ? "dir" : "file"));
//         Context.println("bind path:" + filePath.toString() + " to content buffer id:" + bufferMeta.getId().value());
//     }

//     @Override
//     public String toString() {
//         return "vineyard";
//     }

//     private void expandContent() {
//         byte[] newContent = new byte[content.length * 2];
//         System.arraycopy(content, 0, newContent, 0, length);
//         content = newContent;
//     }

//     @Override
//     public void write(int b) throws IOException {
//         // throw new UnsupportedOperationException("should not call this function.");
//         // Context.println("write lock");
//         try (val lock = this.lock.open()) {
//             byte byteValue = (byte) (b & 0xff);
//             if (length >= content.length) {
//                 expandContent();
//             }
//             content[length] = byteValue;
//             Context.println("write:" + (int)(byteValue & 0xff));
//             length++;
//         } finally {
//             // Context.println("unlock");
//         }
//     }

//     @Override
//     public void write(byte b[], int off, int len) throws IOException {
//         // Context.println("write lock");
//         try (val lock = this.lock.open()) {
//             // content += new String(b, off, len, StandardCharsets.US_ASCII);
//             while (length + len >= content.length) {
//                 expandContent();
//             }
//             System.arraycopy(b, off, content, length, len);
//             length += len;
//         } finally {
//             // Context.println("unlock");
//         }
//     }
// }

class VineyardOutputStream extends OutputStream {
    private byte[] content;
    private int length = 0;
    private Path filePath;
    private IPCClient client;
    private UnsafeMemoryOutput output;
    private static final CloseableReentrantLock lock = new CloseableReentrantLock();

    public VineyardOutputStream(Path filePath, boolean overwrite) throws IOException {
        content = new byte[1];
        client = Context.getClient();
        ObjectID fileObjectID;
        this.filePath = filePath;
        Context.println("create path:" + filePath);
        try {
            fileObjectID = client.getName(filePath.toString());
        } catch (ObjectNotExists e) {
            // file not exist
            client = Context.getClient();
            return;
        }
        Context.println("path:" + filePath + " is already exist. Check if need to overwrite.");
        if (overwrite) {
            Set<ObjectID> objectIDs = new HashSet<ObjectID>();
            objectIDs.add(fileObjectID);
            client.delete(objectIDs, false, false);
            client.dropName(filePath.toString());
        } else {
            throw new IOException("File already exist.");
        }
    }

    @Override
    public void close() throws IOException {
        // Write to vineyard
        ObjectMeta fileMeta = ObjectMeta.empty();
        fileMeta.setTypename("vineyard::File");
        fileMeta.setValue("is_dir_", false);

        // Context.println("content_:" + content);
        // byte[] contentBytes = content.getBytes(StandardCharsets.US_ASCII);

        Context.println("length:" + length);
        Buffer buffer = client.createBuffer(length);
        output = new UnsafeMemoryOutput(buffer.getPointer(), (int)buffer.getSize());
        output.write(content, 0, length);
        output.flush();

        ObjectMeta bufferMeta = ObjectMeta.empty();
        bufferMeta.setId(buffer.getObjectId()); // blob's builder is a special case
        bufferMeta.setInstanceId(client.getInstanceId());

        bufferMeta.setTypename("vineyard::Blob");
        bufferMeta.setNBytes(buffer.getSize());

        // to make resolving the returned object metadata possible
        bufferMeta.setBufferUnchecked(buffer.getObjectId(), buffer);
        client.sealBuffer(buffer.getObjectId());

        fileMeta.addMember("content_", bufferMeta);

        fileMeta.setValue("length_", length);
        fileMeta.setValue("modify_time_", System.currentTimeMillis());
        fileMeta = client.createMetaData(fileMeta);
        client.persist(fileMeta.getId());
        client.putName(fileMeta.getId(), filePath.toString());
        Context.println("put name:" + filePath.toString());
        
        Context.println("bind path:" + filePath.toString() + " to file type:" + (fileMeta.getBooleanValue("is_dir_") ? "dir" : "file"));
        Context.println("bind path:" + filePath.toString() + " to content buffer id:" + bufferMeta.getId().value());
    }

    @Override
    public String toString() {
        return "vineyard";
    }

    private void expandContent() {
        byte[] newContent = new byte[content.length * 2];
        System.arraycopy(content, 0, newContent, 0, length);
        content = newContent;
    }

    @Override
    public void write(int b) throws IOException {
        // throw new UnsupportedOperationException("should not call this function.");
        // Context.println("write lock");
        try (val lock = this.lock.open()) {
            byte byteValue = (byte) (b & 0xff);
            if (length >= content.length) {
                expandContent();
            }
            content[length] = byteValue;
            // Context.println("write:" + (int)(byteValue & 0xff));
            length++;
        } finally {
            // Context.println("unlock");
        }
    }

    // @Override
    // public void write(byte b[], int off, int len) throws IOException {
    //     // Context.println("write lock");
    //     try (val lock = this.lock.open()) {
    //         // content += new String(b, off, len, StandardCharsets.US_ASCII);
    //         while (length + len >= content.length) {
    //             expandContent();
    //         }
    //         System.arraycopy(b, off, content, length, len);
    //         length += len;
    //     } finally {
    //         // Context.println("unlock");
    //     }
    // }
}

class VineyardInputStream extends FSInputStream {
    private byte[] content;
    private Path filePath;
    private IPCClient client;
    private int pos = 0;
    private UnsafeMemoryInput input;

    public VineyardInputStream(Path filePath) throws IOException {
        client = Context.getClient();
        this.filePath = filePath;

        ObjectID objectID = client.getName(filePath.toString());
        ObjectMeta meta = client.getMetaData(objectID);
        if (meta.getBooleanValue("is_dir_")) {
            throw new IOException("Can not open a directory.");
        }
        // content = meta.getStringValue("content_").getBytes(StandardCharsets.UTF_8);
        // Buffer buffer = client.getMetaData(objectID).getBuffer(objectID)
        ObjectID contentObjectID = meta.getMemberMeta("content_").getId();
        Context.println("read path:" + filePath.toString() + " content buffer id:" + contentObjectID.value());
        ObjectMeta contentObjectMeta = client.getMetaData(contentObjectID);
        Buffer buffer = contentObjectMeta.getBuffer(contentObjectID);
        content = new byte[(int)buffer.getSize()];
        input = new UnsafeMemoryInput(buffer.getPointer(), (int)buffer.getSize());
        input.read(content);
    }

    @Override
    public void seek(long offset) throws IOException {
        Context.println("seek:" + offset);
        if (offset > content.length) {
            throw new IOException("Seek offset is out of range.");
        }
        pos = (int)offset;
    }

    @Override
    public long getPos() throws IOException {
        return pos;
    }

    @Override
    public boolean seekToNewSource(long l) throws IOException {
        throw new UnsupportedOperationException(
                "Vineyard input stream not support seekToNewSource.");
    }

    @Override
    public int read() throws IOException {
        int result = -1;
        if (pos >= content.length) {
            return result;
        }
        result = (content[pos] & 0xff);
        pos++;
        return result;
    }

    // @Override
    // public int read(long position, byte[] b, int off, int len) throws IOException {
    //     int realReadBytes = len + position > content.length ? content.length - (int)position : len;
    //     System.arraycopy(content, (int)position, b, off, realReadBytes);
    //     Context.println("read:" + b);
    //     for(int i = 0; i < len; i++) {
    //         Context.println("read:" + (int)(b[off + i]));
    //     }
    //     return realReadBytes;
    // }

    @Override
    public void close() throws IOException {
        // Nothint to do.
    }
}

class VineyardDataInputStream extends FSDataInputStream {
    public VineyardDataInputStream(Path filePath) throws IOException {
        super(new VineyardInputStream(filePath));
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
    static boolean enablePrintAllObjects = true;

    private IPCClient client;
    private static final int DIR_LEN = 1;

    Path workingDir = new Path("vineyard:/");

    public FileSystem() {
        super();
    }

    // private static void printAllFiles(Path p) throws IOException {
    //     FileStatus[] status = fs.listStatus(p);
    //     Queue<FileStatus> queue = new java.util.LinkedList<FileStatus>();
    //     for (FileStatus s : status) {
    //         queue.add(s);
    //     }
    //     while (!queue.isEmpty()) {
    //         FileStatus p1 = queue.poll();
    //         Context.println(p1.getPath().toString());
    //         if (p1.isDirectory()) {
    //             FileStatus[] statusTemp = fs.listStatus(p1.getPath());
    //             for (FileStatus s : statusTemp) {
    //                 queue.add(s);
    //             }
    //         }
    //     }
    // }
    private void printAllFiles(Path p) throws IOException {
        Map<String, ObjectID> objects;
        try {
            objects = client.listNames(".*", true, 255);
        } catch (Exception e) {
            Context.println("Failed to list names: " + e.getMessage());
            return;
        }
        for (val object : objects.entrySet()) {
            try {
                ObjectMeta meta = client.getMetaData(object.getValue());
                if (meta.getTypename().compareTo("vineyard::File") == 0) {
                    String type = meta.getBooleanValue("is_dir_") ? "dir" : "file";
                    Context.println("Type:" + type + " " + object.getKey());
                }
            } catch (Exception e) {
                // Skip some invalid object id.
                Context.println("Failed to get object meta: " + e.getMessage());
                continue;
            }
        }
    }

    private void printAllObjectsWithName() throws IOException {
        if (enablePrintAllObjects) {
            IPCClient client = Context.getClient();
            Context.println("print all objects with name");
            Context.println("====================================");
            Map<String, ObjectID> objects = client.listNames(".*", true, 255);
            for (val object : objects.entrySet()) {
                Context.println(
                        "object name:"
                                + object.getKey()
                                + ", object id:"
                                + object.getValue().value());
            }
            Context.println("====================================");
        }
    }

    private void printAllFiles() throws IOException {
        if (enablePrintAllFiles) {
            Context.println("-----------------------------------");
            printAllFiles(new Path("/opt/hive/data/warehouse"));
            Context.println("-----------------------------------");
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
        Context.println("Initialize vineyard file system: " + name.toString());
        super.initialize(name, conf);
        this.conf = conf;
        this.uri = name;
        this.client = Context.getClient();

        // fs = new RawLocalFileSystem();
        // fs.initialize(URI.create("file:///"), conf);
        mkdirs(new Path(uri.toString()), new FsPermission((short) 777));
    }

    @Override
    public FSDataInputStream open(Path path, int i) throws IOException {
        // Path newPath = new Path(path.toString().replaceAll("vineyard", "file"));
        // FSDataInputStream result = fs.open(newPath);
        Context.println("open:" + path);
        try (val lock = this.lock.open()) {
            FSDataInputStream result = new FSDataInputStream(new VineyardInputStream(new Path(path.toString().replaceAll("/+", "/"))));
            return result;
        } finally {
            // Context.println("unlock");
        }
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
        // Context.println("create lock");
        try (val lock = this.lock.open()) {
            return createInternal(
                    new Path(path.toString().replaceAll("/+", "/")),
                    fsPermission,
                    overwrite,
                    bufferSize,
                    replication,
                    blockSize,
                    progressable);
        } finally {
            // Context.println("unlock");
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
        Context.println("create:" + path);
        Path parentPath = path.getParent();
        try {
            getFileStatusInternal(parentPath);
        } catch (FileNotFoundException e) {
            // parent not exist
            mkdirsInternal(parentPath, new FsPermission((short) 777));
        }
        FSDataOutputStream result = new FSDataOutputStream(new VineyardOutputStream(path, overwrite), null);
        return result;
    }

    @Override
    public FSDataOutputStream append(Path path, int i, Progressable progressable)
            throws IOException {
        return null;
    }

    @Override
    public boolean delete(Path path, boolean b) throws IOException {
        // Context.println("delete lock");
        try (val lock = this.lock.open()) {
            return this.deleteInternal(new Path(path.toString().replaceAll("/+", "/")), b);
        } finally {
            // Context.println("unlock");
        }
    }

    // private void cleanObjectInVineyard(Path filePath) throws IOException {
    //     IPCClient client = Context.getClient();
    //     Queue<Path> queue = new java.util.LinkedList<Path>();
    //     Collection<ObjectID> objectIDs = new ArrayList<ObjectID>();
    //     queue.add(filePath);
    //     Context.println("delete path:" + filePath.toString());
    //     while (!queue.isEmpty()) {
    //         Path path = queue.peek();
    //         Context.println("path:" + path.toString());
    //         try {
    //             FileStatus fileStatus = fs.getFileStatus(path);
    //             if (fileStatus.isDirectory()) {
    //                 FileStatus[] fileStatusArray = fs.listStatus(path);
    //                 for (FileStatus s : fileStatusArray) {
    //                     if (s.getPath().toString().compareTo(filePath.toString()) == 0) {
    //                         continue;
    //                     }
    //                     queue.add(s.getPath());
    //                 }
    //             } else {

    //                 FSDataInputStream in = fs.open(path);
    //                 byte[] objectIDByteArray = new byte[(int)fileStatus.getLen()];
    //                 int len = in.read(objectIDByteArray);
    //                 String[] objectIDStrs =
    //                         new String(objectIDByteArray, 0, len, StandardCharsets.UTF_8)
    //                                 .split("\n");
    //                 for (String objectIDStr : objectIDStrs) {
    //                     try {
    //                         ObjectID objectID = ObjectID.fromString(objectIDStr);
    //                         objectIDs.add(objectID);
    //                     } catch (Exception e) {
    //                         // Skip some invalid file.
    //                         // File content may be not a valid object id.
    //                         Context.println("Failed to parse object id: " + e.getMessage());
    //                     }
    //                 }
    //             }
    //             // String objectName = path.toString().substring(path.toString().indexOf(":") + 1);
    //             // ObjectID objectID = client.getName(objectName);
    //             // objectIDs.add(objectID);
    //             // client.dropName(objectName);
    //         } catch (FileNotFoundException e) {
    //             // file not exist, skip
    //             Context.println("File: " + path.toString() + " not exist.");
    //             continue;
    //         } catch (ObjectNotExists e) {
    //             // object not exist
    //             Context.println("Object of file: " + path.toString() + " not exist.");
    //             continue;
    //         } finally {
    //             queue.poll();
    //         }
    //     }
    //     client.delete(objectIDs, false, false);
    //     printAllObjectsWithName();
    // }

    private void deleteVineyardObjectWithName(String[] names) throws IOException {
        IPCClient client = Context.getClient();
        Set<ObjectID> objectIDs = new HashSet<ObjectID>();
        for (String name : names) {
            // Context.println("delete name:" + name);
            ObjectID objectID = client.getName(name);
            objectIDs.add(objectID);
            client.dropName(name);
        }
        client.delete(objectIDs, true, true);
    }

    private void deleteVineyardObjectWithObjectIDStr(String[] objectIDStrs) throws IOException {
        IPCClient client = Context.getClient();
        Set<ObjectID> objectIDs = new HashSet<ObjectID>();
        for (String objectIDStr : objectIDStrs) {
            try {
                Context.println("delete id:" + ObjectID.fromString(objectIDStr).value());
                ObjectID objectID = ObjectID.fromString(objectIDStr);
                objectIDs.add(objectID);
            } catch (Exception e) {
                // Skip some invalid object id.
                Context.println("Failed to parse object id: " + e.getMessage());
                break;
            }
        }
        client.delete(objectIDs, true, true);
    }

    private boolean deleteInternal(Path path, boolean b) throws IOException {
        Context.println("delete:" + path);
        FileStatus fileStatus;
        try { 
            fileStatus = getFileStatusInternal(path);
        } catch (FileNotFoundException e) {
            // file not exist
            Context.println("File not exist.");
            return false;
        }

        if (fileStatus.isDirectory()) {
            FileStatus[] childFileStatus = listStatusInternal(path);
            if (childFileStatus.length > 0 && !b) {
                throw new IOException("Directory is not empty.");
            }
            for (FileStatus child : childFileStatus) {
                deleteInternal(child.getPath(), b);
            }
            deleteVineyardObjectWithName(new String[] {path.toString()});
            printAllFiles();
            return true;
        }

        try {
            FSDataInputStream in = open(path, 0);
            byte[] objectIDByteArray = new byte[(int)fileStatus.getLen()];
            int len = in.read(objectIDByteArray);
            String[] objectIDStrs =
                    new String(objectIDByteArray, 0, len, StandardCharsets.US_ASCII).split("\n");
            deleteVineyardObjectWithObjectIDStr(objectIDStrs);
            deleteVineyardObjectWithName(new String[] {path.toString()});
        } catch(Exception e) {
            Context.println("Failed to delete file: " + e.getMessage());
        }
        printAllFiles();
        return true;
    }

    @Override
    public boolean rename(Path src, Path dst) throws IOException {
        // Context.println("rename lock");
        try (val lock = this.lock.open()) {
            val watch = StopwatchContext.create();
            String srcString = src.toString().replaceAll("/+", "/");
            String dstString = dst.toString().replaceAll("/+", "/");
            val renamed = this.renameInternal(new Path(srcString), new Path(dstString));
            Context.println("Filesystem rename uses: " + watch.stop());
            return renamed;
        } finally {
            // Context.println("unlock");
        }
    }

    private void deleteFileWithoutObject(Path path) throws IOException {
        Context.println("delete file without object:" + path);
        ObjectID objectID = client.getName(path.toString());
        Set<ObjectID> objectIDs = new HashSet<ObjectID>();
        objectIDs.add(objectID);
        client.delete(objectIDs, true, true);
        client.dropName(path.toString());
    }

    private void mergeFile(Path src, Path dst) throws IOException {
        Context.println("merge file.");
        FSDataInputStream srcInput = open(src, 0);
        FSDataInputStream dstInput = open(dst, 0);
        FileStatus srcStatus = getFileStatusInternal(src);
        FileStatus dstStatus = getFileStatusInternal(dst);
        byte[] srcContent = new byte[(int)srcStatus.getLen()];
        byte[] dstContent = new byte[(int)dstStatus.getLen()];
        srcInput.read(srcContent);
        dstInput.read(dstContent);
        srcInput.close();
        dstInput.close();

        FSDataOutputStream out = createInternal(dst, new FsPermission((short)777), true, 0, (short)1, 1, null);
        out.write(srcContent);
        out.write(dstContent);
        out.close();
    }

    private boolean renameInternal(Path src, Path dst) throws IOException {
        Context.println("rename:" + src + " to " + dst);
        FileStatus srcStatus;
        try {
            srcStatus = getFileStatusInternal(src);
        } catch (FileNotFoundException e) {
            // src file not exist
            Context.println("Src file not exist");
            return false;
        }

        Path dstParentPath = dst.getParent();
        try {
            getFileStatusInternal(dstParentPath);
        } catch (FileNotFoundException e) {
            // dst parent not exist
            Context.println("Dst parent not exist");
            mkdirsInternal(dstParentPath, new FsPermission((short) 777));
        }

        if (srcStatus.isDirectory()) {
            ObjectID objectID = client.getName(src.toString());
            // Context.println("put name:" + dst.toString());
            client.putName(objectID, dst.toString());

            FileStatus[] status = listStatusInternal(src);
            for (FileStatus s : status) {
                renameInternal(
                        s.getPath(), new Path((dst.toString() + "/" + s.getPath().getName()).replaceAll("/+", "/")));
            }
            client.dropName(src.toString());
            return true;
        } else {
            try {
                getFileStatusInternal(dst);
            } catch (FileNotFoundException e) {
                // dst file not exist

                ObjectID objectID = client.getName(src.toString());
                client.putName(objectID, dst.toString());
                Context.println("put name:" + dst.toString());
                client.dropName(src.toString());
                printAllFiles();
                return true;
            }
            // dst file exist
            Context.println("dst exist!");
            mergeFile(src, dst);
            // deleteInternal(newSrc, true);
            deleteFileWithoutObject(src);

            printAllFiles();
            return true;
        }
    }

    // public void syncWithVineyard(String prefix) throws IOException {
    //     IPCClient client = Context.getClient();
    //     String reg = "^" + prefix + ".*";
    //     Map<String, ObjectID> objects = client.listNames(reg, true, 255);
    //     for (val object : objects.entrySet()) {
    //         try {
    //             fs.getFileStatus(new Path("file://" + object.getKey()));
    //         } catch (FileNotFoundException e) {
    //             // file not exist
    //             Path path = new Path("file://" + object.getKey());
    //             FSDataOutputStream out = fs.create(path);
    //             ObjectID id = object.getValue();
    //             out.write((id.toString() + "\n").getBytes(StandardCharsets.UTF_8));
    //             out.close();
    //         }
    //     }
    // }

    // private void syncWithVineyard(String prefix) throws IOException {
    //     IPCClient client = Context.getClient();
    //     String reg = "^" + prefix + ".*";
    //     Map<String, ObjectID> objects = client.listNames(reg, true, 255);
    //     for (val object : objects.entrySet()) {
    //         try {
    //             getFileStatusInternal(new Path(object.getKey()));
    //         } catch (FileNotFoundException e) {
    //             // file not exist
    //             Path path = new Path("file://" + object.getKey());
    //             FSDataOutputStream out = fs.create(path);
    //             ObjectID id = object.getValue();
    //             out.write((id.toString() + "\n").getBytes(StandardCharsets.UTF_8));
    //             out.close();
    //         }
    //     }
    // }

    @Override
    public FileStatus[] listStatus(Path path) throws FileNotFoundException, IOException {
        // Context.println("listStatus lock");
        try (val lock = this.lock.open()) {
            return listStatusInternal(new Path(path.toString().replaceAll("/+", "/")));
        } finally {
            // Context.println("unlock");
        }
    }

    private FileStatus[] listStatusInternal(Path path) throws FileNotFoundException, IOException {
        Context.println("list:" + path);
        List<FileStatus> result = new ArrayList<FileStatus>();
        ObjectID fileObjectID;
        try {
            fileObjectID = client.getName(path.toString());
        } catch (ObjectNotExists e) {
            throw new FileNotFoundException(path.toString() + " is not found.");
        }
        ObjectMeta fileMeta = client.getMetaData(fileObjectID);
        if (!fileMeta.getBooleanValue("is_dir_")) {
            //file
            FileStatus temp =
                    new FileStatus(
                            fileMeta.getIntValue("length_"),
                            false,
                            1,
                            1,
                            fileMeta.getLongValue("modify_time_"),
                            System.currentTimeMillis(),
                            new FsPermission((short) 777),
                            null,
                            null,
                            new Path(path.toString()));
            result.add(temp);
        } else {
            // dir
            Path newPath = new Path(path.toString().replaceAll("/+", "/"));
            // syncWithVineyard(newPath.toString());
            String pattern = "^" + newPath.toString() + "/[^/]*";
            // Context.println(pattern);
            // Context.println("try to list name");
            Map<String, ObjectID> objects = Context.getClient().listNames(pattern, true, 255);
            // Context.println("list end");
            for (val object : objects.entrySet()) {
                // Context.println("find:" + object.getKey());
                ObjectID objectID = object.getValue();
                ObjectMeta meta = client.getMetaData(objectID);
                if (meta.getTypename().compareTo("vineyard::File") != 0) {
                    continue;
                }
                boolean isDir = meta.getBooleanValue("is_dir_");
                int len = isDir ? DIR_LEN : meta.getIntValue("length_");
                long modifyTime = meta.getLongValue("modify_time_");
                FileStatus temp =
                        new FileStatus(
                                len,
                                isDir,
                                1,
                                1,
                                1,
                                modifyTime,
                                new FsPermission((short) 777),
                                null,
                                null,
                                new Path(
                                        SCHEME
                                                + ":///"
                                                + object.getKey()
                                                        .substring(
                                                                object.getKey()
                                                                                .indexOf(
                                                                                        ":")
                                                                        + 1)));
                result.add(temp);
            }
        }
        printAllFiles();
        // Context.println("result size:" + result.size());
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
        // Context.println("mkdirs lock");
        try (val lock = this.lock.open()) {
            return this.mkdirsInternal(new Path(path.toString().replaceAll("/+", "/")), fsPermission);
        } finally {
            // Context.println("unlock");
        }
    }

    private boolean mkdirsInternal(Path path, FsPermission fsPermission) throws IOException {
        Context.println("mkdir:" + path);
        try {
            getFileStatusInternal(path);
        } catch (FileNotFoundException e) {
            // file not exist
            Path parentPath = path.getParent();
            if (parentPath != null) {
                mkdirsInternal(parentPath, fsPermission);
            }

            Context.println("file not found, create dir!");
            ObjectMeta dirMeta = ObjectMeta.empty();
            dirMeta.setTypename("vineyard::File");
            dirMeta.setValue("is_dir_", true);
            dirMeta.setValue("length_", DIR_LEN);
            dirMeta.setValue("modify_time_", System.currentTimeMillis());
            dirMeta = client.createMetaData(dirMeta);
            client.persist(dirMeta.getId());
            Context.println("put name:" + path.toString());
            client.putName(dirMeta.getId(), path.toString());
            printAllFiles();
            return true;
        }
        return false;
    }

    @Override
    public FileStatus getFileStatus(Path path) throws IOException {
        // Context.println(" getFileStatus lock");
        try (val lock = this.lock.open()) {
            return this.getFileStatusInternal(new Path(path.toString().replaceAll("/+", "/")));
        } finally {
            // Context.println("unlock");
        }
    }

    public FileStatus getFileStatusInternal(Path path) throws IOException {
        printAllFiles();
        Context.println("getFileStatus:" + path);
        ObjectID fileObjectID;
        try {
            fileObjectID = client.getName(path.toString());
        } catch (ObjectNotExists e) {
            throw new FileNotFoundException(path.toString() + " is not found.");
        }
        // Context.println("find!");
        ObjectMeta meta = client.getMetaData(fileObjectID);
        boolean isDir = meta.getBooleanValue("is_dir_");
        int len = meta.getIntValue("length_");
        long modifyTime = meta.getLongValue("modify_time_");
        FileStatus result =
                new FileStatus(
                        len,
                        isDir,
                        1,
                        1,
                        1,
                        modifyTime,
                        new FsPermission((short) 777),
                        null,
                        null,
                        path);
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
