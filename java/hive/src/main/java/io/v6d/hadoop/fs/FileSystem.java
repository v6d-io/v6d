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
import io.v6d.core.client.ds.Buffer;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException.ObjectNotExists;
import io.v6d.hive.ql.io.CloseableReentrantLock;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import lombok.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.util.Progressable;
import org.apache.hive.com.esotericsoftware.kryo.io.UnsafeMemoryInput;
import org.apache.hive.com.esotericsoftware.kryo.io.UnsafeMemoryOutput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class VineyardOutputStream extends OutputStream {
    private byte[] content;
    private int length = 0;
    private Path filePath;
    private IPCClient client;
    private UnsafeMemoryOutput output;

    public VineyardOutputStream(Path filePath, boolean overwrite) throws IOException {
        content = new byte[1];
        client = Context.getClient();
        ObjectID fileObjectID;
        this.filePath = filePath;
        try {
            fileObjectID = client.getName(filePath.toString());
        } catch (ObjectNotExists e) {
            // file not exist
            client = Context.getClient();
            return;
        }
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

        Context.println("length:" + length);
        Buffer buffer = client.createBuffer(length);
        output = new UnsafeMemoryOutput(buffer.getPointer(), (int) buffer.getSize());
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
        byte byteValue = (byte) (b & 0xff);
        if (length >= content.length) {
            expandContent();
        }
        content[length] = byteValue;
        length++;
    }
}

class VineyardInputStream extends FSInputStream {
    private byte[] content;
    private IPCClient client;
    private int pos = 0;
    private UnsafeMemoryInput input;

    public VineyardInputStream(Path filePath) throws IOException {
        client = Context.getClient();

        ObjectID objectID = client.getName(filePath.toString());
        ObjectMeta meta = client.getMetaData(objectID);
        if (meta.getBooleanValue("is_dir_")) {
            throw new IOException("Can not open a directory.");
        }
        ObjectID contentObjectID = meta.getMemberMeta("content_").getId();
        ObjectMeta contentObjectMeta = client.getMetaData(contentObjectID);
        Buffer buffer = contentObjectMeta.getBuffer(contentObjectID);
        content = new byte[(int) buffer.getSize()];
        input = new UnsafeMemoryInput(buffer.getPointer(), (int) buffer.getSize());
        input.read(content);
    }

    @Override
    public void seek(long offset) throws IOException {
        if (offset > content.length) {
            throw new IOException("Seek offset is out of range.");
        }
        pos = (int) offset;
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

    private void printAllFiles() throws IOException {
        if (enablePrintAllFiles) {
            Context.println("-----------------------------------");
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
        this.uri = name;
        this.conf = conf;
        this.client = Context.getClient();

        mkdirs(new Path(uri.toString()), new FsPermission((short) 777));
    }

    @Override
    public FSDataInputStream open(Path path, int i) throws IOException {
        try (val lock = this.lock.open()) {
            FSDataInputStream result =
                    new FSDataInputStream(
                            new VineyardInputStream(
                                    new Path(path.toString().replaceAll("/+", "/"))));
            return result;
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
        try (val lock = this.lock.open()) {
            return createInternal(
                    new Path(path.toString().replaceAll("/+", "/")),
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
        Path parentPath = path.getParent();
        try {
            getFileStatusInternal(parentPath);
        } catch (FileNotFoundException e) {
            // parent not exist
            mkdirsInternal(parentPath, new FsPermission((short) 777));
        }
        FSDataOutputStream result =
                new FSDataOutputStream(new VineyardOutputStream(path, overwrite), null);
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
            return this.deleteInternal(new Path(path.toString().replaceAll("/+", "/")), b);
        }
    }

    private void deleteVineyardObjectWithName(String[] names) throws IOException {
        IPCClient client = Context.getClient();
        Set<ObjectID> objectIDs = new HashSet<ObjectID>();
        for (String name : names) {
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
            byte[] objectIDByteArray = new byte[(int) fileStatus.getLen()];
            int len = in.read(objectIDByteArray);
            String[] objectIDStrs =
                    new String(objectIDByteArray, 0, len, StandardCharsets.US_ASCII).split("\n");
            deleteVineyardObjectWithObjectIDStr(objectIDStrs);
            deleteVineyardObjectWithName(new String[] {path.toString()});
        } catch (Exception e) {
            Context.println("Failed to delete file: " + e.getMessage());
        }
        printAllFiles();
        return true;
    }

    @Override
    public boolean rename(Path src, Path dst) throws IOException {
        try (val lock = this.lock.open()) {
            val watch = StopwatchContext.create();
            String srcString = src.toString().replaceAll("/+", "/");
            String dstString = dst.toString().replaceAll("/+", "/");
            val renamed = this.renameInternal(new Path(srcString), new Path(dstString));
            Context.println("Filesystem rename uses: " + watch.stop());
            return renamed;
        }
    }

    private void deleteFileWithoutObject(Path path) throws IOException {
        ObjectID objectID = client.getName(path.toString());
        Set<ObjectID> objectIDs = new HashSet<ObjectID>();
        objectIDs.add(objectID);
        client.delete(objectIDs, true, true);
        client.dropName(path.toString());
    }

    private void mergeFile(Path src, Path dst) throws IOException {
        FSDataInputStream srcInput = open(src, 0);
        FSDataInputStream dstInput = open(dst, 0);
        FileStatus srcStatus = getFileStatusInternal(src);
        FileStatus dstStatus = getFileStatusInternal(dst);
        byte[] srcContent = new byte[(int) srcStatus.getLen()];
        byte[] dstContent = new byte[(int) dstStatus.getLen()];
        srcInput.read(srcContent);
        dstInput.read(dstContent);
        srcInput.close();
        dstInput.close();

        FSDataOutputStream out =
                createInternal(dst, new FsPermission((short) 777), true, 0, (short) 1, 1, null);
        out.write(srcContent);
        out.write(dstContent);
        out.close();
    }

    private boolean renameInternal(Path src, Path dst) throws IOException {
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
            client.putName(objectID, dst.toString());

            FileStatus[] status = listStatusInternal(src);
            for (FileStatus s : status) {
                renameInternal(
                        s.getPath(),
                        new Path(
                                (dst.toString() + "/" + s.getPath().getName())
                                        .replaceAll("/+", "/")));
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
            deleteFileWithoutObject(src);

            printAllFiles();
            return true;
        }
    }

    @Override
    public FileStatus[] listStatus(Path path) throws FileNotFoundException, IOException {
        try (val lock = this.lock.open()) {
            return listStatusInternal(new Path(path.toString().replaceAll("/+", "/")));
        }
    }

    private FileStatus[] listStatusInternal(Path path) throws FileNotFoundException, IOException {
        List<FileStatus> result = new ArrayList<FileStatus>();
        ObjectID fileObjectID;
        try {
            fileObjectID = client.getName(path.toString());
        } catch (ObjectNotExists e) {
            throw new FileNotFoundException(path.toString() + " is not found.");
        }
        ObjectMeta fileMeta = client.getMetaData(fileObjectID);
        if (!fileMeta.getBooleanValue("is_dir_")) {
            // file
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
            String pattern = "^" + newPath.toString() + "/[^/]*";
            Map<String, ObjectID> objects = Context.getClient().listNames(pattern, true, 255);
            for (val object : objects.entrySet()) {
                ObjectID objectID = object.getValue();
                ObjectMeta meta = client.getMetaData(objectID);
                if (meta.getTypename().compareTo("vineyard::File") != 0) {
                    continue;
                }
                boolean isDir = meta.getBooleanValue("is_dir_");
                int len = isDir ? DIR_LEN : meta.getIntValue("length_");
                long modifyTime = meta.getLongValue("modify_time_");
                Path objectPath = new Path(object.getKey());
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
                                objectPath);
                result.add(temp);
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
            return this.mkdirsInternal(
                    new Path(path.toString().replaceAll("/+", "/")), fsPermission);
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
        try (val lock = this.lock.open()) {
            return this.getFileStatusInternal(new Path(path.toString().replaceAll("/+", "/")));
        }
    }

    public FileStatus getFileStatusInternal(Path path) throws IOException {
        printAllFiles();
        ObjectID fileObjectID;
        try {
            fileObjectID = client.getName(path.toString());
        } catch (ObjectNotExists e) {
            throw new FileNotFoundException(path.toString() + " is not found.");
        }
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
