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
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.ObjectID;
import io.v6d.hive.ql.io.CloseableReentrantLock;
import io.v6d.modules.basic.arrow.TableBuilder;
import io.v6d.modules.basic.filesystem.VineyardFile;
import io.v6d.modules.basic.filesystem.VineyardFileStat;
import io.v6d.modules.basic.filesystem.VineyardFileUtils;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import lombok.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.util.Progressable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class VineyardOutputStream extends OutputStream {
    private final VineyardFile file;

    public VineyardOutputStream(Path filePath, boolean overwrite) throws IOException {
        file = new VineyardFile(filePath.toString(), false, VineyardFile.Mode.WRITE, overwrite);
    }

    @Override
    public void close() throws IOException {
        file.close();
    }

    @Override
    public String toString() {
        return "vineyard";
    }

    @Override
    public void write(int b) throws IOException {
        file.write(b);
    }
}

class VineyardInputStream extends FSInputStream {
    private final VineyardFile file;

    public VineyardInputStream(Path filePath) throws IOException {
        file = new VineyardFile(filePath.toString(), false, VineyardFile.Mode.READ, false);
    }

    @Override
    public void seek(long offset) throws IOException {
        file.seek(offset);
    }

    @Override
    public long getPos() throws IOException {
        return file.getPos();
    }

    @Override
    public boolean seekToNewSource(long l) throws IOException {
        throw new UnsupportedOperationException(
                "Vineyard input stream not support seekToNewSource.");
    }

    @Override
    public int read() throws IOException {
        return file.read();
    }

    @Override
    public void close() throws IOException {
        file.close();
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

    private boolean enablePrintAllFiles = false;
    private IPCClient client;

    Path workingDir = new Path("vineyard:/");

    public FileSystem() {
        super();
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

        mkdirs(new Path(uri.toString()), new FsPermission((short) 0777));
    }

    private void printAllFiles() throws IOException {
        if (this.enablePrintAllFiles) {
            VineyardFileUtils.printAllFiles(this.client);
        }
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
            mkdirsInternal(parentPath, new FsPermission((short) 0777));
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
            try (FSDataInputStream in = open(path, 0)) {
                byte[] objectIDByteArray = new byte[(int) fileStatus.getLen()];
                int len = in.read(objectIDByteArray);
                String[] objectIDStrs =
                        new String(objectIDByteArray, 0, len, StandardCharsets.US_ASCII)
                                .split("\n");
                deleteVineyardObjectWithObjectIDStr(objectIDStrs);
            }
            deleteVineyardObjectWithName(new String[] {path.toString()});
        } catch (Exception e) {
            Context.println("Failed to delete file: " + e.getMessage());
            return false;
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
        // check if source file store object id
        String srcContentStr = new String(srcContent, StandardCharsets.US_ASCII);
        String srcObjectIDStr = srcContentStr.substring(0, srcContentStr.length() - 1);
        ObjectID srcObjectID;
        try {
            srcObjectID = ObjectID.fromString(srcObjectIDStr);
        } catch (Exception e) {
            // invalid object id, merge file directly.
            FSDataOutputStream out =
                    createInternal(
                            dst, new FsPermission((short) 0777), true, 0, (short) 1, 1, null);
            out.write(srcContent);
            out.write(dstContent);
            out.close();
            return;
        }
        ObjectMeta srcTableMeta = client.getMetaData(srcObjectID, false);

        String dstContentStr = new String(dstContent, StandardCharsets.US_ASCII);
        String dstObjectIDStr = dstContentStr.substring(0, dstContentStr.length() - 1);
        ObjectID dstObjectID;
        // if dst do not store object id, throw exception.
        dstObjectID = ObjectID.fromString(dstObjectIDStr);
        ObjectMeta dstTableMeta = client.getMetaData(dstObjectID, false);

        // Merge src tables and dst tables
        ObjectMeta mergedTableMeta =
                TableBuilder.mergeTables(client, new ObjectMeta[] {srcTableMeta, dstTableMeta});
        ObjectID mergObjectID = mergedTableMeta.getId();
        client.persist(mergObjectID);
        FSDataOutputStream out =
                createInternal(dst, new FsPermission((short) 0777), true, 0, (short) 1, 1, null);
        out.write((mergObjectID.toString() + "\n").getBytes(StandardCharsets.US_ASCII));
        out.close();

        // clean up
        Set<ObjectID> objectIDs = new HashSet<ObjectID>();
        objectIDs.add(srcObjectID);
        objectIDs.add(dstObjectID);
        client.delete(objectIDs, false, false);
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
            // dst parent not exist, create first
            mkdirsInternal(dstParentPath, new FsPermission((short) 0777));
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
                client.dropName(src.toString());
                printAllFiles();
                return true;
            }
            // dst file exist
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
        VineyardFileStat[] vineyardFileStats =
                VineyardFileUtils.listFileStats(client, path.toString());
        for (VineyardFileStat vineyardFileStat : vineyardFileStats) {
            FileStatus temp =
                    new FileStatus(
                            vineyardFileStat.getLength(),
                            vineyardFileStat.isDir(),
                            vineyardFileStat.getBlockReplication(),
                            vineyardFileStat.getBlockSize(),
                            vineyardFileStat.getModifyTime(),
                            vineyardFileStat.getAccessTime(),
                            new FsPermission(vineyardFileStat.getPermission()),
                            null,
                            null,
                            new Path(vineyardFileStat.getFilePath()));
            result.add(temp);
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
        try {
            getFileStatusInternal(path);
        } catch (FileNotFoundException e) {
            // file not exist
            Path parentPath = path.getParent();
            if (parentPath != null) {
                mkdirsInternal(parentPath, fsPermission);
            }

            new VineyardFile(path.toString(), true, VineyardFile.Mode.WRITE, false);
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
        VineyardFileStat vineyardFileStat =
                VineyardFileUtils.getFileStatus(client, path.toString());
        FileStatus result =
                new FileStatus(
                        vineyardFileStat.getLength(),
                        vineyardFileStat.isDir(),
                        vineyardFileStat.getBlockReplication(),
                        vineyardFileStat.getBlockSize(),
                        vineyardFileStat.getModifyTime(),
                        vineyardFileStat.getAccessTime(),
                        new FsPermission(vineyardFileStat.getPermission()),
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
