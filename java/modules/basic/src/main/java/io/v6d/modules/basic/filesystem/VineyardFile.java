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
package io.v6d.modules.basic.filesystem;

import io.v6d.core.client.Context;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException.ObjectNotExists;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashSet;
import java.util.Set;

public class VineyardFile {
    private byte[] content;
    private String filePath;
    private IPCClient client;
    private int pos = 0;
    private int length = 0;
    private Mode mode;
    private boolean isDir;
    private static final int DIR_LEN = 1;

    public enum Mode {
        READ,
        WRITE
    };

    public VineyardFile(String path, boolean isDir, Mode mode, boolean overwrite)
            throws IOException {
        this.filePath = path;
        this.client = Context.getClient();
        this.mode = mode;
        this.isDir = isDir;
        if (isDir) {
            makeDir();
            return;
        }

        if (mode == Mode.READ) {
            initializeRead();
        } else if (mode == Mode.WRITE) {
            initializeWrite(overwrite);
        } else {
            throw new IOException("Illegal file mode of:" + mode);
        }
    }

    private void makeDir() throws IOException {
        ObjectMeta dirMeta = ObjectMeta.empty();
        dirMeta.setTypename("vineyard::File");
        dirMeta.setValue("is_dir_", true);
        dirMeta.setValue("length_", DIR_LEN);
        dirMeta.setValue("modify_time_", System.currentTimeMillis());
        dirMeta.setValue("access_time_", (long) -1);
        dirMeta = client.createMetaData(dirMeta);
        client.persist(dirMeta.getId());
        client.putName(dirMeta.getId(), filePath);
    }

    private void initializeRead() throws IOException {
        ObjectID objectID = client.getName(filePath);
        // File must be migrated if it is not at local.
        ObjectMeta meta = client.getMetaData(objectID, true);
        if (meta.getTypename().equals("vineyard::File") == false) {
            throw new IOException("Not a vineyard file.");
        }
        if (meta.getBooleanValue("is_dir_")) {
            throw new IOException("Can not open a directory.");
        }
        byte[] base64EncodedContent =
                meta.getStringValue("base64_content_").getBytes(StandardCharsets.UTF_8);
        content = Base64.getDecoder().decode(base64EncodedContent);
        length = content.length;
    }

    private void initializeWrite(boolean overwrite) throws IOException {
        content = new byte[1];
        client = Context.getClient();
        ObjectID fileObjectID;
        try {
            fileObjectID = client.getName(filePath);
        } catch (ObjectNotExists e) {
            // file not exist
            client = Context.getClient();
            return;
        }
        if (overwrite) {
            Set<ObjectID> objectIDs = new HashSet<ObjectID>();
            objectIDs.add(fileObjectID);
            client.delete(objectIDs, false, false);
            client.dropName(filePath);
        } else {
            throw new IOException("File already exist.");
        }
    }

    public void close() throws IOException {
        if (this.isDir || this.mode == Mode.READ) {
            // do nothing
            return;
        }

        // Write to vineyard
        ObjectMeta fileMeta = ObjectMeta.empty();
        fileMeta.setTypename("vineyard::File");
        fileMeta.setValue("is_dir_", false);
        fileMeta.setValue("modify_time_", System.currentTimeMillis());
        fileMeta.setValue("access_time_", (long) -1);

        content = Arrays.copyOfRange(content, 0, length);
        byte[] base64EncodedContent = Base64.getEncoder().encode(content);
        fileMeta.setValue(
                "base64_content_", new String(base64EncodedContent, StandardCharsets.UTF_8));
        fileMeta.setValue("length_", length);

        fileMeta = client.createMetaData(fileMeta);
        client.persist(fileMeta.getId());
        client.putName(fileMeta.getId(), filePath);
    }

    private void expandContent() {
        byte[] newContent = new byte[content.length * 2];
        System.arraycopy(content, 0, newContent, 0, length);
        content = newContent;
    }

    public void write(int b) throws IOException {
        byte byteValue = (byte) (b & 0xff);
        if (length >= content.length) {
            expandContent();
        }
        content[length] = byteValue;
        length++;
    }

    public void seek(long offset) throws IOException {
        if (offset > content.length || offset < 0) {
            throw new IOException("Seek offset is out of range.");
        }
        pos = (int) offset;
    }

    public long getPos() throws IOException {
        return pos;
    }

    public int read() throws IOException {
        int result = -1;
        if (pos >= content.length) {
            return result;
        }
        result = content[pos] & 0xff;
        pos++;
        return result;
    }
}
