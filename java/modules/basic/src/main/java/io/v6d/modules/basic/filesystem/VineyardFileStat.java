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

public class VineyardFileStat {
    private String filePath;
    private boolean isDir;
    private long modifyTime;
    private long length;
    private long accessTime;
    private int blockReplication;
    private long blockSize;
    private String owner;
    private String group;
    private short permission;

    public VineyardFileStat(
            String filePath,
            boolean isDir,
            long modifyTime,
            long length,
            long accessTime,
            int blockReplication,
            long blockSize,
            String owner,
            String group,
            short permission) {
        this.filePath = filePath;
        this.isDir = isDir;
        this.modifyTime = modifyTime;
        this.length = length;
        this.accessTime = accessTime;
        this.blockReplication = blockReplication;
        this.blockSize = blockSize;
        this.owner = owner;
        this.group = group;
        this.permission = permission;
    }

    public String getFilePath() {
        return filePath;
    }

    public boolean isDir() {
        return isDir;
    }

    public long getModifyTime() {
        return modifyTime;
    }

    public long getLength() {
        return length;
    }

    public long getAccessTime() {
        return accessTime;
    }

    public int getBlockReplication() {
        return blockReplication;
    }

    public long getBlockSize() {
        return blockSize;
    }

    public String getOwner() {
        return owner;
    }

    public String getGroup() {
        return group;
    }

    public short getPermission() {
        return permission;
    }

    public void setFilePath(String filePath) {
        this.filePath = filePath;
    }

    public void setDir(boolean dir) {
        isDir = dir;
    }

    public void setModifyTime(long modifyTime) {
        this.modifyTime = modifyTime;
    }

    public void setLength(long length) {
        this.length = length;
    }

    public void setAccessTime(long accessTime) {
        this.accessTime = accessTime;
    }

    public void setBlockReplication(int blockReplication) {
        this.blockReplication = blockReplication;
    }

    public void setBlockSize(long blockSize) {
        this.blockSize = blockSize;
    }

    public void setOwner(String owner) {
        this.owner = owner;
    }

    public void setGroup(String group) {
        this.group = group;
    }

    public void setPermission(short permission) {
        this.permission = permission;
    }
}
