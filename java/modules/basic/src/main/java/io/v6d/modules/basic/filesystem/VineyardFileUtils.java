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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import lombok.val;

public class VineyardFileUtils {
    public static VineyardFileStat getFileStatus(IPCClient client, String path) throws IOException {
        ObjectID fileObjectID;
        try {
            fileObjectID = client.getName(path);
        } catch (ObjectNotExists e) {
            throw new FileNotFoundException(path + " is not found.");
        }
        // File must be migrated if it is not at local.
        ObjectMeta meta = client.getMetaData(fileObjectID, true);
        boolean isDir = meta.getBooleanValue("is_dir_");
        int len = meta.getIntValue("length_");
        long modifyTime = meta.getLongValue("modify_time_");
        long accessTime = meta.getLongValue("access_time_");
        return new VineyardFileStat(
                path, isDir, modifyTime, len, accessTime, 1, 1, null, null, (short) 0777);
    }

    public static VineyardFileStat[] listFileStats(IPCClient client, String path)
            throws IOException {
        List<VineyardFileStat> result = new ArrayList<VineyardFileStat>();
        VineyardFileStat fileStat;
        try {
            fileStat = VineyardFileUtils.getFileStatus(client, path.toString());
        } catch (FileNotFoundException e) {
            throw new FileNotFoundException(path.toString() + " is not found.");
        }
        if (!fileStat.isDir()) {
            // file
            result.add(fileStat);
        } else {
            // dir
            String pattern = "^" + path + "/[^/]*";
            Map<String, ObjectID> objects = Context.getClient().listNames(pattern, true, 255);
            for (val object : objects.entrySet()) {
                ObjectID objectID = object.getValue();
                // File must be migrated if it is not at local.
                ObjectMeta meta = client.getMetaData(objectID, true);
                if (meta.getTypename().compareTo("vineyard::File") != 0) {
                    continue;
                }
                boolean isDir = meta.getBooleanValue("is_dir_");
                int len = meta.getIntValue("length_");
                long modifyTime = meta.getLongValue("modify_time_");
                long accessTime = meta.getLongValue("access_time_");
                VineyardFileStat temp =
                        new VineyardFileStat(
                                object.getKey(),
                                isDir,
                                modifyTime,
                                len,
                                accessTime,
                                1,
                                1,
                                null,
                                null,
                                (short) 0777);
                result.add(temp);
            }
        }
        return result.toArray(new VineyardFileStat[result.size()]);
    }

    public static void printAllFiles(IPCClient client) throws IOException {
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
                // File must be migrated if it is not at local.
                ObjectMeta meta = client.getMetaData(object.getValue(), true);
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
