/** Copyright 2020-2021 Alibaba Group Holding Limited.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */
package io.v6d.core.client.ds;

import java.io.*;
import org.scijava.nativelib.NativeLoader;

public class ObjectMetaForeign {
    private ObjectMeta metadata;

    static {
        try {
            NativeLoader.loadLibrary("vineyard-core");
        } catch (IOException e) {
            e.printStackTrace();
        }
        // NarSystem.loadLibrary();
    }

    public ObjectMetaForeign(ObjectMeta metadata) {
        this.metadata = metadata;
    }

    public long construct() {
        return constructNative("meta", new long[]{1,2,3,4});
    }

    private native long constructNative(String meta, long [] blobs);
}
