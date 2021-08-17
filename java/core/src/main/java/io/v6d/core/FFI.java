/** Copyright 2020-2021 Alibaba Group Holding Limited.
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
package io.v6d.core;

import java.io.IOException;
import lombok.SneakyThrows;
import org.scijava.nativelib.NativeLoader;

public class FFI {
    public static final String JNI_LIBRARY_NAME = "vineyard-core";
    public static final String JNI_LIBRARY_DEFAULT_VERSION = "1.0-SNAPSHOT";

    private static volatile Boolean loaded = null;

    @SneakyThrows(IOException.class)
    public static void loadNativeLibrary() {
        Boolean localLoaded = loaded;
        if (localLoaded == null) {
            synchronized (FFI.class) {
                localLoaded = loaded;
                if (localLoaded == null) {
                    localLoaded = loaded = true;
                    try {
                        NativeLoader.loadLibrary(FFI.JNI_LIBRARY_NAME);
                    } catch (IOException e) {
                        NativeLoader.loadLibrary(resolveVersionedLibraryName());
                    }
                }
            }
        }
    }

    private static String resolveVersionedLibraryName() {
        String version = FFI.class.getPackage().getImplementationVersion();
        if (null == version || version.length() == 0) {
            version = JNI_LIBRARY_DEFAULT_VERSION;
        }
        return JNI_LIBRARY_NAME + "-" + version;
    }
}
