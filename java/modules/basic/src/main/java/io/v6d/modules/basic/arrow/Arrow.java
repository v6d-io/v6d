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
package io.v6d.modules.basic.arrow;

import org.apache.arrow.memory.RootAllocator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Arrow {
    public static final RootAllocator default_allocator = new RootAllocator();
    public static final Logger logger = LoggerFactory.getLogger(Arrow.class);

    public static void instantiate() {
        Buffer.instantiate();
        DoubleArray.instantiate();
        FloatArray.instantiate();
        Int32Array.instantiate();
        Int64Array.instantiate();
        RecordBatch.instantiate();
        Schema.instantiate();
    }
}
