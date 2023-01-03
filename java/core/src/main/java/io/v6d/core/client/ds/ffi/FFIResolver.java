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
package io.v6d.core.client.ds.ffi;

import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;

public abstract class FFIResolver extends ObjectFactory.Resolver {
    public Object resolve(final ObjectMeta metadata) {
        long address = new io.v6d.core.client.ds.ffi.ObjectMeta(metadata).resolve();
        if (address == 0) {
            return null;
        }
        return resolve(metadata, address);
    }

    public abstract Object resolve(final ObjectMeta metadata, long address);
}
