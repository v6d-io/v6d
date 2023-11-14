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
package io.v6d.modules.basic.arrow;

import io.v6d.core.client.Client;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.common.util.VineyardException;
import io.v6d.modules.basic.arrow.util.ObjectTransformer;
import io.v6d.modules.basic.columnar.ColumnarDataBuilder;
import org.apache.arrow.vector.FieldVector;

public interface ArrayBuilder extends ObjectBuilder {
    public abstract FieldVector getArray();

    public default ColumnarDataBuilder columnar(ObjectTransformer transformer) {
        return new ColumnarDataBuilder(getArray(), transformer);
    }

    public default ColumnarDataBuilder columnar() {
        return new ColumnarDataBuilder(getArray(), new ObjectTransformer());
    }

    public abstract void shrink(Client client, long size) throws VineyardException;
}
