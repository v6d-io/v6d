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
package io.v6d.modules.basic.scalar;

import io.v6d.core.client.ds.Object;
import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;

public class FloatValue extends Object {
    private final float value;

    public static void instantiate() {
        ObjectFactory.getFactory().register("vineyard::Scalar<float>", new FloatValueResolver());
    }

    public FloatValue(final ObjectMeta meta, final float value) {
        super(meta);
        this.value = value;
    }

    public float getValue() {
        return value;
    }

    @Override
    public String toString() {
        return String.valueOf(value);
    }
}

class FloatValueResolver extends ObjectFactory.Resolver {

    @Override
    public Object resolve(ObjectMeta meta) {
        return new FloatValue(meta, meta.getFloatValue("value_"));
    }
}
