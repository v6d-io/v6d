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

import static io.v6d.modules.basic.arrow.Arrow.logger;

import io.v6d.core.client.ds.ObjectFactory;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.modules.basic.arrow.util.SchemaSerializer;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import lombok.SneakyThrows;
import lombok.val;
import org.apache.arrow.util.Collections2;
import org.apache.arrow.vector.types.pojo.Field;

/** Hello world! */
public class Schema extends org.apache.arrow.vector.types.pojo.Schema {
    public static void instantiate() {
        ObjectFactory.getFactory().register("vineyard::SchemaProxy", new SchemaResolver());
    }

    public Schema(List<Field> fields, Map<String, String> metadata) {
        super(Collections2.immutableListCopy(fields), Collections2.immutableMapCopy(metadata));
    }
}

class SchemaResolver extends ObjectFactory.Resolver {
    @Override
    @SneakyThrows(IOException.class)
    public Object resolve(ObjectMeta meta) {
        val buffer = (Buffer) ObjectFactory.getFactory().resolve(meta.getMemberMeta("buffer_"));
        val schema = SchemaSerializer.deserialize(buffer.getBuffer(), Arrow.default_allocator);
        logger.debug("arrow schema is: {}", schema);
        return new Schema(schema.getFields(), schema.getCustomMetadata());
    }
}
