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
package io.v6d.core.client.ds;

import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ObjectFactory {
    private Logger logger = LoggerFactory.getLogger(ObjectFactory.class);

    public abstract static class Resolver {
        public abstract Object resolve(final ObjectMeta metadata);
    }

    private Map<String, Resolver> resolvers;

    private static volatile ObjectFactory factory = null;

    public static ObjectFactory getFactory() {
        ObjectFactory localFactory = factory;
        if (localFactory == null) {
            synchronized (ObjectFactory.class) {
                localFactory = factory;
                if (localFactory == null) {
                    factory = localFactory = new ObjectFactory();
                }
            }
        }
        return localFactory;
    }

    private ObjectFactory() {
        this.resolvers = new HashMap<>();
    }

    public void register(String typename, Resolver resolver) {
        this.resolvers.put(typename, resolver);
    }

    public Object resolve(final ObjectMeta metadata) {
        return this.resolve(metadata.getTypename(), metadata);
    }

    public Object resolve(String typename, final ObjectMeta metadata) {
        if (resolvers.containsKey(typename)) {
            return resolvers.get(typename).resolve(metadata);
        } else {
            throw new RuntimeException("Failed to find resolver for typename " + typename);
        }
    }
}
