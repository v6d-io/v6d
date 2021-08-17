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

import io.v6d.core.client.ds.ffi.ObjectMeta;

import java.util.HashMap;
import java.util.Map;

public class ObjectFactory {
    public static abstract class Resolver {
        public abstract Object resolve(io.v6d.core.client.ds.ObjectMeta metadata);
    }

    public static abstract class FFIResolver {
        public Object resolve(io.v6d.core.client.ds.ObjectMeta metadata) {
            return resolve(new ObjectMeta(metadata).construct());
        }

        public abstract Object resolve(long address);
    }

    private Map<String, Resolver> resolvers;

    private volatile ObjectFactory factory = null;

    public ObjectFactory getFactory() {
        ObjectFactory localFactory = factory;
        if (localFactory == null) {
            synchronized (this) {
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

    public Object resolve(String typename, io.v6d.core.client.ds.ObjectMeta metadata) {
        if (resolvers.containsKey(typename)) {
            return resolvers.get(typename).resolve(metadata);
        } else {
            throw new RuntimeException("Failed to find resolver for typename " + typename);
        }
    }
}

