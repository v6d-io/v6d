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
package io.v6d.core.common.util;

/** Vineyard ObjectID definition. */
public class ObjectID {
    public static ObjectID InvalidObjectID = new ObjectID(-1L);
    private long id = -1L;

    public ObjectID(long id) {
        this.id = id;
    }

    public long Value() {
        return this.id;
    }

    public static ObjectID fromString(String id) {
        return new ObjectID(Long.parseUnsignedLong(id.substring(1), 16));
    }

    @Override
    public String toString() {
        return String.format("o%016x", id);
    }

    @Override
    public boolean equals(Object other) {
        return this.id == ((ObjectID) other).id;
    }
}
