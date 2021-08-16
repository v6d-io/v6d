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

import java.util.Objects;

/** Vineyard Signature definition. */
public class Signature {
    public static Signature InvalidSignature = new Signature(-1L);
    private long id = -1L;

    public Signature(long id) {
        this.id = id;
    }

    public long Value() {
        return this.id;
    }

    public static Signature fromString(String id) {
        return new Signature(Long.parseUnsignedLong(id.substring(1), 16));
    }

    @Override
    public String toString() {
        return String.format("o%016x", id);
    }

    @Override
    public boolean equals(Object other) {
        return this.id == ((Signature) other).id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
