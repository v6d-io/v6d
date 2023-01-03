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

import static com.google.common.base.MoreObjects.toStringHelper;
import static java.util.Objects.requireNonNull;

import io.v6d.core.client.Client;
import io.v6d.core.common.util.ObjectID;
import io.v6d.core.common.util.VineyardException;

public abstract class Object implements ObjectBase {
    protected final ObjectID id;
    protected final ObjectMeta meta;

    public Object(ObjectMeta meta) {
        this.meta = requireNonNull(meta, "meta is null");
        this.id = meta.getId();
    }

    public ObjectID getId() {
        return id;
    }

    public ObjectMeta getMeta() {
        return meta;
    }

    public int getNBytes() {
        return meta.getIntValue("nbytes");
    }

    @Override
    public void build(Client client) throws VineyardException {}

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        return this.meta;
    }

    public void persist(Client client) throws VineyardException {}

    public boolean isLocal() {
        return false;
    }

    public boolean isPersist() {
        return false;
    }

    public boolean isGlobal() {
        return false;
    }

    @Override
    public String toString() {
        return toStringHelper(this).add("id", id).add("meta", meta).toString();
    }
}
