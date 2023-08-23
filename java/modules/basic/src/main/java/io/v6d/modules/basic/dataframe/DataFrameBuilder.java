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
package io.v6d.modules.basic.arrow;

import static java.util.Objects.requireNonNull;

import io.v6d.core.client.Client;
import io.v6d.core.client.IPCClient;
import io.v6d.core.client.ds.ObjectBase;
import io.v6d.core.client.ds.ObjectBuilder;
import io.v6d.core.client.ds.ObjectMeta;
import io.v6d.core.common.util.VineyardException;
import java.util.HashMap;
import lombok.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataFrameBuilder implements ObjectBuilder {
    private Logger logger = LoggerFactory.getLogger(DataFrameBuilder.class);

    private final HashMap<String, ObjectBase> columns;

    public DataFrameBuilder(final IPCClient client) {
      this.columns = new HashMap<String, ObjectBase>();
    }

    public void addColumn(String name, Tensor column) {
      this.columns.put(name, column);
    }

    public void addColumn(String name, TensorBuilder builder) {
      this.columns.put(name, builder);
    }

    public HashMap<String, ObjectBase> getColumns() {
      return this.columns;
    }

    public int getColumnSize() {
      return this.columns.size();
    }

    @Override
    public void build(Client client) throws VineyardException {}

    @Override
    public ObjectMeta seal(Client client) throws VineyardException {
        this.build(client);

        val meta = ObjectMeta.empty();

        meta.setTypename("vineyard::DataFrame");
        meta.setValue("nbytes", 0); // FIXME
        meta.setValue("partition_index_row_", -1); // FIXME
        meta.setValue("partition_index_column_", -1); //FIXME
        meta.setValue("row_batch_index_", 0);

        meta.setValue("__values_-size", columns.size());
        int index = 0;
        for(String key : columns.keySet()) {
          meta.addValue("__values_-key-" + index, key);
          meta.addMember("__values_-value-" + index, columns.get(key).seal(client));
          index ++;
        }
        for(int index = 0; index < columns.size(); ++index) {
          // Seal Column Names.
        }
        return client.createMetaData(meta);
    }
}

*/
