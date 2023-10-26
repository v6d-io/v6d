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
package io.v6d.hive.ql.io;

import java.util.Properties;
import lombok.val;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hive.serde2.AbstractSerDe;
import org.apache.hadoop.hive.serde2.SerDeException;
import org.apache.hadoop.hive.serde2.SerDeStats;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.typeinfo.StructTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils;
import org.apache.hadoop.io.*;

public class VineyardSerDe extends AbstractSerDe {
    private StructTypeInfo rowTypeInfo;
    private TypeInfo[] targetTypeInfos;
    private StructObjectInspector objectInspector;
    private ObjectInspector[] objectInspectors;

    private RecordWrapperWritable.VineyardStructInspector inspector;
    private RecordWrapperWritable writable = new RecordWrapperWritable();

    @Override
    public void initialize(Configuration configuration, Properties tableProperties)
            throws SerDeException {
        initializeTypeInfo(configuration, tableProperties);
    }

    public void initializeTypeInfo(Configuration configuration, Properties tableProperties) {
        this.rowTypeInfo = TypeContext.computeStructTypeInfo(tableProperties);
        this.targetTypeInfos =
                TypeContext.computeTargetTypeInfos(
                        this.rowTypeInfo, ObjectInspectorUtils.ObjectInspectorCopyOption.JAVA);
        this.objectInspector =
                (StructObjectInspector)
                        TypeInfoUtils.getStandardWritableObjectInspectorFromTypeInfo(
                                this.rowTypeInfo);
        this.objectInspectors =
                this.objectInspector.getAllStructFieldRefs().stream()
                        .map(f -> f.getFieldObjectInspector())
                        .toArray(ObjectInspector[]::new);

        this.inspector = new RecordWrapperWritable.VineyardStructInspector(this.rowTypeInfo);
    }

    @Override
    public Class<? extends Writable> getSerializedClass() {
        return RecordWrapperWritable.class;
    }

    @Override
    public RecordWrapperWritable serialize(Object obj, ObjectInspector objInspector) {
        val inspectors =
                ((StructObjectInspector) objInspector)
                        .getAllStructFieldRefs().stream()
                                .map(f -> f.getFieldObjectInspector())
                                .toArray(ObjectInspector[]::new);
        val values =
                copyToStandardObject(
                        (Object[]) obj,
                        inspectors,
                        ObjectInspectorUtils.ObjectInspectorCopyOption.JAVA);
        writable.setValues(values);
        return writable;
    }

    /**
     * Like `ObjectInspectorUtils.copyToStandardObject`, but respects the inspector's size, rather
     * than object's size.
     *
     * <p>Referred from OrcOutputFormat (more specifically, orc.WriterImpl).
     */
    public static Object[] copyToStandardObject(
            Object[] o,
            ObjectInspector[] oi,
            ObjectInspectorUtils.ObjectInspectorCopyOption objectInspectorOption) {
        if (o == null) {
            return null;
        }
        assert (o.length >= oi.length);

        Object[] result = new Object[oi.length];
        for (int i = 0; i < oi.length; i++) {
            result[i] =
                    ObjectInspectorUtils.copyToStandardObject(o[i], oi[i], objectInspectorOption);
        }
        return result;
    }

    @Override
    public Object deserialize(Writable writable) {
        return writable;
    }

    @Override
    public SerDeStats getSerDeStats() {
        return null;
    }

    @Override
    public ObjectInspector getObjectInspector() {
        return this.inspector;
    }
}
