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

import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils.getStandardWritableObjectInspectorFromTypeInfo;

import io.v6d.core.client.Context;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;
import org.apache.hadoop.hive.ql.io.arrow.ArrowWrapperWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.StructTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils;

class VineyardRowWritable extends ArrowWrapperWritable {
    private List<Object> colValues;
    private TypeInfo[] targetTypeInfos;

    public VineyardRowWritable(final List<Object> colValues, StructTypeInfo rowTypeInfo) {
        // long startTime = System.nanoTime();
        long startTime = System.nanoTime();
        this.colValues = colValues;
        StructObjectInspector rowObjectInspector =
                (StructObjectInspector) getStandardWritableObjectInspectorFromTypeInfo(rowTypeInfo);
        final List<? extends StructField> fields = rowObjectInspector.getAllStructFieldRefs();
        final int count = fields.size();
        long endTime = System.nanoTime();
        Context.println("Stage 1 takes " + (endTime - startTime) + " ns");
        targetTypeInfos = new TypeInfo[count];
        for (int i = 0; i < count; i++) {
            final StructField field = fields.get(i);
            final ObjectInspector fieldInspector = field.getFieldObjectInspector();
            final TypeInfo typeInfo =
                    TypeInfoUtils.getTypeInfoFromTypeString(fieldInspector.getTypeName());

            targetTypeInfos[i] = typeInfo;
        }
        Context.println("Stage 2 takes " + (System.nanoTime() - endTime) + " ns");
        // Context.println("Create row writable takes " + (System.nanoTime() - startTime) + " ns");
    }

    public VineyardRowWritable(final List<Object> colValues, TypeInfo[] targetTypeInfos) {
        this.colValues = colValues;
        this.targetTypeInfos = targetTypeInfos;
    }

    public List<Object> getValues() {
        return colValues;
    }

    public TypeInfo[] getTargetTypeInfos() {
        return targetTypeInfos;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        Context.println("write");
        throw new IOException("write is not implemented");
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        Context.println("readFields");
        throw new IOException("readFields is not implemented");
    }
}
