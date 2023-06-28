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

import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils.getStandardJavaObjectInspectorFromTypeInfo;

import com.google.common.base.Stopwatch;
import com.google.common.base.StopwatchContext;
import io.v6d.core.client.Context;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import lombok.val;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hive.ql.io.arrow.ArrowColumnarBatchSerDe;
import org.apache.hadoop.hive.serde2.SerDeException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.StructTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils;
import org.apache.hadoop.io.Writable;

public class VineyardArrowSerDe extends ArrowColumnarBatchSerDe {
    StructTypeInfo rowTypeInfo;
    private TypeInfo[] targetTypeInfos;

    private long elements = 0;
    private Stopwatch serializeWatch = StopwatchContext.createUnstarted();
    private Stopwatch deserializeWatch = StopwatchContext.createUnstarted();

    @Override
    public void initialize(Configuration configuration, Properties tableProperties)
            throws SerDeException {
        super.initialize(configuration, tableProperties);
        initializeTypeInfo(configuration, tableProperties);
    }

    @Override
    public void initialize(
            Configuration configuration, Properties tableProperties, Properties partitionProperties)
            throws SerDeException {
        super.initialize(configuration, tableProperties, partitionProperties);
        initializeTypeInfo(configuration, tableProperties);
    }

    public void initializeTypeInfo(Configuration configuration, Properties tableProperties) {
        String columnNameProperty = tableProperties.getProperty("columns");
        String columnTypeProperty = tableProperties.getProperty("columns.types");
        String columnNameDelimiter =
                tableProperties.containsKey("column.name.delimiter")
                        ? tableProperties.getProperty("column.name.delimiter")
                        : String.valueOf(',');
        Object columnNames;
        if (columnNameProperty.length() == 0) {
            columnNames = new ArrayList();
        } else {
            columnNames = Arrays.asList(columnNameProperty.split(columnNameDelimiter));
        }

        ArrayList columnTypes;
        if (columnTypeProperty.length() == 0) {
            columnTypes = new ArrayList();
        } else {
            columnTypes = TypeInfoUtils.getTypeInfosFromTypeString(columnTypeProperty);
        }
        rowTypeInfo =
                (StructTypeInfo) TypeInfoFactory.getStructTypeInfo((List) columnNames, columnTypes);

        StructObjectInspector rowObjectInspector =
                (StructObjectInspector) getStandardJavaObjectInspectorFromTypeInfo(rowTypeInfo);
        final List<? extends StructField> fields = rowObjectInspector.getAllStructFieldRefs();
        final int count = fields.size();
        targetTypeInfos = new TypeInfo[count];
        for (int i = 0; i < count; i++) {
            final StructField field = fields.get(i);
            final ObjectInspector fieldInspector = field.getFieldObjectInspector();
            final TypeInfo typeInfo =
                    TypeInfoUtils.getTypeInfoFromTypeString(fieldInspector.getTypeName());

            targetTypeInfos[i] = typeInfo;
        }
    }

    @Override
    public VineyardRowWritable serialize(Object obj, ObjectInspector objInspector) {
        Context.println("serialize value: obj: " + obj + ", is array: " + obj.getClass().isArray());

        serializeWatch.start();
        List<Object> standardObjects = new ArrayList<Object>();
        ObjectInspectorUtils.copyToStandardObject(
                standardObjects,
                obj,
                ((StructObjectInspector) objInspector),
                ObjectInspectorUtils.ObjectInspectorCopyOption.JAVA);
        val writable = new VineyardRowWritable(standardObjects, targetTypeInfos);
        serializeWatch.stop();
        elements++;
        if (elements % 1000000 == 0) {
            Context.printf("serialize %d elements use %s\n", elements, serializeWatch.toString());
        }
        return writable;
    }

    @Override
    public Object deserialize(Writable writable) {
        deserializeWatch.start();
        // System.out.println("deserialize called");
        val values = ((RowWritable) writable).getValues();
        deserializeWatch.start();
        elements++;
        if (elements % 1000000 == 0) {
            Context.printf(
                    "deserialize %d elements use %s\n", elements, deserializeWatch.toString());
        }
        return values;
    }
}
