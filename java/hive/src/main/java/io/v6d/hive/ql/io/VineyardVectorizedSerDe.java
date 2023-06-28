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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hive.ql.io.arrow.ArrowColumnarBatchSerDe;
import org.apache.hadoop.hive.serde2.SerDeException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.StructTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils;

public class VineyardVectorizedSerDe extends ArrowColumnarBatchSerDe {
    StructTypeInfo rowTypeInfo;

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
    }

    @Override
    public VineyardRowWritable serialize(Object obj, ObjectInspector objInspector) {
        List<Object> standardObjects = new ArrayList<Object>();
        ObjectInspectorUtils.copyToStandardObject(
                standardObjects,
                obj,
                ((StructObjectInspector) objInspector),
                ObjectInspectorUtils.ObjectInspectorCopyOption.WRITABLE);
        return getRowWritable(standardObjects);
    }

    private VineyardRowWritable getRowWritable(List<Object> standardObjects) {
        VineyardRowWritable rowWritable = new VineyardRowWritable(standardObjects, rowTypeInfo);
        if (rowWritable.getValues() == null) {
            System.out.println("get row writable is null");
        }
        return rowWritable;
    }
}
