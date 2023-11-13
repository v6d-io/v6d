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

import io.v6d.modules.basic.columnar.ColumnarData;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import lombok.val;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.SettableStructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.JavaHiveCharObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.JavaHiveDecimalObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.JavaHiveVarcharObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.CharTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.DecimalTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.ListTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.MapTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.PrimitiveTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.StructTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.VarcharTypeInfo;
import org.apache.hadoop.io.*;

public class RecordWrapperWritable implements WritableComparable {
    private Object[] values;

    public RecordWrapperWritable() {}

    public RecordWrapperWritable(Schema schema) {
        this.values = new Object[schema.getFields().size()];
    }

    public RecordWrapperWritable(List<TypeInfo> fieldTypes) {
        this.values = new Object[fieldTypes.size()];
    }

    public void setValue(int index, Object value) {
        values[index] = value;
    }

    public void setValues(Object[] values) {
        this.values = values;
    }

    public void setValues(ColumnarData[] columns, int index) {
        for (int i = 0; i < columns.length; i++) {
            values[i] = columns[i].getObject(index);
        }
    }

    public Object getValue(int index) {
        if (index >= values.length) {
            return null;
        }
        return values[index];
    }

    public Object[] getValues() {
        return values;
    }

    @Override
    public void write(DataOutput out) {
        throw new UnsupportedOperationException("write");
    }

    @Override
    public void readFields(DataInput in) {
        throw new UnsupportedOperationException("readFields");
    }

    @Override
    public int compareTo(Object o) {
        return 0;
    }

    @Override
    public boolean equals(Object o) {
        return true;
    }

    static class Field implements StructField {

        private final String name;
        private final ObjectInspector inspector;
        private final int offset;

        public Field(String name, ObjectInspector inspector, int offset) {
            this.name = name;
            this.inspector = inspector;
            this.offset = offset;
        }

        @Override
        public String getFieldName() {
            return this.name;
        }

        @Override
        public ObjectInspector getFieldObjectInspector() {
            return this.inspector;
        }

        @Override
        public int getFieldID() {
            return this.offset;
        }

        @Override
        public String getFieldComment() {
            return null;
        }
    }

    static class VineyardStructInspector extends SettableStructObjectInspector {
        private List<String> fieldNames;
        private List<TypeInfo> fieldTypes;
        private List<StructField> fields;

        public VineyardStructInspector(StructTypeInfo info) {
            this.fieldNames = info.getAllStructFieldNames();
            this.fieldTypes = info.getAllStructFieldTypeInfos();
            this.fields = new ArrayList<>(fieldNames.size());
            for (int i = 0; i < fieldNames.size(); ++i) {
                this.fields.add(
                        new Field(fieldNames.get(i), createObjectInspector(fieldTypes.get(i)), i));
            }
        }

        @Override
        public String getTypeName() {
            return Arrays.deepToString(fields.toArray());
        }

        @Override
        public Category getCategory() {
            return Category.STRUCT;
        }

        @Override
        public Object create() {
            return new RecordWrapperWritable(fieldTypes);
        }

        @Override
        public Object setStructFieldData(Object struct, StructField field, Object fieldValue) {
            int offset = ((Field) field).offset;
            RecordWrapperWritable writable = (RecordWrapperWritable) struct;
            writable.setValue(offset, fieldValue);
            return struct;
        }

        @Override
        public List<? extends StructField> getAllStructFieldRefs() {
            return fields;
        }

        @Override
        public StructField getStructFieldRef(String fieldName) {
            for (val field : fields) {
                if (field.getFieldName().equalsIgnoreCase(fieldName)) {
                    return field;
                }
            }
            return null;
        }

        @Override
        public Object getStructFieldData(Object data, StructField field) {
            if (data == null) {
                return null;
            }
            int offset = ((Field) field).offset;
            RecordWrapperWritable writable = (RecordWrapperWritable) data;
            return writable.getValue(offset);
        }

        @Override
        public List<Object> getStructFieldsDataAsList(Object data) {
            if (data == null) {
                return null;
            }
            RecordWrapperWritable writable = (RecordWrapperWritable) data;
            return Arrays.asList(writable.getValues());
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            List<? extends StructField> fields = getAllStructFieldRefs();
            sb.append(getClass().getName());
            sb.append("<");
            for (int i = 0; i < fields.size(); i++) {
                if (i > 0) {
                    sb.append(",");
                }
                sb.append(fields.get(i).getFieldObjectInspector().toString());
            }
            sb.append(">");
            return sb.toString();
        }
    }

    static ObjectInspector createObjectInspector(StructTypeInfo info) {
        return new VineyardStructInspector(info);
    }

    static ObjectInspector createObjectInspector(TypeInfo info) {
        return createObjectInspector(info, 0);
    }

    static ObjectInspector createObjectInspector(TypeInfo info, int level) {
        if (info.getCategory() != ObjectInspector.Category.PRIMITIVE) {
            switch (info.getCategory()) {
                case LIST:
                    TypeInfo elementInfo = ((ListTypeInfo) info).getListElementTypeInfo();
                    return ObjectInspectorFactory.getStandardListObjectInspector(
                            createObjectInspector(elementInfo, level + 1));
                case MAP:
                    TypeInfo keyTypeInfo = ((MapTypeInfo) info).getMapKeyTypeInfo();
                    TypeInfo valueTypeInfo = ((MapTypeInfo) info).getMapValueTypeInfo();
                    return ObjectInspectorFactory.getStandardMapObjectInspector(
                            createObjectInspector(keyTypeInfo, level + 1),
                            createObjectInspector(valueTypeInfo, level + 1));
                case STRUCT:
                    List<TypeInfo> elemTypes = ((StructTypeInfo) info).getAllStructFieldTypeInfos();
                    List<String> elemNames = ((StructTypeInfo) info).getAllStructFieldNames();
                    List<ObjectInspector> elemInspectors = new ArrayList<ObjectInspector>();

                    for (int i = 0; i < elemTypes.size(); i++) {
                        elemInspectors.add(createObjectInspector(elemTypes.get(i), level + 1));
                    }
                    return ObjectInspectorFactory.getStandardStructObjectInspector(
                            elemNames, elemInspectors);
                default:
                    throw new UnsupportedOperationException("Unsupported type: " + info);
            }
        } else {
            switch (((PrimitiveTypeInfo) info).getPrimitiveCategory()) {
                case BOOLEAN:
                    return PrimitiveObjectInspectorFactory.javaBooleanObjectInspector;
                case BYTE:
                    return PrimitiveObjectInspectorFactory.javaByteObjectInspector;
                case SHORT:
                    return PrimitiveObjectInspectorFactory.javaShortObjectInspector;
                case INT:
                    return PrimitiveObjectInspectorFactory.javaIntObjectInspector;
                case LONG:
                    return PrimitiveObjectInspectorFactory.javaLongObjectInspector;
                case FLOAT:
                    return PrimitiveObjectInspectorFactory.javaFloatObjectInspector;
                case DOUBLE:
                    return PrimitiveObjectInspectorFactory.javaDoubleObjectInspector;
                case STRING:
                    return PrimitiveObjectInspectorFactory.javaStringObjectInspector;
                case CHAR:
                    return new JavaHiveCharObjectInspector((CharTypeInfo) info);
                case VARCHAR:
                    return new JavaHiveVarcharObjectInspector((VarcharTypeInfo) info);
                case BINARY:
                    return PrimitiveObjectInspectorFactory.javaByteArrayObjectInspector;
                case DATE:
                    return PrimitiveObjectInspectorFactory.javaDateObjectInspector;
                case TIMESTAMP:
                    return PrimitiveObjectInspectorFactory.javaTimestampObjectInspector;
                case DECIMAL:
                    return new JavaHiveDecimalObjectInspector((DecimalTypeInfo) info);
                default:
                    throw new UnsupportedOperationException("Unsupported type: " + info);
            }
        }
    }
}
