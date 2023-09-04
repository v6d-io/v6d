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

import io.v6d.modules.basic.arrow.Arrow;
import io.v6d.modules.basic.columnar.ColumnarData;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiConsumer;
import lombok.val;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.SettableStructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.PrimitiveTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.StructTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.*;

public class RowWritable implements Writable {
    private Writable[] values;
    private boolean[] nullIndicators;
    private BiConsumer<Writable, Object>[] setters;

    public RowWritable(Schema schema) {
        this.values = new Writable[schema.getFields().size()];
        this.nullIndicators = new boolean[schema.getFields().size()];
        this.setters = new BiConsumer[schema.getFields().size()];
        for (int i = 0; i < schema.getFields().size(); i++) {
            val dtype = schema.getFields().get(i).getFieldType().getType();
            if (Arrow.Type.Boolean.equals(dtype)) {
                this.values[i] = new BooleanWritable();
                this.setters[i] = RowWritable::setBool;
            } else if (Arrow.Type.Int.equals(dtype) || Arrow.Type.UInt.equals(dtype)) {
                this.values[i] = new IntWritable();
                this.setters[i] = RowWritable::setInt;
            } else if (Arrow.Type.Int64.equals(dtype) || Arrow.Type.UInt64.equals(dtype)) {
                this.values[i] = new LongWritable();
                this.setters[i] = RowWritable::setLong;
            } else if (Arrow.Type.Float.equals(dtype)) {
                this.values[i] = new FloatWritable();
                this.setters[i] = RowWritable::setFloat;
            } else if (Arrow.Type.Double.equals(dtype)) {
                this.values[i] = new DoubleWritable();
                this.setters[i] = RowWritable::setDouble;
            } else if (Arrow.Type.VarChar.equals(dtype)
                    || Arrow.Type.LargeVarChar.equals(dtype)
                    || Arrow.Type.VarBinary.equals(dtype)
                    || Arrow.Type.LargeVarBinary.equals(dtype)) {
                this.values[i] = new Text();
                this.setters[i] = RowWritable::setString;
            } else {
                throw new UnsupportedOperationException("Unsupported type: " + dtype);
            }
        }
    }

    public RowWritable(List<TypeInfo> fieldTypes) {
        this.values = new Writable[fieldTypes.size()];
        this.nullIndicators = new boolean[fieldTypes.size()];
        this.setters = new BiConsumer[fieldTypes.size()];
        for (int i = 0; i < fieldTypes.size(); i++) {
            val info = fieldTypes.get(i);
            if (info.getCategory() != ObjectInspector.Category.PRIMITIVE) {
                throw new UnsupportedOperationException("Unsupported type: " + info);
            }
            switch (((PrimitiveTypeInfo) info).getPrimitiveCategory()) {
                case BOOLEAN:
                    this.values[i] = new BooleanWritable();
                    this.setters[i] = RowWritable::setBool;
                    break;
                case BYTE:
                    this.values[i] = new ByteWritable();
                    this.setters[i] = RowWritable::setByte;
                    break;
                case SHORT:
                    this.values[i] = new ShortWritable();
                    this.setters[i] = RowWritable::setShort;
                    break;
                case INT:
                    this.values[i] = new IntWritable();
                    this.setters[i] = RowWritable::setInt;
                    break;
                case LONG:
                    this.values[i] = new LongWritable();
                    this.setters[i] = RowWritable::setLong;
                    break;
                case FLOAT:
                    this.values[i] = new FloatWritable();
                    this.setters[i] = RowWritable::setFloat;
                    break;
                case DOUBLE:
                    this.values[i] = new DoubleWritable();
                    this.setters[i] = RowWritable::setDouble;
                    break;
                case STRING:
                    this.values[i] = new Text();
                    this.setters[i] = RowWritable::setString;
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported type: " + info);
            }
        }
    }

    public Object[] getValues() {
        return values;
    }

    public void setValues(ColumnarData[] columns, int index) {
        for (int i = 0; i < columns.length; i++) {
            setters[i].accept(values[i], columns[i].getObject(index));
        }
    }

    public Object getValue(int index) {
        return values[index];
    }

    public void setValue(int index, Object value) {
        // n.b.: need to use setters, as values from "setStructFieldData"
        // are already writables.
        values[index] = (Writable) value;
    }

    @Override
    public void write(DataOutput out) {
        System.out.println("write");
    }

    @Override
    public void readFields(DataInput in) {
        System.out.println("readFields");
    }

    private BooleanWritable makeWritable(boolean value) {
        return new BooleanWritable(value);
    }

    private static void setBool(Writable w, Object value) {
        ((BooleanWritable) w).set((boolean) value);
    }

    private static void setByte(Writable w, Object value) {
        ((ByteWritable) w).set((byte) value);
    }

    private static void setShort(Writable w, Object value) {
        ((ShortWritable) w).set((short) value);
    }

    private static void setInt(Writable w, Object value) {
        ((IntWritable) w).set((int) value);
    }

    private static void setLong(Writable w, Object value) {
        ((LongWritable) w).set((long) value);
    }

    private static void setFloat(Writable w, Object value) {
        ((FloatWritable) w).set((float) value);
    }

    private static void setDouble(Writable w, Object value) {
        ((DoubleWritable) w).set((double) value);
    }

    private static void setString(Writable w, Object value) {
        ((Text) w).set(((org.apache.arrow.vector.util.Text) value).toString());
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
            return new RowWritable(fieldTypes);
        }

        @Override
        public Object setStructFieldData(Object struct, StructField field, Object fieldValue) {
            int offset = ((Field) field).offset;
            RowWritable writable = (RowWritable) struct;
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
            RowWritable writable = (RowWritable) data;
            return writable.getValue(offset);
        }

        @Override
        public List<Object> getStructFieldsDataAsList(Object data) {
            if (data == null) {
                return null;
            }
            RowWritable writable = (RowWritable) data;
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
        if (info.getCategory() == ObjectInspector.Category.STRUCT) {
            return createObjectInspector((StructTypeInfo) info);
        }
        if (info.getCategory() != ObjectInspector.Category.PRIMITIVE) {
            throw new UnsupportedOperationException("Unsupported type: " + info);
        }
        switch (((PrimitiveTypeInfo) info).getPrimitiveCategory()) {
            case BOOLEAN:
                return PrimitiveObjectInspectorFactory.writableBooleanObjectInspector;
            case BYTE:
                return PrimitiveObjectInspectorFactory.writableByteObjectInspector;
            case SHORT:
                return PrimitiveObjectInspectorFactory.writableShortObjectInspector;
            case INT:
                return PrimitiveObjectInspectorFactory.writableIntObjectInspector;
            case LONG:
                return PrimitiveObjectInspectorFactory.writableLongObjectInspector;
            case FLOAT:
                return PrimitiveObjectInspectorFactory.writableFloatObjectInspector;
            case DOUBLE:
                return PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
            case STRING:
                return PrimitiveObjectInspectorFactory.writableStringObjectInspector;
            default:
                throw new UnsupportedOperationException("Unsupported type: " + info);
        }
    }
}
