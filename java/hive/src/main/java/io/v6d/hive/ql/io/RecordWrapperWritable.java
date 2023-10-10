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

import io.v6d.core.client.Context;
import io.v6d.modules.basic.arrow.Arrow;
import io.v6d.modules.basic.columnar.ColumnarData;
import java.io.DataInput;
import java.io.DataOutput;
import java.math.BigDecimal;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.function.BiConsumer;
import lombok.val;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.types.pojo.ArrowType.Decimal;
import org.apache.hadoop.hive.common.type.HiveDecimal;
import org.apache.hadoop.hive.serde2.io.DateWritable;
import org.apache.hadoop.hive.serde2.io.HiveDecimalWritable;
import org.apache.hadoop.hive.serde2.io.TimestampWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.SettableStructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.JavaHiveCharObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.JavaHiveDecimalObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.JavaHiveVarcharObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.WritableHiveDecimalObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.BaseCharTypeInfo;
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
    // output format: raw java objects, e.g., integer, string
    // input format: writable
    private Object[] values;

    // for input format
    private boolean[] nullIndicators;
    private BiConsumer<Writable, Object>[] setters;

    // for output format
    public RecordWrapperWritable() {}

    // for input format
    public RecordWrapperWritable(Schema schema) {
        this.values = new Object[schema.getFields().size()];
        this.nullIndicators = new boolean[schema.getFields().size()];
        this.setters = new BiConsumer[schema.getFields().size()];
        for (int i = 0; i < schema.getFields().size(); i++) {
            val dtype = schema.getFields().get(i).getFieldType().getType();
            if (Arrow.Type.Boolean.equals(dtype)) {
                this.values[i] = new BooleanWritable();
                this.setters[i] = RecordWrapperWritable::setBool;
            } else if (Arrow.Type.Int.equals(dtype) || Arrow.Type.UInt.equals(dtype)) {
                this.values[i] = new IntWritable();
                this.setters[i] = RecordWrapperWritable::setInt;
            } else if (Arrow.Type.Int64.equals(dtype) || Arrow.Type.UInt64.equals(dtype)) {
                this.values[i] = new LongWritable();
                this.setters[i] = RecordWrapperWritable::setLong;
            } else if (Arrow.Type.Float.equals(dtype)) {
                this.values[i] = new FloatWritable();
                this.setters[i] = RecordWrapperWritable::setFloat;
            } else if (Arrow.Type.Double.equals(dtype)) {
                this.values[i] = new DoubleWritable();
                this.setters[i] = RecordWrapperWritable::setDouble;
            } else if (Arrow.Type.VarChar.equals(dtype) || Arrow.Type.LargeVarChar.equals(dtype)) {
                continue;
            } else if (Arrow.Type.VarBinary.equals(dtype)) {
                this.values[i] = new BytesWritable();
                this.setters[i] = RecordWrapperWritable::setBytes;
            } else if (Arrow.Type.TinyInt.equals(dtype)) {
                this.values[i] = new ByteWritable();
                this.setters[i] = RecordWrapperWritable::setByte;
            } else if (Arrow.Type.SmallInt.equals(dtype)) {
                this.values[i] = new org.apache.hadoop.hive.serde2.io.ShortWritable();
                this.setters[i] = RecordWrapperWritable::setShort;
            } else if (Arrow.Type.Date.equals(dtype)) {
                this.values[i] = new DateWritable();
                this.setters[i] = RecordWrapperWritable::setDate;
            } else if (Arrow.Type.TimeStampMicro.equals(dtype)
                    || Arrow.Type.TimeStampMilli.equals(dtype)
                    || Arrow.Type.TimeStampNano.equals(dtype)
                    || Arrow.Type.TimeStampSec.equals(dtype)) {
                this.values[i] = new TimestampWritable();
                this.setters[i] = RecordWrapperWritable::setTimestamp;
            } else if (dtype instanceof Decimal) {
                this.values[i] = new HiveDecimalWritable();
                this.setters[i] = RecordWrapperWritable::setDecimal;
            } else if (Arrow.Type.List.equals(dtype)) {
                this.values = new List[schema.getFields().size()];
                break;
            } else if (Arrow.Type.Struct.equals(dtype)) {
                this.values = new List[schema.getFields().size()];
            } else if (Arrow.Type.Map.equals(dtype)) {
                this.values = new HashMap[schema.getFields().size()];
            } else {
                throw new UnsupportedOperationException("Unsupported type: " + dtype);
            }
        }
    }

    // for input format
    public RecordWrapperWritable(List<TypeInfo> fieldTypes) {
        this.values = new Object[fieldTypes.size()];
        this.nullIndicators = new boolean[fieldTypes.size()];
        this.setters = new BiConsumer[fieldTypes.size()];
        for (int i = 0; i < fieldTypes.size(); i++) {
            val info = fieldTypes.get(i);
            if (info.getCategory() != ObjectInspector.Category.PRIMITIVE) {
                switch (info.getCategory()) {
                    case LIST:
                        this.values = new List[fieldTypes.size()];
                        break;
                    case MAP:
                        this.values = new HashMap[fieldTypes.size()];
                        break;
                    case STRUCT:
                        this.values = new List[fieldTypes.size()];
                        break;
                    case UNION:
                        Context.println("type:" + info);
                        throw new UnsupportedOperationException("Unsupported type: " + info);
                    default:
                        throw new UnsupportedOperationException("Unsupported type: " + info);
                }
                break;
            } else {
                switch (((PrimitiveTypeInfo) info).getPrimitiveCategory()) {
                    case BOOLEAN:
                        this.values[i] = new BooleanWritable();
                        this.setters[i] = RecordWrapperWritable::setBool;
                        break;
                    case BYTE:
                        this.values[i] = new ByteWritable();
                        this.setters[i] = RecordWrapperWritable::setByte;
                        break;
                    case SHORT:
                        this.values[i] = new ShortWritable();
                        this.setters[i] = RecordWrapperWritable::setShort;
                        break;
                    case INT:
                        this.values[i] = new IntWritable();
                        this.setters[i] = RecordWrapperWritable::setInt;
                        break;
                    case LONG:
                        this.values[i] = new LongWritable();
                        this.setters[i] = RecordWrapperWritable::setLong;
                        break;
                    case FLOAT:
                        this.values[i] = new FloatWritable();
                        this.setters[i] = RecordWrapperWritable::setFloat;
                        break;
                    case DOUBLE:
                        this.values[i] = new DoubleWritable();
                        this.setters[i] = RecordWrapperWritable::setDouble;
                        break;
                    case BINARY:
                        this.values[i] = new BytesWritable();
                        this.setters[i] = RecordWrapperWritable::setBytes;
                        break;
                    case CHAR:
                    case VARCHAR:
                    case STRING:
                        break;
                    case DATE:
                        this.values[i] = new DateWritable();
                        this.setters[i] = RecordWrapperWritable::setDate;
                        break;
                    case TIMESTAMP:
                        this.values[i] = new TimestampWritable();
                        this.setters[i] = RecordWrapperWritable::setTimestamp;
                        break;
                    case DECIMAL:
                        this.values[i] = new HiveDecimalWritable();
                        this.setters[i] = RecordWrapperWritable::setDecimal;
                        break;
                    default:
                        throw new UnsupportedOperationException("Unsupported type: " + info);
                }
            }
        }
    }

    public Object[] getValues() {
        return values;
    }

    // for output format
    public void setValues(Object[] values) {
        Context.println("setValues");
        this.values = values;
    }

    // for input format
    public void setWritables(ColumnarData[] columns, int index) {
        Context.println("setWritables");
        for (int i = 0; i < columns.length; i++) {
            if (values[i] instanceof Writable) {
                setters[i].accept((Writable) values[i], columns[i].getObject(index));
            } else {
                values[i] = columns[i].getObject(index);
            }
        }
    }

    public Object getValue(int index) {
        if (index >= values.length) {
            return null;
        }
        return values[index];
    }

    // for output format
    public void setValue(int index, Object value) {
        Context.println("setValue, class:" + value.getClass().getName() + ", value:" + value);
        values[index] = value;
        if (value instanceof Object[]) {
            Object[] objects = (Object[])value;
            for (int i = 0; i < objects.length; i++) {
                Context.println("objects[" + i + "]=" + objects[i]);
            }
        } else {
            Context.println("objects[" + index + "]=" + value);
        }
    }

    // for input format
    public void setWritable(int index, Writable value) {
        // n.b.: need to use setters, as values from "setStructFieldData"
        // are already writables.
        values[index] = value;
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

    private static void setBool(Writable w, Object value) {
        ((BooleanWritable) w).set((boolean) value);
    }

    private static void setByte(Writable w, Object value) {
        ((ByteWritable) w).set((byte) value);
    }

    private static void setBytes(Writable w, Object value) {
        ((BytesWritable) w).set((byte[]) value, 0, ((byte[]) value).length);
    }

    private static void setShort(Writable w, Object value) {
        ((org.apache.hadoop.hive.serde2.io.ShortWritable) w).set((short) value);
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

    private static void setDate(Writable w, Object value) {
        ((DateWritable) w).set((int)value);
    }

    private static void setTimestamp(Writable w, Object value) {
        Timestamp t = new Timestamp(0);
        t.setTime((long) value / 1000000000 * 1000);
        t.setNanos((int) ((long) value % 1000000000));
        ((TimestampWritable) w).set(t);
    }

    private static void setDecimal(Writable w, Object value) {
        BigDecimal decimal = (BigDecimal) value;
        HiveDecimal hiveDecimal = HiveDecimal.create(decimal);
        ((HiveDecimalWritable) w).set(hiveDecimal, decimal.precision(), decimal.scale());
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
            List<String> names = info.getAllStructFieldNames();
            for (int i = 0; i < names.size(); i++) {
                Context.println("name:" + names.get(i) + ", type:" + fieldTypes.get(i));
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
                    Context.println("type1:" + info);
                    TypeInfo elementInfo = ((ListTypeInfo)info).getListElementTypeInfo();
                    return ObjectInspectorFactory.getStandardListObjectInspector(createObjectInspector(elementInfo, level + 1));
                case MAP:
                    Context.println("type2:" + info);
                    TypeInfo keyTypeInfo = ((MapTypeInfo)info).getMapKeyTypeInfo();
                    TypeInfo valueTypeInfo = ((MapTypeInfo)info).getMapValueTypeInfo();
                    return ObjectInspectorFactory.getStandardMapObjectInspector(createObjectInspector(keyTypeInfo, level + 1), createObjectInspector(valueTypeInfo, level + 1));
                case STRUCT:
                    Context.println("type3:" + info);
                    List<TypeInfo> elemTypes = ((StructTypeInfo)info).getAllStructFieldTypeInfos();
                    List<String> elemNames = ((StructTypeInfo)info).getAllStructFieldNames();
                    List<ObjectInspector> elemInspectors = new ArrayList<ObjectInspector>();
                    
                    for (int i = 0; i < elemTypes.size(); i++) {
                        elemInspectors.add(createObjectInspector(elemTypes.get(i), level + 1));
                    }
                    return ObjectInspectorFactory.getStandardStructObjectInspector(elemNames, elemInspectors);
                case UNION:
                    Context.println("type4:" + info);
                    throw new UnsupportedOperationException("Unsupported type: " + info);
                default:
                    throw new UnsupportedOperationException("Unsupported type: " + info);
            }
        } else {
            switch (((PrimitiveTypeInfo) info).getPrimitiveCategory()) {
                case BOOLEAN:
                    if (level == 0) {
                        return PrimitiveObjectInspectorFactory.writableBooleanObjectInspector;
                    } else {
                        return PrimitiveObjectInspectorFactory.javaBooleanObjectInspector;
                    }
                case BYTE:
                    if (level == 0) {
                        return PrimitiveObjectInspectorFactory.writableByteObjectInspector;
                    } else {
                        return PrimitiveObjectInspectorFactory.javaByteObjectInspector;
                    }
                case SHORT:
                    if (level == 0) {
                        return PrimitiveObjectInspectorFactory.writableShortObjectInspector;
                    } else {
                        return PrimitiveObjectInspectorFactory.javaShortObjectInspector;
                    }
                case INT:
                    if (level == 0) {
                        return PrimitiveObjectInspectorFactory.writableIntObjectInspector;
                    } else {
                        return PrimitiveObjectInspectorFactory.javaIntObjectInspector;
                    }
                case LONG:
                    if (level == 0) {
                        return PrimitiveObjectInspectorFactory.writableLongObjectInspector;
                    } else {
                        return PrimitiveObjectInspectorFactory.javaLongObjectInspector;
                    }
                case FLOAT:
                    if (level == 0) {
                        return PrimitiveObjectInspectorFactory.writableFloatObjectInspector;
                    } else {
                        return PrimitiveObjectInspectorFactory.javaFloatObjectInspector;
                    }
                case DOUBLE:
                    if (level == 0) {
                        return PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
                    } else {
                        return PrimitiveObjectInspectorFactory.javaDoubleObjectInspector;
                    }
                case STRING:
                    return PrimitiveObjectInspectorFactory.javaStringObjectInspector;
                case CHAR:
                    return new JavaHiveCharObjectInspector((CharTypeInfo) info);
                case VARCHAR:
                    return new JavaHiveVarcharObjectInspector((VarcharTypeInfo) info);
                case BINARY:
                    if (level == 0) {
                        return PrimitiveObjectInspectorFactory.writableBinaryObjectInspector;
                    } else {
                        return PrimitiveObjectInspectorFactory.javaByteArrayObjectInspector;
                    }
                case DATE:
                    if (level == 0) {
                        return PrimitiveObjectInspectorFactory.writableDateObjectInspector;
                    } else {
                        return PrimitiveObjectInspectorFactory.javaDateObjectInspector;
                    }
                case TIMESTAMP:
                    if (level == 0) {
                        return PrimitiveObjectInspectorFactory.writableTimestampObjectInspector;
                    } else {
                        return PrimitiveObjectInspectorFactory.javaTimestampObjectInspector;
                    }
                case DECIMAL:
                    if (level == 0) {
                        WritableHiveDecimalObjectInspector oi = new WritableHiveDecimalObjectInspector((DecimalTypeInfo)info);
                        Context.println("Create io:" + oi.getTypeName());
                        Context.println("Typeinfo :" + info.getTypeName());
                        return oi;
                    } else {
                        ObjectInspector oi = new JavaHiveDecimalObjectInspector((DecimalTypeInfo)info);
                        return oi;
                    }
                default:
                    throw new UnsupportedOperationException("Unsupported type: " + info);
            }
        }
    }
}
