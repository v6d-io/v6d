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
package io.v6d.modules.basic.columnar;

import io.v6d.modules.basic.arrow.util.ObjectTransformer;
import java.math.BigDecimal;
import java.sql.Date;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateMilliVector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.IntervalDayVector;
import org.apache.arrow.vector.IntervalYearVector;
import org.apache.arrow.vector.LargeVarBinaryVector;
import org.apache.arrow.vector.LargeVarCharVector;
import org.apache.arrow.vector.NullVector;
import org.apache.arrow.vector.SmallIntVector;
import org.apache.arrow.vector.TimeStampMicroTZVector;
import org.apache.arrow.vector.TimeStampMicroVector;
import org.apache.arrow.vector.TimeStampMilliTZVector;
import org.apache.arrow.vector.TimeStampMilliVector;
import org.apache.arrow.vector.TimeStampNanoTZVector;
import org.apache.arrow.vector.TimeStampNanoVector;
import org.apache.arrow.vector.TimeStampSecTZVector;
import org.apache.arrow.vector.TimeStampSecVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.UInt1Vector;
import org.apache.arrow.vector.UInt2Vector;
import org.apache.arrow.vector.UInt4Vector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.ValueVector;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.MapVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.holders.NullableIntervalDayHolder;
import org.apache.arrow.vector.holders.NullableVarCharHolder;
import org.apache.arrow.vector.util.Text;

/** A visitor for arrow arrays. */
public class ColumnarDataBuilder {
    private final ArrowFieldVectorAccessor accessor;

    public ColumnarDataBuilder(FieldVector vector, ObjectTransformer transformer) {
        if (vector instanceof BitVector) {
            accessor = new BooleanAccessor((BitVector) vector, transformer);
        } else if (vector instanceof TinyIntVector) {
            accessor = new ByteAccessor((TinyIntVector) vector, transformer);
        } else if (vector instanceof UInt1Vector) {
            accessor = new UByteAccessor((UInt1Vector) vector, transformer);
        } else if (vector instanceof SmallIntVector) {
            accessor = new ShortAccessor((SmallIntVector) vector, transformer);
        } else if (vector instanceof UInt2Vector) {
            accessor = new UShortAccessor((UInt2Vector) vector, transformer);
        } else if (vector instanceof IntVector) {
            accessor = new IntAccessor((IntVector) vector, transformer);
        } else if (vector instanceof UInt4Vector) {
            accessor = new UIntAccessor((UInt4Vector) vector, transformer);
        } else if (vector instanceof BigIntVector) {
            accessor = new LongAccessor((BigIntVector) vector, transformer);
        } else if (vector instanceof UInt8Vector) {
            accessor = new ULongAccessor((UInt8Vector) vector, transformer);
        } else if (vector instanceof Float4Vector) {
            accessor = new FloatAccessor((Float4Vector) vector, transformer);
        } else if (vector instanceof Float8Vector) {
            accessor = new DoubleAccessor((Float8Vector) vector, transformer);
        } else if (vector instanceof DecimalVector) {
            accessor = new DecimalAccessor((DecimalVector) vector, transformer);
        } else if (vector instanceof VarCharVector) {
            accessor = new StringAccessor((VarCharVector) vector, transformer);
        } else if (vector instanceof LargeVarCharVector) {
            accessor = new LargeStringAccessor((LargeVarCharVector) vector, transformer);
        } else if (vector instanceof VarBinaryVector) {
            accessor = new BinaryAccessor((VarBinaryVector) vector, transformer);
        } else if (vector instanceof LargeVarBinaryVector) {
            accessor = new LargeBinaryAccessor((LargeVarBinaryVector) vector, transformer);
        } else if (vector instanceof DateMilliVector) {
            accessor = new DateAccessor((DateMilliVector) vector, transformer);
        } else if (vector instanceof TimeStampNanoTZVector) {
            accessor = new TimestampNanoAccessor((TimeStampNanoTZVector) vector, transformer);
        } else if (vector instanceof TimeStampNanoVector) {
            accessor = new TimestampNanoNTZAccessor((TimeStampNanoVector) vector, transformer);
        } else if (vector instanceof NullVector) {
            accessor = new NullAccessor((NullVector) vector, null);
        } else if (vector instanceof IntervalYearVector) {
            accessor = new IntervalYearAccessor((IntervalYearVector) vector, transformer);
        } else if (vector instanceof IntervalDayVector) {
            accessor = new IntervalDayAccessor((IntervalDayVector) vector, transformer);
        } else if (vector instanceof ListVector) {
            accessor = new NestedVectorAccessor((ListVector) vector, transformer);
        } else if (vector instanceof StructVector) {
            accessor = new NestedVectorAccessor((StructVector) vector, transformer);
        } else {
            throw new UnsupportedOperationException(
                    "array type is not supported yet: " + vector.getClass());
        }
    }

    public ArrowFieldVectorAccessor getAccessor() {
        return accessor;
    }

    public ValueVector getVector() {
        return accessor.getVector();
    }

    public boolean hasNull() {
        return accessor.getNullCount() > 0;
    }

    public int nullCount() {
        return accessor.getNullCount();
    }

    public int valueCount() {
        return accessor.getValueCount();
    }

    public void close() {
        accessor.close();
    }

    public boolean isNullAt(int rowId) {
        return accessor.isNullAt(rowId);
    }

    public Object getObject(int rowId) {
        return accessor.getObject(rowId);
    }

    public void setObject(int rowId, Object object) {
        accessor.setSafe(rowId, object);
    }

    public boolean getBoolean(int rowId) {
        return (boolean) accessor.getObject(rowId);
    }

    public void setBoolean(int rowId, boolean value) {
        accessor.setObject(rowId, value);
    }

    public byte getByte(int rowId) {
        return (byte) accessor.getObject(rowId);
    }

    public void setByte(int rowId, byte value) {
        accessor.setObject(rowId, value);
    }

    public short getShort(int rowId) {
        return (short) accessor.getObject(rowId);
    }

    public void setShort(int rowId, short value) {
        accessor.setObject(rowId, value);
    }

    public int getInt(int rowId) {
        return (int) accessor.getObject(rowId);
    }

    public void setInt(int rowId, int value) {
        accessor.setObject(rowId, value);
    }

    public long getLong(int rowId) {
        return (long) accessor.getObject(rowId);
    }

    public void setLong(int rowId, long value) {
        accessor.setObject(rowId, value);
    }

    public float getFloat(int rowId) {
        return (float) accessor.getObject(rowId);
    }

    public void setFloat(int rowId, float value) {
        accessor.setObject(rowId, value);
    }

    public double getDouble(int rowId) {
        return (double) accessor.getObject(rowId);
    }

    public void setDouble(int rowId, double value) {
        accessor.setObject(rowId, value);
    }

    public BigDecimal getDecimal(int rowId, int precision, int scale) {
        if (isNullAt(rowId)) {
            return null;
        }
        return (BigDecimal) accessor.getObject(rowId);
    }

    public void setDecimal(int rowId, BigDecimal value) {
        accessor.setObject(rowId, value);
    }

    public Text getUTF8String(int rowId) {
        if (isNullAt(rowId)) {
            return null;
        }
        return new Text(accessor.getObject(rowId).toString());
    }

    public void setUTF8String(int rowId, Text value) {
        accessor.setObject(rowId, value);
    }

    public byte[] getBinary(int rowId) {
        if (isNullAt(rowId)) {
            return null;
        }
        return (byte[]) accessor.getObject(rowId);
    }

    public void setBinary(int rowId, byte[] value) {
        accessor.setObject(rowId, value);
    }

    private static class BooleanAccessor extends ArrowFieldVectorAccessor {

        private final BitVector accessor;
        private final ObjectTransformer transformer;

        BooleanAccessor(BitVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId) == 1;
        }

        @Override
        void setObject(int rowId, Object value) {
            this.accessor.setSafe(rowId, transformer.transformBoolean(value));
        }
    }

    private static class ByteAccessor extends ArrowFieldVectorAccessor {

        private final TinyIntVector accessor;
        private final ObjectTransformer transformer;

        ByteAccessor(TinyIntVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, transformer.transformByte(value));
        }
    }

    private static class UByteAccessor extends ArrowFieldVectorAccessor {

        private final UInt1Vector accessor;
        private final ObjectTransformer transformer;

        UByteAccessor(UInt1Vector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, transformer.transformByte(value));
        }
    }

    private static class ShortAccessor extends ArrowFieldVectorAccessor {

        private final SmallIntVector accessor;
        private final ObjectTransformer transformer;

        ShortAccessor(SmallIntVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, transformer.transformShort(value));
        }
    }

    private static class UShortAccessor extends ArrowFieldVectorAccessor {

        private final UInt2Vector accessor;
        private final ObjectTransformer transformer;

        UShortAccessor(UInt2Vector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, transformer.transformShort(value));
        }
    }

    private static class IntAccessor extends ArrowFieldVectorAccessor {

        private final IntVector accessor;
        private final ObjectTransformer transformer;

        IntAccessor(IntVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, transformer.transformInt(value));
        }
    }

    private static class UIntAccessor extends ArrowFieldVectorAccessor {

        private final UInt4Vector accessor;
        private final ObjectTransformer transformer;

        UIntAccessor(UInt4Vector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, transformer.transformInt(value));
        }
    }

    private static class LongAccessor extends ArrowFieldVectorAccessor {

        private final BigIntVector accessor;
        private final ObjectTransformer transformer;

        LongAccessor(BigIntVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, transformer.transformLong(value));
        }
    }

    private static class ULongAccessor extends ArrowFieldVectorAccessor {

        private final UInt8Vector accessor;
        private final ObjectTransformer transformer;

        ULongAccessor(UInt8Vector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, transformer.transformLong(value));
        }
    }

    private static class FloatAccessor extends ArrowFieldVectorAccessor {

        private final Float4Vector accessor;
        private final ObjectTransformer transformer;

        FloatAccessor(Float4Vector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.accessor.setSafe(rowId, transformer.transformFloat(value));
        }
    }

    private static class DoubleAccessor extends ArrowFieldVectorAccessor {

        private final Float8Vector accessor;
        private final ObjectTransformer transformer;

        DoubleAccessor(Float8Vector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, transformer.transformDouble(value));
        }
    }

    private static class DecimalAccessor extends ArrowFieldVectorAccessor {

        private final DecimalVector accessor;
        private final ObjectTransformer transformer;

        DecimalAccessor(DecimalVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.getObject(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.accessor.setSafe(
                    rowId,
                    transformer.transformDecimal(
                            value, accessor.getPrecision(), accessor.getScale(), 128));
        }
    }

    private static class StringAccessor extends ArrowFieldVectorAccessor {

        private final VarCharVector accessor;
        private final NullableVarCharHolder stringResult = new NullableVarCharHolder();
        private final ObjectTransformer transformer;

        StringAccessor(VarCharVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.getObject(rowId).toString();
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, transformer.transformUtf8(value));
        }
    }

    private static class LargeStringAccessor extends ArrowFieldVectorAccessor {

        private final LargeVarCharVector accessor;
        private final NullableVarCharHolder stringResult = new NullableVarCharHolder();
        private final ObjectTransformer transformer;

        LargeStringAccessor(LargeVarCharVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.getObject(rowId).toString();
        }

        @Override
        void setObject(int rowId, Object value) {
            this.accessor.setSafe(rowId, transformer.transformLargeUtf8(value));
        }
    }

    private static class BinaryAccessor extends ArrowFieldVectorAccessor {

        private final VarBinaryVector accessor;
        private final ObjectTransformer transformer;

        BinaryAccessor(VarBinaryVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.getObject(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, transformer.transformBinary(value));
        }
    }

    private static class LargeBinaryAccessor extends ArrowFieldVectorAccessor {

        private final LargeVarBinaryVector accessor;
        private final ObjectTransformer transformer;

        LargeBinaryAccessor(LargeVarBinaryVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.getObject(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, transformer.transformLargeBinary(value));
        }
    }

    private static class DateAccessor extends ArrowFieldVectorAccessor {

        private final DateMilliVector accessor;
        private final ObjectTransformer transformer;

        DateAccessor(DateMilliVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            long millis = ((Date) transformer.transformDate(value)).getTime();
            accessor.setSafe(rowId, millis);
        }
    }

    private static class TimestampMicroAccessor extends ArrowFieldVectorAccessor {

        private final TimeStampMicroTZVector accessor;
        private final ObjectTransformer transformer;

        TimestampMicroAccessor(TimeStampMicroTZVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, (Long) value);
        }
    }

    private static class TimestampMicroNTZAccessor extends ArrowFieldVectorAccessor {

        private final TimeStampMicroVector accessor;

        TimestampMicroNTZAccessor(TimeStampMicroVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(
                    rowId,
                    (long) (((java.sql.Timestamp) (value)).getTime()) * 1000
                            + (((java.sql.Timestamp) (value)).getNanos() % 1000000) / 1000);
        }
    }

    private static class TimestampMilliAccessor extends ArrowFieldVectorAccessor {

        private final TimeStampMilliTZVector accessor;

        TimestampMilliAccessor(TimeStampMilliTZVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(rowId, (Long) value);
        }
    }

    private static class TimestampMilliNTZAccessor extends ArrowFieldVectorAccessor {

        private final TimeStampMilliVector accessor;

        TimestampMilliNTZAccessor(TimeStampMilliVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.getObject(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.set(rowId, (long) (((java.sql.Timestamp) value).getTime()));
        }
    }

    private static class TimestampSecAccessor extends ArrowFieldVectorAccessor {

        private final TimeStampSecTZVector accessor;

        TimestampSecAccessor(TimeStampSecTZVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return this.accessor.getObject(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.accessor.setSafe(rowId, (Long) value);
        }
    }

    private static class TimestampSecNTZAccessor extends ArrowFieldVectorAccessor {

        private final TimeStampSecVector accessor;

        TimestampSecNTZAccessor(TimeStampSecVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.getObject(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.set(rowId, (long) (((java.sql.Timestamp) value).getTime()) / 1000);
        }
    }

    private static class TimestampNanoAccessor extends ArrowFieldVectorAccessor {

        private final TimeStampNanoTZVector accessor;
        private final ObjectTransformer transformer;

        TimestampNanoAccessor(TimeStampNanoTZVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return this.accessor.getObject(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.accessor.setSafe(rowId, (Long) transformer.transformTimestamp(value).getTime());
        }
    }

    private static class TimestampNanoNTZAccessor extends ArrowFieldVectorAccessor {

        private final TimeStampNanoVector accessor;
        private final ObjectTransformer transformer;

        TimestampNanoNTZAccessor(TimeStampNanoVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            accessor.setSafe(
                    rowId,
                    (long) (((java.sql.Timestamp) transformer.transformTimestamp(value)).getTime())
                                    * DateTimeConstants.NANOS_PER_MILLIS
                            + (((java.sql.Timestamp) transformer.transformTimestamp(value))
                                            .getNanos()
                                    % DateTimeConstants.NANOS_PER_MILLIS));
        }
    }

    private static class NullAccessor extends ArrowFieldVectorAccessor {
        NullAccessor(NullVector vector, ObjectTransformer transformer) {
            super(vector);
        }
    }

    private static class IntervalYearAccessor extends ArrowFieldVectorAccessor {

        private final IntervalYearVector accessor;
        private final ObjectTransformer transformer;

        IntervalYearAccessor(IntervalYearVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.getObject(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.accessor.setSafe(rowId, (Integer) transformer.transform(value));
        }
    }

    private static class IntervalDayAccessor extends ArrowFieldVectorAccessor {

        private final IntervalDayVector accessor;
        private final NullableIntervalDayHolder intervalDayHolder = new NullableIntervalDayHolder();
        private final ObjectTransformer transformer;

        IntervalDayAccessor(IntervalDayVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return Math.addExact(
                    Math.multiplyExact(
                            accessor.getObject(rowId).toDays(), DateTimeConstants.MICROS_PER_DAY),
                    intervalDayHolder.milliseconds * DateTimeConstants.MICROS_PER_MILLIS);
        }

        @Override
        void setObject(int rowId, Object value) {
            intervalDayHolder.milliseconds =
                    (int) Math.floorMod((long) value, DateTimeConstants.MICROS_PER_MILLIS);
            intervalDayHolder.days =
                    (int) Math.floorDiv((long) value, DateTimeConstants.MICROS_PER_DAY);
            accessor.setSafe(rowId, intervalDayHolder);
        }
    }

    private static class NestedVectorAccessor extends ArrowFieldVectorAccessor {

        private final FieldVector accessor;
        private final ObjectTransformer transformer;
        private Map<FieldVector, ColumnarDataBuilder> childBuilders;

        NestedVectorAccessor(FieldVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
            this.childBuilders = new HashMap<>();
            cacheChildVector(accessor);
        }

        private void cacheChildVector(FieldVector vector) {
            if (vector instanceof StructVector) {
                List<FieldVector> childVectors = ((StructVector) vector).getChildrenFromFields();
                for (FieldVector childVector : childVectors) {
                    cacheChildVector(childVector);
                }
            } else if (vector instanceof MapVector) {
                FieldVector childVector = ((MapVector) vector).getDataVector();
                cacheChildVector(childVector);
            } else if (vector instanceof ListVector) {
                FieldVector childVector = ((ListVector) vector).getDataVector();
                cacheChildVector(childVector);
            } else {
                childBuilders.put(vector, new ColumnarDataBuilder(vector, transformer));
            }
        }

        @Override
        void setObject(int rowId, Object value) {
            setObject(accessor, rowId, value);
        }

        private void setObject(FieldVector vector, int rowId, Object value) {
            if (vector instanceof StructVector) {
                ArrayList valueList = (ArrayList) value;
                StructVector structVector = (StructVector) vector;
                List<FieldVector> childVectors = structVector.getChildrenFromFields();
                for (int i = 0; i < valueList.size(); i++) {
                    setObject(childVectors.get(i), rowId, valueList.get(i));
                }
                structVector.setValueCount(structVector.getValueCount() + 1);
            } else if (vector instanceof MapVector) {
                MapVector mapVector = (MapVector) vector;
                if (value == null) {
                    mapVector.setNull(rowId);
                } else {
                    HashMap valueMap = (HashMap) value;
                    List<Object> objects = new ArrayList<>();

                    for (Object key : valueMap.keySet()) {
                        List<Object> temp = new ArrayList<>();
                        temp.add(key);
                        temp.add(valueMap.get(key));
                        objects.add(temp);
                    }

                    mapVector.startNewValue(rowId);
                    int childRowId = mapVector.getDataVector().getValueCount();
                    for (int i = 0; i < objects.size(); i++) {
                        setObject(mapVector.getDataVector(), childRowId + i, objects.get(i));
                    }
                    mapVector.endValue(rowId, objects.size());
                    vector.setValueCount(vector.getValueCount() + objects.size());
                }
            } else if (vector instanceof ListVector) {
                ListVector listVector = (ListVector) vector;
                if (value == null) {
                    listVector.setNull(rowId);
                } else {
                    ArrayList valueList = (ArrayList) value;
                    listVector.startNewValue(rowId);
                    int childRowId = listVector.getDataVector().getValueCount();
                    for (int i = 0; i < valueList.size(); i++) {
                        setObject(listVector.getDataVector(), childRowId + i, valueList.get(i));
                    }
                    listVector.endValue(rowId, valueList.size());
                    vector.setValueCount(vector.getValueCount() + valueList.size());
                }
            } else {
                // Primitive type
                /* 
                 * Due to the fact that FieldVector does not determine the equality of
                 * two FieldVectors by comparing their contained values, doing so will
                 * not incur a significant overhead.
                 */
                ColumnarDataBuilder columnarDataBuilder = childBuilders.get(vector);
                columnarDataBuilder.setObject(rowId, value);
                vector.setValueCount(rowId + 1);
            }
        }
    }

    // refer from spark:
    // https://github.com/apache/spark/blob/master/common/unsafe/src/main/java/org/apache/spark/sql/catalyst/util/DateTimeConstants.java
    private class DateTimeConstants {
        public static final int MONTHS_PER_YEAR = 12;

        public static final byte DAYS_PER_WEEK = 7;

        public static final long HOURS_PER_DAY = 24L;

        public static final long MINUTES_PER_HOUR = 60L;

        public static final long SECONDS_PER_MINUTE = 60L;
        public static final long SECONDS_PER_HOUR = MINUTES_PER_HOUR * SECONDS_PER_MINUTE;
        public static final long SECONDS_PER_DAY = HOURS_PER_DAY * SECONDS_PER_HOUR;

        public static final long MILLIS_PER_SECOND = 1000L;
        public static final long MILLIS_PER_MINUTE = SECONDS_PER_MINUTE * MILLIS_PER_SECOND;
        public static final long MILLIS_PER_HOUR = MINUTES_PER_HOUR * MILLIS_PER_MINUTE;
        public static final long MILLIS_PER_DAY = HOURS_PER_DAY * MILLIS_PER_HOUR;

        public static final long MICROS_PER_MILLIS = 1000L;
        public static final long MICROS_PER_SECOND = MILLIS_PER_SECOND * MICROS_PER_MILLIS;
        public static final long MICROS_PER_MINUTE = SECONDS_PER_MINUTE * MICROS_PER_SECOND;
        public static final long MICROS_PER_HOUR = MINUTES_PER_HOUR * MICROS_PER_MINUTE;
        public static final long MICROS_PER_DAY = HOURS_PER_DAY * MICROS_PER_HOUR;

        public static final long NANOS_PER_MICROS = 1000L;
        public static final long NANOS_PER_MILLIS = MICROS_PER_MILLIS * NANOS_PER_MICROS;
        public static final long NANOS_PER_SECOND = MILLIS_PER_SECOND * NANOS_PER_MILLIS;
    }
}
