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
    private final ArrowVectorAccessor accessor;

    public ColumnarDataBuilder(ValueVector vector, ObjectTransformer transformer) {
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

    public ArrowVectorAccessor getAccessor() {
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
        accessor.setObject(rowId, object);
    }

    public boolean getBoolean(int rowId) {
        return accessor.getBoolean(rowId);
    }

    public void setBoolean(int rowId, boolean value) {
        accessor.setBoolean(rowId, value);
    }

    public byte getByte(int rowId) {
        return accessor.getByte(rowId);
    }

    public void setByte(int rowId, byte value) {
        accessor.setByte(rowId, value);
    }

    public short getShort(int rowId) {
        return accessor.getShort(rowId);
    }

    public void setShort(int rowId, short value) {
        accessor.setShort(rowId, value);
    }

    public int getInt(int rowId) {
        return accessor.getInt(rowId);
    }

    public void setInt(int rowId, int value) {
        accessor.setInt(rowId, value);
    }

    public long getLong(int rowId) {
        return accessor.getLong(rowId);
    }

    public void setLong(int rowId, long value) {
        accessor.setLong(rowId, value);
    }

    public float getFloat(int rowId) {
        return accessor.getFloat(rowId);
    }

    public void setFloat(int rowId, float value) {
        accessor.setFloat(rowId, value);
    }

    public double getDouble(int rowId) {
        return accessor.getDouble(rowId);
    }

    public void setDouble(int rowId, double value) {
        accessor.setDouble(rowId, value);
    }

    public BigDecimal getDecimal(int rowId, int precision, int scale) {
        if (isNullAt(rowId)) {
            return null;
        }
        return accessor.getDecimal(rowId, precision, scale);
    }

    public void setDecimal(int rowId, BigDecimal value) {
        accessor.setDecimal(rowId, value);
    }

    public Text getUTF8String(int rowId) {
        if (isNullAt(rowId)) {
            return null;
        }
        return accessor.getUTF8String(rowId);
    }

    public void setUTF8String(int rowId, Text value) {
        accessor.setUTF8String(rowId, value);
    }

    public byte[] getBinary(int rowId) {
        if (isNullAt(rowId)) {
            return null;
        }
        return accessor.getBinary(rowId);
    }

    public void setBinary(int rowId, byte[] value) {
        accessor.setBinary(rowId, value);
    }

    private abstract static class ArrowVectorAccessor {
        private final ValueVector vector;

        ArrowVectorAccessor(ValueVector vector) {
            this.vector = vector;
        }

        public final ValueVector getVector() {
            return vector;
        }

        final boolean isNullAt(int rowId) {
            return vector.isNull(rowId);
        }

        final int getNullCount() {
            return vector.getNullCount();
        }

        final int getValueCount() {
            return vector.getValueCount();
        }

        final void close() {
            vector.close();
        }

        Object getObject(int rowId) {
            throw new UnsupportedOperationException();
        }

        void setObject(int rowId, Object value) {
            throw new UnsupportedOperationException();
        }

        boolean getBoolean(int rowId) {
            throw new UnsupportedOperationException();
        }

        void setBoolean(int rowId, boolean value) {
            throw new UnsupportedOperationException();
        }

        byte getByte(int rowId) {
            throw new UnsupportedOperationException();
        }

        void setByte(int rowId, byte value) {
            throw new UnsupportedOperationException();
        }

        short getShort(int rowId) {
            throw new UnsupportedOperationException();
        }

        void setShort(int rowId, short value) {
            throw new UnsupportedOperationException();
        }

        int getInt(int rowId) {
            throw new UnsupportedOperationException();
        }

        void setInt(int rowId, int value) {
            throw new UnsupportedOperationException();
        }

        long getLong(int rowId) {
            throw new UnsupportedOperationException();
        }

        void setLong(int rowId, long value) {
            throw new UnsupportedOperationException();
        }

        float getFloat(int rowId) {
            throw new UnsupportedOperationException();
        }

        void setFloat(int rowId, float value) {
            throw new UnsupportedOperationException();
        }

        double getDouble(int rowId) {
            throw new UnsupportedOperationException();
        }

        void setDouble(int rowId, double value) {
            throw new UnsupportedOperationException();
        }

        BigDecimal getDecimal(int rowId, int precision, int scale) {
            throw new UnsupportedOperationException();
        }

        void setDecimal(int rowId, BigDecimal value) {
            throw new UnsupportedOperationException();
        }

        // FIXME: avoid the underlying copy
        Text getUTF8String(int rowId) {
            throw new UnsupportedOperationException();
        }

        void setUTF8String(int rowId, Text value) {
            throw new UnsupportedOperationException();
        }

        byte[] getBinary(int rowId) {
            throw new UnsupportedOperationException();
        }

        void setBinary(int rowId, byte[] value) {
            throw new UnsupportedOperationException();
        }
    }

    private static class BooleanAccessor extends ArrowVectorAccessor {

        private final BitVector accessor;
        private final ObjectTransformer transformer;

        BooleanAccessor(BitVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getBoolean(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value == null) {
                this.accessor.setNull(rowId);
            } else {
                this.accessor.setSafe(rowId, transformer.transformBoolean(value));
            }
        }

        @Override
        final boolean getBoolean(int rowId) {
            return accessor.get(rowId) == 1;
        }

        @Override
        void setBoolean(int rowId, boolean value) {
            accessor.set(rowId, value ? 1 : 0);
        }
    }

    private static class ByteAccessor extends ArrowVectorAccessor {

        private final TinyIntVector accessor;
        private final ObjectTransformer transformer;

        ByteAccessor(TinyIntVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getByte(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value == null) {
                accessor.setNull(rowId);
            } else {
                this.setByte(rowId, transformer.transformByte(value));
            }
        }

        @Override
        final byte getByte(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setByte(int rowId, byte value) {
            accessor.set(rowId, value);
        }
    }

    private static class UByteAccessor extends ArrowVectorAccessor {

        private final UInt1Vector accessor;
        private final ObjectTransformer transformer;

        UByteAccessor(UInt1Vector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getByte(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.setByte(rowId, transformer.transformByte(value));
        }

        @Override
        final byte getByte(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setByte(int rowId, byte value) {
            accessor.set(rowId, value);
        }
    }

    private static class ShortAccessor extends ArrowVectorAccessor {

        private final SmallIntVector accessor;
        private final ObjectTransformer transformer;

        ShortAccessor(SmallIntVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getShort(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value == null) {
                accessor.setNull(rowId);
            } else {
                this.setShort(rowId, transformer.transformShort(value));
            }
        }

        @Override
        final short getShort(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setShort(int rowId, short value) {
            accessor.set(rowId, value);
        }
    }

    private static class UShortAccessor extends ArrowVectorAccessor {

        private final UInt2Vector accessor;
        private final ObjectTransformer transformer;

        UShortAccessor(UInt2Vector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getShort(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.setShort(rowId, transformer.transformShort(value));
        }

        @Override
        final short getShort(int rowId) {
            return (short) accessor.get(rowId);
        }

        @Override
        final void setShort(int rowId, short value) {
            accessor.set(rowId, value);
        }
    }

    private static class IntAccessor extends ArrowVectorAccessor {

        private final IntVector accessor;
        private final ObjectTransformer transformer;

        IntAccessor(IntVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getInt(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value != null) {
                this.setInt(rowId, transformer.transformInt(value));
            } else {
                accessor.setNull(rowId);
            }
        }

        @Override
        final int getInt(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setInt(int rowId, int value) {
            accessor.set(rowId, value);
        }
    }

    private static class UIntAccessor extends ArrowVectorAccessor {

        private final UInt4Vector accessor;
        private final ObjectTransformer transformer;

        UIntAccessor(UInt4Vector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getInt(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.setInt(rowId, transformer.transformInt(value));
        }

        @Override
        final int getInt(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setInt(int rowId, int value) {
            accessor.set(rowId, value);
        }
    }

    private static class LongAccessor extends ArrowVectorAccessor {

        private final BigIntVector accessor;
        private final ObjectTransformer transformer;

        LongAccessor(BigIntVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value == null) {
                accessor.setNull(rowId);
            } else {
                accessor.setSafe(rowId, transformer.transformLong(value));
            }
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setLong(int rowId, long value) {
            accessor.set(rowId, value);
        }
    }

    private static class ULongAccessor extends ArrowVectorAccessor {

        private final UInt8Vector accessor;
        private final ObjectTransformer transformer;

        ULongAccessor(UInt8Vector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.setLong(rowId, transformer.transformLong(value));
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setLong(int rowId, long value) {
            accessor.set(rowId, value);
        }
    }

    private static class FloatAccessor extends ArrowVectorAccessor {

        private final Float4Vector accessor;
        private final ObjectTransformer transformer;

        FloatAccessor(Float4Vector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getFloat(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value == null) {
                this.accessor.setNull(rowId);
            } else {
                this.accessor.setSafe(rowId, transformer.transformFloat(value));
            }
        }

        @Override
        final float getFloat(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setFloat(int rowId, float value) {
            accessor.set(rowId, value);
        }
    }

    private static class DoubleAccessor extends ArrowVectorAccessor {

        private final Float8Vector accessor;
        private final ObjectTransformer transformer;

        DoubleAccessor(Float8Vector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getDouble(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value == null) {
                accessor.setNull(rowId);
            } else {
                this.setDouble(rowId, transformer.transformDouble(value));
            }
        }

        @Override
        final double getDouble(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setDouble(int rowId, double value) {
            accessor.set(rowId, value);
        }
    }

    private static class DecimalAccessor extends ArrowVectorAccessor {

        private final DecimalVector accessor;
        private final ObjectTransformer transformer;

        DecimalAccessor(DecimalVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getDecimal(rowId, 24, 8);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value == null) {
                accessor.setNull(rowId);
            } else {
                this.accessor.setSafe(
                        rowId,
                        transformer.transformDecimal(
                                value, accessor.getPrecision(), accessor.getScale(), 128));
            }
        }

        @Override
        final BigDecimal getDecimal(int rowId, int precision, int scale) {
            return accessor.getObject(rowId);
        }
    }

    private static class StringAccessor extends ArrowVectorAccessor {

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
            if (value == null) {
                accessor.setNull(rowId);
            } else {
                accessor.setSafe(rowId, transformer.transformUtf8(value));
            }
        }
    }

    private static class LargeStringAccessor extends ArrowVectorAccessor {

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
            return getUTF8String(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value == null) {
                accessor.setNull(rowId);
            } else {
                this.accessor.setSafe(rowId, transformer.transformLargeUtf8(value));
            }
        }
    }

    private static class BinaryAccessor extends ArrowVectorAccessor {

        private final VarBinaryVector accessor;
        private final ObjectTransformer transformer;

        BinaryAccessor(VarBinaryVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getBinary(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value == null) {
                accessor.setNull(rowId);
            } else {
                this.setBinary(rowId, transformer.transformBinary(value));
            }
        }

        @Override
        final byte[] getBinary(int rowId) {
            return accessor.getObject(rowId);
        }

        @Override
        final void setBinary(int rowId, byte[] value) {
            accessor.setSafe(rowId, value);
        }
    }

    private static class LargeBinaryAccessor extends ArrowVectorAccessor {

        private final LargeVarBinaryVector accessor;
        private final ObjectTransformer transformer;

        LargeBinaryAccessor(LargeVarBinaryVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getBinary(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.setBinary(rowId, transformer.transformLargeBinary(value));
        }

        @Override
        final byte[] getBinary(int rowId) {
            return accessor.getObject(rowId);
        }

        @Override
        final void setBinary(int rowId, byte[] value) {
            accessor.set(rowId, value);
        }
    }

    private static class DateAccessor extends ArrowVectorAccessor {

        private final DateMilliVector accessor;
        private final ObjectTransformer transformer;

        DateAccessor(DateMilliVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value == null) {
                accessor.setNull(rowId);
            } else {
                long millis = ((Date) transformer.transformDate(value)).getTime();
                accessor.set(rowId, millis);
            }
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setLong(int rowId, long value) {
            accessor.set(rowId, value);
        }
    }

    private static class TimestampMicroAccessor extends ArrowVectorAccessor {

        private final TimeStampMicroTZVector accessor;
        private final ObjectTransformer transformer;

        TimestampMicroAccessor(TimeStampMicroTZVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.setLong(rowId, (Long) value);
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setLong(int rowId, long value) {
            accessor.set(rowId, value);
        }
    }

    private static class TimestampMicroNTZAccessor extends ArrowVectorAccessor {

        private final TimeStampMicroVector accessor;

        TimestampMicroNTZAccessor(TimeStampMicroVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value instanceof Long) {
                this.setLong(rowId, (Long) value);
            } else {
                accessor.set(
                        rowId,
                        (long) (((java.sql.Timestamp) (value)).getTime()) * 1000
                                + (((java.sql.Timestamp) (value)).getNanos() % 1000000) / 1000);
            }
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setLong(int rowId, long value) {
            accessor.set(rowId, value);
        }
    }

    private static class TimestampMilliAccessor extends ArrowVectorAccessor {

        private final TimeStampMilliTZVector accessor;

        TimestampMilliAccessor(TimeStampMilliTZVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.setLong(rowId, (Long) value);
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setLong(int rowId, long value) {
            accessor.set(rowId, value);
        }
    }

    private static class TimestampMilliNTZAccessor extends ArrowVectorAccessor {

        private final TimeStampMilliVector accessor;

        TimestampMilliNTZAccessor(TimeStampMilliVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value == null) {
                accessor.setNull(rowId);
            } else {
                accessor.set(rowId, (long) (((java.sql.Timestamp) value).getTime()));
            }
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setLong(int rowId, long value) {
            accessor.set(rowId, value);
        }
    }

    private static class TimestampSecAccessor extends ArrowVectorAccessor {

        private final TimeStampSecTZVector accessor;

        TimestampSecAccessor(TimeStampSecTZVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.setLong(rowId, (Long) value);
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setLong(int rowId, long value) {
            accessor.set(rowId, value);
        }
    }

    private static class TimestampSecNTZAccessor extends ArrowVectorAccessor {

        private final TimeStampSecVector accessor;

        TimestampSecNTZAccessor(TimeStampSecVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value instanceof Long) {
                this.setLong(rowId, (Long) value);
            } else {
                accessor.set(rowId, (long) (((java.sql.Timestamp) value).getTime()) / 1000);
            }
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setLong(int rowId, long value) {
            accessor.set(rowId, value);
        }
    }

    private static class TimestampNanoAccessor extends ArrowVectorAccessor {

        private final TimeStampNanoTZVector accessor;
        private final ObjectTransformer transformer;

        TimestampNanoAccessor(TimeStampNanoTZVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.setLong(rowId, (Long) transformer.transformTimestamp(value).getTime());
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setLong(int rowId, long value) {
            accessor.set(rowId, value);
        }
    }

    private static class TimestampNanoNTZAccessor extends ArrowVectorAccessor {

        private final TimeStampNanoVector accessor;
        private final ObjectTransformer transformer;

        TimestampNanoNTZAccessor(TimeStampNanoVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            if (value == null) {
                accessor.setNull(rowId);
            } else {
                accessor.setSafe(
                        rowId,
                        (long)
                                                (((java.sql.Timestamp)
                                                                transformer.transformTimestamp(
                                                                        value))
                                                        .getTime())
                                        * DateTimeConstants.NANOS_PER_MILLIS
                                + (((java.sql.Timestamp) transformer.transformTimestamp(value))
                                                .getNanos()
                                        % DateTimeConstants.NANOS_PER_MILLIS));
            }
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setLong(int rowId, long value) {
            accessor.set(rowId, value);
        }
    }

    private static class NullAccessor extends ArrowVectorAccessor {
        NullAccessor(NullVector vector, ObjectTransformer transformer) {
            super(vector);
        }
    }

    private static class IntervalYearAccessor extends ArrowVectorAccessor {

        private final IntervalYearVector accessor;
        private final ObjectTransformer transformer;

        IntervalYearAccessor(IntervalYearVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
        }

        @Override
        Object getObject(int rowId) {
            return getInt(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.setInt(rowId, (Integer) transformer.transform(value));
        }

        @Override
        int getInt(int rowId) {
            return accessor.get(rowId);
        }

        @Override
        final void setInt(int rowId, int value) {
            accessor.set(rowId, value);
        }
    }

    private static class IntervalDayAccessor extends ArrowVectorAccessor {

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
            return getLong(rowId);
        }

        @Override
        void setObject(int rowId, Object value) {
            this.setLong(rowId, (Long) transformer.transform(value));
        }

        @Override
        long getLong(int rowId) {
            accessor.set(rowId, intervalDayHolder);
            return Math.addExact(
                    Math.multiplyExact(intervalDayHolder.days, DateTimeConstants.MICROS_PER_DAY),
                    intervalDayHolder.milliseconds * DateTimeConstants.MICROS_PER_MILLIS);
        }

        @Override
        final void setLong(int rowId, long value) {
            intervalDayHolder.milliseconds =
                    (int) Math.floorMod(value, DateTimeConstants.MICROS_PER_MILLIS);
            intervalDayHolder.days = (int) Math.floorDiv(value, DateTimeConstants.MICROS_PER_DAY);
            accessor.set(rowId, intervalDayHolder);
        }
    }

    private static class NestedVectorAccessor extends ArrowVectorAccessor {

        private final FieldVector accessor;
        private final ObjectTransformer transformer;

        NestedVectorAccessor(FieldVector vector, ObjectTransformer transformer) {
            super(vector);
            this.accessor = vector;
            this.transformer = transformer;
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
                if (value == null) {
                    vector.setNull(rowId);
                    vector.setValueCount(vector.getValueCount() + 1);
                } else if (vector instanceof IntVector) {
                    IntVector intVector = (IntVector) vector;
                    intVector.setSafe(rowId, transformer.transformInt(value));
                    intVector.setValueCount(intVector.getValueCount() + 1);
                } else if (vector instanceof BigIntVector) {
                    BigIntVector bigIntVector = (BigIntVector) vector;
                    bigIntVector.setSafe(rowId, transformer.transformLong(value));
                    bigIntVector.setValueCount(bigIntVector.getValueCount() + 1);
                } else if (vector instanceof SmallIntVector) {
                    SmallIntVector smallIntVector = (SmallIntVector) vector;
                    smallIntVector.setSafe(rowId, transformer.transformShort(value));
                    smallIntVector.setValueCount(smallIntVector.getValueCount() + 1);
                } else if (vector instanceof TinyIntVector) {
                    TinyIntVector tinyIntVector = (TinyIntVector) vector;
                    tinyIntVector.setSafe(rowId, transformer.transformByte(value));
                    tinyIntVector.setValueCount(tinyIntVector.getValueCount() + 1);
                } else if (vector instanceof Float8Vector) {
                    Float8Vector doubleVector = (Float8Vector) vector;
                    doubleVector.setSafe(rowId, transformer.transformDouble(value));
                    doubleVector.setValueCount(doubleVector.getValueCount() + 1);
                } else if (vector instanceof Float4Vector) {
                    Float4Vector floatVector = (Float4Vector) vector;
                    floatVector.setSafe(rowId, transformer.transformFloat(value));
                    floatVector.setValueCount(floatVector.getValueCount() + 1);
                } else if (vector instanceof BitVector) {
                    BitVector bitVector = (BitVector) vector;
                    bitVector.setSafe(rowId, transformer.transformBoolean(value));
                    bitVector.setValueCount(bitVector.getValueCount() + 1);
                } else if (vector instanceof DateMilliVector) {
                    DateMilliVector dateMilliVector = (DateMilliVector) vector;
                    dateMilliVector.setSafe(rowId, (transformer.transformDate(value)).getTime());
                    dateMilliVector.setValueCount(dateMilliVector.getValueCount() + 1);
                } else if (vector instanceof TimeStampNanoVector) {
                    TimeStampNanoVector timeStampNanoVector = (TimeStampNanoVector) vector;
                    timeStampNanoVector.setSafe(
                            rowId,
                            (long) ((transformer.transformTimestamp(value)).getTime())
                                            * DateTimeConstants.NANOS_PER_MILLIS
                                    + ((transformer.transformTimestamp(value)).getNanos()
                                            % DateTimeConstants.NANOS_PER_MILLIS));
                    timeStampNanoVector.setValueCount(timeStampNanoVector.getValueCount() + 1);
                } else if (vector instanceof DecimalVector) {
                    DecimalVector decimalVector = (DecimalVector) vector;
                    BigDecimal bigDecimal =
                            transformer.transformDecimal(
                                    value,
                                    decimalVector.getPrecision(),
                                    decimalVector.getScale(),
                                    128);
                    decimalVector.setSafe(rowId, bigDecimal);
                    decimalVector.setValueCount(decimalVector.getValueCount() + 1);
                } else if (vector instanceof LargeVarCharVector) {
                    LargeVarCharVector largeVarCharVector = (LargeVarCharVector) vector;
                    largeVarCharVector.setSafe(rowId, transformer.transformLargeUtf8(value));
                    largeVarCharVector.setValueCount(largeVarCharVector.getValueCount() + 1);
                } else if (vector instanceof VarCharVector) {
                    VarCharVector varCharVector = (VarCharVector) vector;
                    varCharVector.setSafe(rowId, transformer.transformUtf8(value));
                    varCharVector.setValueCount(varCharVector.getValueCount() + 1);
                } else if (vector instanceof VarBinaryVector) {
                    VarBinaryVector varBinaryVector = (VarBinaryVector) vector;
                    varBinaryVector.setSafe(rowId, transformer.transformBinary(value));
                    varBinaryVector.setValueCount(varBinaryVector.getValueCount() + 1);
                } else if (vector instanceof LargeVarBinaryVector) {
                    LargeVarBinaryVector largeVarBinaryVector = (LargeVarBinaryVector) vector;
                    largeVarBinaryVector.setSafe(rowId, transformer.transformLargeBinary(value));
                    largeVarBinaryVector.setValueCount(largeVarBinaryVector.getValueCount() + 1);
                } else {
                    assert false : "Unsupported vector type:" + vector.getClass().getName();
                }
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
