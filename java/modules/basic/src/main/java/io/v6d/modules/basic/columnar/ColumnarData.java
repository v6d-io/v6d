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

/**
 * The implementation is heavily referred from spark, see also
 *
 * <p>https://github.com/apache/spark/blob/master/sql/catalyst/src/main/java/org/apache/spark/sql/vectorized/ArrowColumnVector.java
 *
 * <p>The original file has the following copyright header:
 *
 * <p>* Licensed to the Apache Software Foundation (ASF) under one or more * contributor license
 * agreements. See the NOTICE file distributed with * this work for additional information regarding
 * copyright ownership. * The ASF licenses this file to You under the Apache License, Version 2.0 *
 * (the "License"); you may not use this file except in compliance with * the License. You may
 * obtain a copy of the License at * * http://www.apache.org/licenses/LICENSE-2.0 * * Unless
 * required by applicable law or agreed to in writing, software * distributed under the License is
 * distributed on an "AS IS" BASIS, * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. * See the License for the specific language governing permissions and * limitations
 * under the License.
 */
import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
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
import org.apache.arrow.vector.TimeStampVector;
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

import io.v6d.core.client.Context;
import io.v6d.modules.basic.arrow.util.ArrowVectorUtils;

/** A visitor for arrow arrays. */
public class ColumnarData {
    private final ArrowVectorAccessor accessor;

    public ColumnarData(ValueVector vector) {
        if (vector instanceof BitVector) {
            accessor = new BooleanAccessor((BitVector) vector);
        } else if (vector instanceof TinyIntVector) {
            accessor = new ByteAccessor((TinyIntVector) vector);
        } else if (vector instanceof UInt1Vector) {
            accessor = new UByteAccessor((UInt1Vector) vector);
        } else if (vector instanceof SmallIntVector) {
            accessor = new ShortAccessor((SmallIntVector) vector);
        } else if (vector instanceof UInt2Vector) {
            accessor = new UShortAccessor((UInt2Vector) vector);
        } else if (vector instanceof IntVector) {
            accessor = new IntAccessor((IntVector) vector);
        } else if (vector instanceof UInt4Vector) {
            accessor = new UIntAccessor((UInt4Vector) vector);
        } else if (vector instanceof BigIntVector) {
            accessor = new LongAccessor((BigIntVector) vector);
        } else if (vector instanceof UInt8Vector) {
            accessor = new ULongAccessor((UInt8Vector) vector);
        } else if (vector instanceof Float4Vector) {
            accessor = new FloatAccessor((Float4Vector) vector);
        } else if (vector instanceof Float8Vector) {
            accessor = new DoubleAccessor((Float8Vector) vector);
        } else if (vector instanceof DecimalVector) {
            accessor = new DecimalAccessor((DecimalVector) vector);
        } else if (vector instanceof VarCharVector) {
            accessor = new StringAccessor((VarCharVector) vector);
        } else if (vector instanceof LargeVarCharVector) {
            accessor = new LargeStringAccessor((LargeVarCharVector) vector);
        } else if (vector instanceof VarBinaryVector) {
            accessor = new BinaryAccessor((VarBinaryVector) vector);
        } else if (vector instanceof LargeVarBinaryVector) {
            accessor = new LargeBinaryAccessor((LargeVarBinaryVector) vector);
        } else if (vector instanceof DateDayVector) {
            accessor = new DateAccessor((DateDayVector) vector);
        } else if (vector instanceof TimeStampMicroTZVector) {
            accessor = new TimestampMicroAccessor((TimeStampMicroTZVector) vector);
        } else if (vector instanceof TimeStampMicroVector) {
            accessor = new TimestampMicroNTZAccessor((TimeStampMicroVector) vector);
        } else if (vector instanceof TimeStampSecTZVector) {
            accessor = new TimestampSecAccessor((TimeStampSecTZVector) vector);
        } else if (vector instanceof TimeStampSecVector) {
            accessor = new TimestampSecNTZAccessor((TimeStampSecVector) vector);
        } else if (vector instanceof TimeStampMilliTZVector) {
            accessor = new TimestampMilliAccessor((TimeStampMilliTZVector) vector);
        } else if (vector instanceof TimeStampMilliVector) {
            accessor = new TimestampMilliNTZAccessor((TimeStampMilliVector) vector);
        } else if (vector instanceof TimeStampNanoTZVector) {
            accessor = new TimestampNanoAccessor((TimeStampNanoTZVector) vector);
        } else if (vector instanceof TimeStampNanoVector) {
            accessor = new TimestampNanoNTZAccessor((TimeStampNanoVector) vector);
        } else if (vector instanceof NullVector) {
            accessor = new NullAccessor((NullVector) vector);
        } else if (vector instanceof IntervalYearVector) {
            accessor = new IntervalYearAccessor((IntervalYearVector) vector);
        } else if (vector instanceof IntervalDayVector) {
            accessor = new IntervalDayAccessor((IntervalDayVector) vector);
        } else if (vector instanceof ListVector) {
            accessor = new NestedVectorAccessor((FieldVector)vector);
        } else if (vector instanceof StructVector) {
            accessor = new NestedVectorAccessor((FieldVector) vector);
        } else if (vector instanceof MapVector) {
            accessor = new NestedVectorAccessor((FieldVector) vector);
        } else {
            throw new UnsupportedOperationException(
                    "array type is not supported yet: " + vector.getClass());
        }
    }

    // public ColumnarData(FieldVector []vectors, ArrowTypeID type) {
    //     Context.println("columndata 2");
    //     if (type == ArrowTypeID.List) {
    //         accessor = new ListNestVectorAccessor(vectors);
    //     } else {
    //         throw new UnsupportedOperationException(
    //                 "array type is not supported yet: " + type);
    //     }
    // }

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

    public boolean getBoolean(int rowId) {
        return accessor.getBoolean(rowId);
    }

    public byte getByte(int rowId) {
        return accessor.getByte(rowId);
    }

    public short getShort(int rowId) {
        return accessor.getShort(rowId);
    }

    public int getInt(int rowId) {
        return accessor.getInt(rowId);
    }

    public long getLong(int rowId) {
        return accessor.getLong(rowId);
    }

    public float getFloat(int rowId) {
        return accessor.getFloat(rowId);
    }

    public double getDouble(int rowId) {
        return accessor.getDouble(rowId);
    }

    public BigDecimal getDecimal(int rowId, int precision, int scale) {
        if (isNullAt(rowId)) {
            return null;
        }
        return accessor.getDecimal(rowId, precision, scale);
    }

    public Text getUTF8String(int rowId) {
        if (isNullAt(rowId)) {
            return null;
        }
        return accessor.getUTF8String(rowId);
    }

    public byte[] getBinary(int rowId) {
        if (isNullAt(rowId)) {
            return null;
        }
        return accessor.getBinary(rowId);
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

        boolean getBoolean(int rowId) {
            throw new UnsupportedOperationException();
        }

        byte getByte(int rowId) {
            throw new UnsupportedOperationException();
        }

        short getShort(int rowId) {
            throw new UnsupportedOperationException();
        }

        int getInt(int rowId) {
            throw new UnsupportedOperationException();
        }

        long getLong(int rowId) {
            throw new UnsupportedOperationException();
        }

        float getFloat(int rowId) {
            throw new UnsupportedOperationException();
        }

        double getDouble(int rowId) {
            throw new UnsupportedOperationException();
        }

        BigDecimal getDecimal(int rowId, int precision, int scale) {
            throw new UnsupportedOperationException();
        }

        // FIXME: avoid the underlying copy
        Text getUTF8String(int rowId) {
            throw new UnsupportedOperationException();
        }

        byte[] getBinary(int rowId) {
            throw new UnsupportedOperationException();
        }

        // Object getAllObjects() {
        //     throw new UnsupportedOperationException();
        // }
    }

    private static class BooleanAccessor extends ArrowVectorAccessor {

        private final BitVector accessor;

        BooleanAccessor(BitVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getBoolean(rowId);
        }

        @Override
        final boolean getBoolean(int rowId) {
            return accessor.get(rowId) == 1;
        }
    }

    private static class ByteAccessor extends ArrowVectorAccessor {

        private final TinyIntVector accessor;

        ByteAccessor(TinyIntVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getByte(rowId);
        }

        @Override
        final byte getByte(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class UByteAccessor extends ArrowVectorAccessor {

        private final UInt1Vector accessor;

        UByteAccessor(UInt1Vector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getByte(rowId);
        }

        @Override
        final byte getByte(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class ShortAccessor extends ArrowVectorAccessor {

        private final SmallIntVector accessor;

        ShortAccessor(SmallIntVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getShort(rowId);
        }

        @Override
        final short getShort(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class UShortAccessor extends ArrowVectorAccessor {

        private final UInt2Vector accessor;

        UShortAccessor(UInt2Vector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getShort(rowId);
        }

        @Override
        final short getShort(int rowId) {
            return (short) accessor.get(rowId);
        }
    }

    private static class IntAccessor extends ArrowVectorAccessor {

        private final IntVector accessor;

        IntAccessor(IntVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getInt(rowId);
        }

        @Override
        final int getInt(int rowId) {
            return accessor.get(rowId);
        }

        // @Override
        // Object getAllObjects() {
        //     List<Integer> result = new ArrayList<>();
        //     for (int i = 0; i < accessor.getValueCount(); i++) {
        //         result.add(accessor.get(i));
        //     }
        //     return result;
        // }
    }

    private static class UIntAccessor extends ArrowVectorAccessor {

        private final UInt4Vector accessor;

        UIntAccessor(UInt4Vector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getInt(rowId);
        }

        @Override
        final int getInt(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class LongAccessor extends ArrowVectorAccessor {

        private final BigIntVector accessor;

        LongAccessor(BigIntVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class ULongAccessor extends ArrowVectorAccessor {

        private final UInt8Vector accessor;

        ULongAccessor(UInt8Vector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class FloatAccessor extends ArrowVectorAccessor {

        private final Float4Vector accessor;

        FloatAccessor(Float4Vector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getFloat(rowId);
        }

        @Override
        final float getFloat(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class DoubleAccessor extends ArrowVectorAccessor {

        private final Float8Vector accessor;

        DoubleAccessor(Float8Vector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getDouble(rowId);
        }

        @Override
        final double getDouble(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class DecimalAccessor extends ArrowVectorAccessor {

        private final DecimalVector accessor;

        DecimalAccessor(DecimalVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getDecimal(rowId, 24, 8);
        }

        @Override
        final BigDecimal getDecimal(int rowId, int precision, int scale) {
            return accessor.getObject(rowId);
        }
    }

    private static class StringAccessor extends ArrowVectorAccessor {

        private final VarCharVector accessor;
        private final NullableVarCharHolder stringResult = new NullableVarCharHolder();

        StringAccessor(VarCharVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.getObject(rowId).toString();
        }
    }

    private static class LargeStringAccessor extends ArrowVectorAccessor {

        private final LargeVarCharVector accessor;
        private final NullableVarCharHolder stringResult = new NullableVarCharHolder();

        LargeStringAccessor(LargeVarCharVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getUTF8String(rowId);
        }

        @Override
        final Text getUTF8String(int rowId) {
            return accessor.getObject(rowId);
        }
    }

    private static class BinaryAccessor extends ArrowVectorAccessor {

        private final VarBinaryVector accessor;

        BinaryAccessor(VarBinaryVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getBinary(rowId);
        }

        @Override
        final byte[] getBinary(int rowId) {
            return accessor.getObject(rowId);
        }
    }

    private static class LargeBinaryAccessor extends ArrowVectorAccessor {

        private final LargeVarBinaryVector accessor;

        LargeBinaryAccessor(LargeVarBinaryVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getBinary(rowId);
        }

        @Override
        final byte[] getBinary(int rowId) {
            return accessor.getObject(rowId);
        }
    }

    private static class DateAccessor extends ArrowVectorAccessor {

        private final DateDayVector accessor;

        DateAccessor(DateDayVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getInt(rowId);
        }

        @Override
        final int getInt(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class TimestampMicroAccessor extends ArrowVectorAccessor {

        private final TimeStampMicroTZVector accessor;

        TimestampMicroAccessor(TimeStampMicroTZVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class TimestampMicroNTZAccessor extends ArrowVectorAccessor {

        private final TimeStampMicroVector accessor;

        TimestampMicroNTZAccessor(TimeStampMicroVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
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
        final long getLong(int rowId) {
            return accessor.get(rowId);
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
        final long getLong(int rowId) {
            return accessor.get(rowId);
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
        final long getLong(int rowId) {
            return accessor.get(rowId);
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
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class TimestampNanoAccessor extends ArrowVectorAccessor {

        private final TimeStampNanoTZVector accessor;

        TimestampNanoAccessor(TimeStampNanoTZVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class TimestampNanoNTZAccessor extends ArrowVectorAccessor {

        private final TimeStampNanoVector accessor;

        TimestampNanoNTZAccessor(TimeStampNanoVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class NullAccessor extends ArrowVectorAccessor {

        NullAccessor(NullVector vector) {
            super(vector);
        }
    }

    private static class IntervalYearAccessor extends ArrowVectorAccessor {

        private final IntervalYearVector accessor;

        IntervalYearAccessor(IntervalYearVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getInt(rowId);
        }

        @Override
        int getInt(int rowId) {
            return accessor.get(rowId);
        }
    }

    private static class IntervalDayAccessor extends ArrowVectorAccessor {

        private final IntervalDayVector accessor;
        private final NullableIntervalDayHolder intervalDayHolder = new NullableIntervalDayHolder();

        IntervalDayAccessor(IntervalDayVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return getLong(rowId);
        }

        @Override
        long getLong(int rowId) {
            accessor.get(rowId, intervalDayHolder);
            return Math.addExact(
                    Math.multiplyExact(intervalDayHolder.days, DateTimeConstants.MICROS_PER_DAY),
                    intervalDayHolder.milliseconds * DateTimeConstants.MICROS_PER_MILLIS);
        }
    }

    private static class ListVectorAccessor extends ArrowVectorAccessor {

        private final ListVector accessor;

        ListVectorAccessor(ListVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            Object value = GetNestedObjectUtils.getObject(accessor, rowId);
            Context.println("value type:" + value.getClass().getName() + " value:" + value);
            return value;
        }
    }

    private static class StructVectorAccessor extends ArrowVectorAccessor {

        private final StructVector accessor;

        StructVectorAccessor(StructVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            Object value = GetNestedObjectUtils.getObject(accessor, rowId);
            return value;
        }
    }

    private static class NestedVectorAccessor extends ArrowVectorAccessor {
    
        private final FieldVector vector;

        NestedVectorAccessor(FieldVector vector) {
            super(vector);
            this.vector = vector;
        }

        @Override
        Object getObject(int rowId) {
            Object value = getObject(vector, rowId, 1);
            Context.println("value type:" + value.getClass().getName() + " value:" + value);
            return ((Object[])value)[0];
        }

        private Object getObject(FieldVector vector, int rowId, int rows) {
            List<Object> result = new ArrayList<>();
            Context.println("rows:" + rows);
            if (vector.getValueCount() == 0) {
                return null;
            }

            if (vector instanceof ListVector) {
                Context.println("ListVector");
                /*
                 * TODO:
                 * This code refer to apache arrow. Call ListVector.getObject will trigger Exception.
                 * So we need to get the value by ourself. It may be fixed in the future.
                 */
                if (vector instanceof MapVector) {
                    HashMap<Object, Object> map = new HashMap<>();
                    FieldVector fv = ((ListVector)vector).getDataVector();

                    for (int i = rowId; i < rowId + rows; i++) {
                        List<Object> vals = new ArrayList<>();
                        int start = ((ListVector)vector).getOffsetBuffer().getInt((long)(i * 4));
                        int end = ((ListVector)vector).getOffsetBuffer().getInt((long)((i + 1) * 4));
                        Context.println("start:" + start + " end:" + end);

                        // struct<typeA, typeB>, struct<typeA, typeB> .....
                        Object value = getObject(fv, start, end - start); 
                        for (int j = 0; j < ((Object[])value).length; j++) {
                            // value[j] is struct<typeA, typeB>
                            List<Object> kvList = (List)(((Object[])value)[j]);
                            map.put(kvList.get(0), kvList.get(1));
                        }
                        Context.println("map:" + map);
                    }
                    result.add(map);
                } else {
                    FieldVector fv = ((ListVector)vector).getDataVector();

                    for (int i = rowId; i < rowId + rows; i++) {
                        List<Object> vals = new ArrayList<>();
                        int start = ((ListVector)vector).getOffsetBuffer().getInt((long)(i * 4));
                        int end = ((ListVector)vector).getOffsetBuffer().getInt((long)((i + 1) * 4));
                        Context.println("start:" + start + " end:" + end);

                        Object value = getObject(fv, start, end - start);
                        for (int j = 0; j < ((Object[])value).length; j++) {
                            vals.add(((Object[])value)[j]);
                        }
                        result.add(vals);
                        Context.println("result:" + result);
                    }
                }
            } else if (vector instanceof StructVector) {
                Context.println("StructVector");
                List<FieldVector> childFieldVectors =  ((StructVector)vector).getChildrenFromFields();
                for (int i = rowId; i < rowId + rows; i++) {
                    List<Object> vals = new ArrayList<>();
                    for (int j = 0; j < childFieldVectors.size(); j++) {
                        Object value = getObject(childFieldVectors.get(j), i, 1);
                        for (int k = 0; k < ((Object[])value).length; k++) {
                            vals.add(((Object[])value)[k]);
                        }
                    }
                    result.add(vals);
                }
            } else {
                // primitive type
                Context.println("Primitive vector");
                if (vector instanceof DateDayVector) {
                    for (int i = 0; i < rows; i++) {
                        Date date = new Date(((long)((DateDayVector)vector).get(rowId)) * (DateTimeConstants.MILLIS_PER_DAY));
                        result.add(date);
                    }
                } else if (vector instanceof TimeStampNanoTZVector) {
                    for (int i = 0; i < rows; i++) {
                        long value = ((TimeStampVector)vector).get(rowId + i);
                        Timestamp t = new Timestamp(0);
                        t.setTime((long) value / DateTimeConstants.NANOS_PER_SECOND * DateTimeConstants.NANOS_PER_MICROS);
                        t.setNanos((int) ((long) value % DateTimeConstants.NANOS_PER_SECOND));
                        result.add(t);
                    }
                } else if (vector instanceof DecimalVector) {
                    for (int i = 0; i < rows; i++) {
                        BigDecimal bigDecimal = (BigDecimal)vector.getObject(rowId + i);
                        Context.println("Trans:" + ArrowVectorUtils.TransBigDecimalToHiveDecimal(bigDecimal));
                        result.add(ArrowVectorUtils.TransBigDecimalToHiveDecimal(bigDecimal));
                        Context.println("read:" + vector.getObject(rowId + i));
                    }
                } else if (vector instanceof VarCharVector) {
                    for (int i = 0; i < rows; i++) {
                        result.add(vector.getObject(rowId + i).toString());
                        Context.println("read:" + vector.getObject(rowId + i));
                    }
                } else {
                    for (int i = 0; i < rows; i++) {
                        result.add(vector.getObject(rowId + i));
                        Context.println("read:" + vector.getObject(rowId + i));
                    }
                }
            }
            return result.toArray();
        }
    }

    private static class GetNestedObjectUtils {
        public static Object getObject(ValueVector vector, int index) {
            if (vector.getValueCount() == 0) {
                return null;
            } else {
                if (vector instanceof ListVector) {
                    /*
                     * TODO:
                     * This code refer to apache arrow. Call ListVector.getObject will trigger Exception.
                     * So we need to get the value by ourself. It may be fixed in the future.
                     */
                    java.util.List<Object> vals = new ArrayList<>();
                    int start = ((ListVector)vector).getOffsetBuffer().getInt((long)(index * 4));
                    int end = ((ListVector)vector).getOffsetBuffer().getInt((long)((index + 1) * 4));
                    ValueVector vv = ((ListVector)vector).getDataVector();
                    Context.println("start:" + start + " end:" + end);

                    if (vv instanceof ListVector || vv instanceof StructVector) {
                        for(int i = start; i < end; ++i) {
                            vals.add(getObject(vv, i));
                        }
                    } else {
                        for(int i = start; i < end; ++i) {
                            if (vv instanceof TimeStampVector) {
                                long value = ((TimeStampVector)vv).get(i);
                                Timestamp t = new Timestamp(0);
                                t.setTime((long) value / 1000000000 * 1000);
                                t.setNanos((int) ((long) value % 1000000000));
                                vals.add(t);
                            } else if (vv instanceof DateDayVector) {
                                Date date = new Date((((long)((DateDayVector)vv).get(i))) * (24 * 60 * 60 * 1000));
                                vals.add(date);
                            } 
                            else {
                                vals.add(vv.getObject(i));
                            }
                        }
                    }
                    return vals;
                } else if (vector instanceof StructVector) {
                    Context.println("StructVector");
                    java.util.List<Object> vals = new ArrayList<>();
                    StructVector sv = (StructVector)vector;
                    for (int i = 0; i < sv.getChildrenFromFields().size(); i++) {
                        FieldVector fv = sv.getChildrenFromFields().get(i);
                        if (fv instanceof ListVector || fv instanceof StructVector) {
                            vals.add(getObject(sv.getChildrenFromFields().get(index), i));
                        } else if (fv instanceof DateDayVector) {
                            Context.println("index:" + index);
                            Context.println("day:" + ((DateDayVector)fv).get(index));
                            Context.println("day:" + ((long)((DateDayVector)fv).get(index)) * (24 * 60 * 60 * 1000));
                            Date date = new Date(((long)((DateDayVector)fv).get(index)) * (24 * 60 * 60 * 1000));
                            vals.add(date);
                        } else if (fv instanceof TimeStampNanoTZVector) {
                            long value = ((TimeStampVector)fv).get(index);
                            Timestamp t = new Timestamp(0);
                            t.setTime((long) value / 1000000000 * 1000);
                            t.setNanos((int) ((long) value % 1000000000));
                            vals.add(t);
                        } else {
                            for (int j = 0; j < fv.getValueCount(); j++) {
                                vals.add(fv.getObject(j));
                            }
                        }
                    }
                    return vals;
                } else {
                    return vector.getObject(index);
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
