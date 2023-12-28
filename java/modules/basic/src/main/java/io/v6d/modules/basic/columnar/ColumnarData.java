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
import io.v6d.modules.basic.arrow.util.ObjectResolver;
import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Timestamp;
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
public class ColumnarData {
    private final ArrowFieldVectorAccessor accessor;

    public ColumnarData(FieldVector vector) {
        this(vector, new ObjectResolver());
    }

    public ColumnarData(FieldVector vector, ObjectResolver resolver) {
        if (vector instanceof BitVector) {
            accessor = new BooleanAccessor((BitVector) vector, resolver);
        } else if (vector instanceof TinyIntVector) {
            accessor = new ByteAccessor((TinyIntVector) vector, resolver);
        } else if (vector instanceof UInt1Vector) {
            accessor = new UByteAccessor((UInt1Vector) vector, resolver);
        } else if (vector instanceof SmallIntVector) {
            accessor = new ShortAccessor((SmallIntVector) vector, resolver);
        } else if (vector instanceof UInt2Vector) {
            accessor = new UShortAccessor((UInt2Vector) vector, resolver);
        } else if (vector instanceof IntVector) {
            accessor = new IntAccessor((IntVector) vector, resolver);
        } else if (vector instanceof UInt4Vector) {
            accessor = new UIntAccessor((UInt4Vector) vector, resolver);
        } else if (vector instanceof BigIntVector) {
            accessor = new LongAccessor((BigIntVector) vector, resolver);
        } else if (vector instanceof UInt8Vector) {
            accessor = new ULongAccessor((UInt8Vector) vector, resolver);
        } else if (vector instanceof Float4Vector) {
            accessor = new FloatAccessor((Float4Vector) vector, resolver);
        } else if (vector instanceof Float8Vector) {
            accessor = new DoubleAccessor((Float8Vector) vector, resolver);
        } else if (vector instanceof DecimalVector) {
            accessor = new DecimalAccessor((DecimalVector) vector, resolver);
        } else if (vector instanceof VarCharVector) {
            accessor = new StringAccessor((VarCharVector) vector, resolver);
        } else if (vector instanceof LargeVarCharVector) {
            accessor = new LargeStringAccessor((LargeVarCharVector) vector, resolver);
        } else if (vector instanceof VarBinaryVector) {
            accessor = new BinaryAccessor((VarBinaryVector) vector, resolver);
        } else if (vector instanceof LargeVarBinaryVector) {
            accessor = new LargeBinaryAccessor((LargeVarBinaryVector) vector, resolver);
        } else if (vector instanceof DateMilliVector) {
            accessor = new DateAccessor((DateMilliVector) vector, resolver);
        } else if (vector instanceof TimeStampNanoTZVector) {
            accessor = new TimestampNanoAccessor((TimeStampNanoTZVector) vector, resolver);
        } else if (vector instanceof TimeStampNanoVector) {
            accessor = new TimestampNanoNTZAccessor((TimeStampNanoVector) vector, resolver);
        } else if (vector instanceof NullVector) {
            accessor = new NullAccessor((NullVector) vector, resolver);
        } else if (vector instanceof IntervalYearVector) {
            accessor = new IntervalYearAccessor((IntervalYearVector) vector, resolver);
        } else if (vector instanceof IntervalDayVector) {
            accessor = new IntervalDayAccessor((IntervalDayVector) vector, resolver);
        } else if (vector instanceof ListVector) {
            accessor = new NestedVectorAccessor((FieldVector) vector, resolver);
        } else if (vector instanceof StructVector) {
            accessor = new NestedVectorAccessor((FieldVector) vector, resolver);
        } else if (vector instanceof MapVector) {
            accessor = new NestedVectorAccessor((FieldVector) vector, resolver);
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

    public boolean getBoolean(int rowId) {
        return (boolean) accessor.getObject(rowId);
    }

    public Byte getByte(int rowId) {
        return (byte) accessor.getObject(rowId);
    }

    public short getShort(int rowId) {
        return (short) accessor.getObject(rowId);
    }

    public int getInt(int rowId) {
        return (int) accessor.getObject(rowId);
    }

    public long getLong(int rowId) {
        return (long) accessor.getObject(rowId);
    }

    public float getFloat(int rowId) {
        return (float) accessor.getObject(rowId);
    }

    public double getDouble(int rowId) {
        return (double) accessor.getObject(rowId);
    }

    public BigDecimal getDecimal(int rowId, int precision, int scale) {
        if (isNullAt(rowId)) {
            return null;
        }
        return (BigDecimal) accessor.getObject(rowId);
    }

    public Text getUTF8String(int rowId) {
        if (isNullAt(rowId)) {
            return null;
        }
        return new Text(accessor.getObject(rowId).toString());
    }

    public byte[] getBinary(int rowId) {
        if (isNullAt(rowId)) {
            return null;
        }
        return (byte[]) accessor.getObject(rowId);
    }

    private static class BooleanAccessor extends ArrowFieldVectorAccessor {

        private final BitVector accessor;
        private final ObjectResolver resolver;

        BooleanAccessor(BitVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            return resolver.resolveBoolean(accessor.get(rowId));
        }
    }

    private static class ByteAccessor extends ArrowFieldVectorAccessor {

        private final TinyIntVector accessor;
        private final ObjectResolver resolver;

        ByteAccessor(TinyIntVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            return resolver.resolveByte(accessor.getObject(rowId));
        }
    }

    private static class UByteAccessor extends ArrowFieldVectorAccessor {

        private final UInt1Vector accessor;
        private final ObjectResolver resolver;

        UByteAccessor(UInt1Vector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            return resolver.resolveByte(accessor.getObject(rowId));
        }
    }

    private static class ShortAccessor extends ArrowFieldVectorAccessor {

        private final SmallIntVector accessor;
        private final ObjectResolver resolver;

        ShortAccessor(SmallIntVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            return resolver.resolveShort(accessor.getObject(rowId));
        }
    }

    private static class UShortAccessor extends ArrowFieldVectorAccessor {

        private final UInt2Vector accessor;
        private final ObjectResolver resolver;

        UShortAccessor(UInt2Vector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            return resolver.resolveInt(accessor.getObject(rowId));
        }
    }

    private static class IntAccessor extends ArrowFieldVectorAccessor {

        private final IntVector accessor;
        private final ObjectResolver resolver;

        IntAccessor(IntVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            return resolver.resolveInt(accessor.get(rowId));
        }
    }

    private static class UIntAccessor extends ArrowFieldVectorAccessor {

        private final UInt4Vector accessor;
        private final ObjectResolver resolver;

        UIntAccessor(UInt4Vector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            return resolver.resolveInt(accessor.getObject(rowId));
        }
    }

    private static class LongAccessor extends ArrowFieldVectorAccessor {

        private final BigIntVector accessor;
        private final ObjectResolver resolver;

        LongAccessor(BigIntVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            return resolver.resolveLong(accessor.getObject(rowId));
        }
    }

    private static class ULongAccessor extends ArrowFieldVectorAccessor {

        private final UInt8Vector accessor;
        private final ObjectResolver resolver;

        ULongAccessor(UInt8Vector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            return resolver.resolveLong(accessor.getObject(rowId));
        }
    }

    private static class FloatAccessor extends ArrowFieldVectorAccessor {

        private final Float4Vector accessor;
        private final ObjectResolver resolver;

        FloatAccessor(Float4Vector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            return resolver.resolveFloat(accessor.getObject(rowId));
        }
    }

    private static class DoubleAccessor extends ArrowFieldVectorAccessor {

        private final Float8Vector accessor;
        private final ObjectResolver resolver;

        DoubleAccessor(Float8Vector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            return resolver.resolveDouble(accessor.getObject(rowId));
        }
    }

    private static class DecimalAccessor extends ArrowFieldVectorAccessor {

        private final DecimalVector accessor;
        private ObjectResolver resolver;

        DecimalAccessor(DecimalVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            return resolver.resolveDecimal(
                    accessor.getObject(rowId), accessor.getPrecision(), accessor.getScale(), 128);
        }
    }

    private static class StringAccessor extends ArrowFieldVectorAccessor {

        private final VarCharVector accessor;
        private final NullableVarCharHolder stringResult = new NullableVarCharHolder();
        private final ObjectResolver resolver;

        StringAccessor(VarCharVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            return resolver.resolveUtf8(accessor.getObject(rowId).toString());
        }
    }

    private static class LargeStringAccessor extends ArrowFieldVectorAccessor {

        private final LargeVarCharVector accessor;
        private final NullableVarCharHolder stringResult = new NullableVarCharHolder();
        private final ObjectResolver resolver;

        LargeStringAccessor(LargeVarCharVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            return resolver.resolveLargeUtf8(accessor.getObject(rowId).toString());
        }
    }

    private static class BinaryAccessor extends ArrowFieldVectorAccessor {

        private final VarBinaryVector accessor;
        private final ObjectResolver resolver;

        BinaryAccessor(VarBinaryVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            return resolver.resolveBinary(accessor.getObject(rowId));
        }
    }

    private static class LargeBinaryAccessor extends ArrowFieldVectorAccessor {

        private final LargeVarBinaryVector accessor;
        private final ObjectResolver resolver;

        LargeBinaryAccessor(LargeVarBinaryVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            return resolver.resolveLargeBinary(accessor.getObject(rowId));
        }
    }

    private static class DateAccessor extends ArrowFieldVectorAccessor {

        private final DateMilliVector accessor;
        private final ObjectResolver resolver;

        DateAccessor(DateMilliVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            long millis = accessor.get(rowId);
            Date date = new Date(millis);
            return resolver.resolveDate(date);
        }
    }

    private static class TimestampMicroAccessor extends ArrowFieldVectorAccessor {

        private final TimeStampMicroTZVector accessor;

        TimestampMicroAccessor(TimeStampMicroTZVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.getObject(rowId);
        }
    }

    private static class TimestampMicroNTZAccessor extends ArrowFieldVectorAccessor {

        private final TimeStampMicroVector accessor;

        TimestampMicroNTZAccessor(TimeStampMicroVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        Object getObject(int rowId) {
            return accessor.getObject(rowId);
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
            return accessor.getObject(rowId);
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
            if (accessor.isNull(rowId)) {
                return null;
            }
            // mills
            long value = accessor.get(rowId);
            Timestamp t = new Timestamp(value);
            return t;
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
            return accessor.getObject(rowId);
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
    }

    private static class TimestampNanoAccessor extends ArrowFieldVectorAccessor {

        private final TimeStampNanoTZVector accessor;
        private final ObjectResolver resolver;

        TimestampNanoAccessor(TimeStampNanoTZVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            return resolver.resolve(accessor.getObject(rowId));
        }
    }

    private static class TimestampNanoNTZAccessor extends ArrowFieldVectorAccessor {

        private final TimeStampNanoVector accessor;
        private final ObjectResolver resolver;

        TimestampNanoNTZAccessor(TimeStampNanoVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            if (accessor.isNull(rowId)) {
                return null;
            }
            long value = accessor.get(rowId);
            long nano = value % DateTimeConstants.NANOS_PER_SECOND;
            long second = value / DateTimeConstants.NANOS_PER_SECOND;
            if (nano < 0) {
                nano += DateTimeConstants.NANOS_PER_SECOND;
                second -= 1;
            }
            Timestamp t = new Timestamp(second * DateTimeConstants.MILLIS_PER_SECOND);
            t.setNanos((int) nano);
            return resolver.resolveTimestamp(t);
        }
    }

    private static class NullAccessor extends ArrowFieldVectorAccessor {

        NullAccessor(NullVector vector, ObjectResolver resolver) {
            super(vector);
        }
    }

    private static class IntervalYearAccessor extends ArrowFieldVectorAccessor {

        private final IntervalYearVector accessor;
        private final ObjectResolver resolver;

        IntervalYearAccessor(IntervalYearVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            return resolver.resolve(accessor.getObject(rowId));
        }
    }

    private static class IntervalDayAccessor extends ArrowFieldVectorAccessor {

        private final IntervalDayVector accessor;
        private final NullableIntervalDayHolder intervalDayHolder = new NullableIntervalDayHolder();
        private final ObjectResolver resolver;

        IntervalDayAccessor(IntervalDayVector vector, ObjectResolver resolver) {
            super(vector);
            this.accessor = vector;
            this.resolver = resolver;
        }

        @Override
        Object getObject(int rowId) {
            accessor.get(rowId, intervalDayHolder);
            Object value =
                    Math.addExact(
                            Math.multiplyExact(
                                    intervalDayHolder.days, DateTimeConstants.MICROS_PER_DAY),
                            intervalDayHolder.milliseconds * DateTimeConstants.MICROS_PER_MILLIS);
            return resolver.resolve(value);
        }
    }

    private static class NestedVectorAccessor extends ArrowFieldVectorAccessor {
        private final FieldVector vector;
        private final ObjectResolver resolver;
        private Map<FieldVector, ColumnarData> childColumnarData;

        NestedVectorAccessor(FieldVector vector, ObjectResolver resolver) {
            super(vector);
            this.vector = vector;
            this.resolver = resolver;
            this.childColumnarData = new HashMap<>();
            cacheChildVector(vector);
        }

        private void cacheChildVector(FieldVector vector) {
            if (vector instanceof StructVector) {
                List<FieldVector> childFieldVectors =
                        ((StructVector) vector).getChildrenFromFields();
                for (FieldVector childFieldVector : childFieldVectors) {
                    cacheChildVector(childFieldVector);
                }
            } else if (vector instanceof ListVector) {
                FieldVector childFieldVector = ((ListVector) vector).getDataVector();
                cacheChildVector(childFieldVector);
            } else if (vector instanceof MapVector) {
                FieldVector childFieldVector = ((MapVector) vector).getDataVector();
                cacheChildVector(childFieldVector);
            } else {
                // primitive type
                childColumnarData.put(vector, new ColumnarData(vector, resolver));
            }
        }

        @Override
        Object getObject(int rowId) {
            Object value = getObject(vector, rowId, 1);
            if (value == null) {
                return null;
            }
            return ((Object[]) value)[0];
        }

        private Object getObject(FieldVector vector, int rowId, int rows) {
            List<Object> result = new ArrayList<>();
            if (vector.getValueCount() == 0) {
                return null;
            }

            if (vector instanceof ListVector) {
                /*
                 * TODO:
                 * This code refer to apache arrow. Call ListVector.getObject will trigger Exception.
                 * So we need to get the value by ourself. It may be fixed in the future.
                 */
                if (vector instanceof MapVector) {
                    HashMap<Object, Object> map = new HashMap<>();
                    FieldVector fv = ((ListVector) vector).getDataVector();

                    for (int i = rowId; i < rowId + rows; i++) {
                        int start = ((ListVector) vector).getOffsetBuffer().getInt((long) (i * 4));
                        int end =
                                ((ListVector) vector)
                                        .getOffsetBuffer()
                                        .getInt((long) ((i + 1) * 4));

                        // struct<typeA, typeB>, struct<typeA, typeB> .....
                        Object value = getObject(fv, start, end - start);
                        for (int j = 0; j < ((Object[]) value).length; j++) {
                            // value[j] is struct<typeA, typeB>
                            List<Object> kvList = (List) (((Object[]) value)[j]);
                            map.put(kvList.get(0), kvList.get(1));
                        }
                    }
                    result.add(map);
                } else {
                    FieldVector fv = ((ListVector) vector).getDataVector();

                    for (int i = rowId; i < rowId + rows; i++) {
                        if (vector.isNull(i)) {
                            result.add(null);
                        } else {
                            List<Object> vals = new ArrayList<>();
                            int start =
                                    ((ListVector) vector).getOffsetBuffer().getInt((long) (i * 4));
                            int end =
                                    ((ListVector) vector)
                                            .getOffsetBuffer()
                                            .getInt((long) ((i + 1) * 4));

                            Object value = getObject(fv, start, end - start);
                            for (int j = 0; j < ((Object[]) value).length; j++) {
                                vals.add(((Object[]) value)[j]);
                            }
                            result.add(vals);
                        }
                    }
                }
            } else if (vector instanceof StructVector) {
                List<FieldVector> childFieldVectors =
                        ((StructVector) vector).getChildrenFromFields();
                for (int i = rowId; i < rowId + rows; i++) {
                    List<Object> vals = new ArrayList<>();
                    for (int j = 0; j < childFieldVectors.size(); j++) {
                        Object value = getObject(childFieldVectors.get(j), i, 1);
                        for (int k = 0; k < ((Object[]) value).length; k++) {
                            vals.add(((Object[]) value)[k]);
                        }
                    }
                    result.add(vals);
                }
            } else {
                // primitive type
                ColumnarData columnarData = childColumnarData.get(vector);
                for (int i = 0; i < rows; i++) {
                    result.add(columnarData.getObject(rowId + i));
                }
            }
            return result.toArray();
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
