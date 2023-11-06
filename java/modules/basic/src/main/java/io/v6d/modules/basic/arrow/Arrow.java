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
package io.v6d.modules.basic.arrow;

import com.google.common.collect.ImmutableList;
import io.v6d.modules.basic.stream.RecordBatchStream;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Arrow {
    public static final RootAllocator default_allocator = new RootAllocator();
    public static final Logger logger = LoggerFactory.getLogger(Arrow.class);

    public static class Type {
        public static final ArrowType Null = new ArrowType.Null();
        public static final ArrowType TinyInt = new ArrowType.Int(8, true);
        public static final ArrowType SmallInt = new ArrowType.Int(16, true);
        public static final ArrowType Int = new ArrowType.Int(32, true);
        public static final ArrowType UInt = new ArrowType.Int(32, false);
        public static final ArrowType Int64 = new ArrowType.Int(64, true);
        public static final ArrowType UInt64 = new ArrowType.Int(64, false);
        public static final ArrowType Float =
                new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE);
        public static final ArrowType Double =
                new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
        public static final ArrowType Boolean = new ArrowType.Bool();
        public static final ArrowType VarChar = new ArrowType.Utf8();
        public static final ArrowType LargeVarChar = new ArrowType.LargeUtf8();
        public static final ArrowType VarBinary = new ArrowType.Binary();
        public static final ArrowType LargeVarBinary = new ArrowType.LargeBinary();
        public static final ArrowType Date = new ArrowType.Date(DateUnit.DAY);
        public static final ArrowType TimeStampSec =
                new ArrowType.Timestamp(TimeUnit.SECOND, "UTC");
        public static final ArrowType TimeStampMilli =
                new ArrowType.Timestamp(TimeUnit.MILLISECOND, "UTC");
        public static final ArrowType TimeStampMicro =
                new ArrowType.Timestamp(TimeUnit.MICROSECOND, "UTC");
        public static final ArrowType TimeStampNano =
                new ArrowType.Timestamp(TimeUnit.NANOSECOND, "UTC");
        public static final ArrowType List = new ArrowType.List();
        public static final ArrowType Struct = new ArrowType.Struct();
        public static final ArrowType Map = new ArrowType.Map(true);
    }

    public static class FieldType {
        public static final org.apache.arrow.vector.types.pojo.FieldType Null =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.Null, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType Int =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Arrow.Type.Int, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType TinyInt =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.TinyInt, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType SmallInt =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.SmallInt, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType UInt =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.UInt, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType Int64 =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.Int64, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType UInt64 =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.UInt64, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType Float =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.Float, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType Double =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.Double, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType Boolean =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.Boolean, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType VarChar =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.VarChar, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType LargeVarChar =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.LargeVarChar, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType VarBinary =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.VarBinary, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType LargeVarBinary =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.LargeVarBinary, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType Date =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.Date, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType TimeStampMilli =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.TimeStampMilli, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType TimeStampMicro =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.TimeStampMicro, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType TimeStampNano =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.TimeStampNano, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType TimeStampSec =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.TimeStampSec, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType List =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.List, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType Struct =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.Struct, null);
        public static final org.apache.arrow.vector.types.pojo.FieldType Map =
                new org.apache.arrow.vector.types.pojo.FieldType(false, Type.Map, null);
    }

    public static Field makeField(
            String name, org.apache.arrow.vector.types.pojo.FieldType fieldType) {
        return new Field(name, fieldType, ImmutableList.of());
    }

    public static Field makeField(String name, ArrowType type) {
        return new Field(
                name,
                new org.apache.arrow.vector.types.pojo.FieldType(false, type, null),
                ImmutableList.of());
    }

    public static void instantiate() {
        Table.instantiate();
        RecordBatchStream.instantiate();
    }
}
