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
package io.v6d.modules.basic.arrow.util;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.sql.Date;
import java.sql.Timestamp;

public class ObjectTransformer {
    public ObjectTransformer() {}

    public Object transform(Object object) {
        return object;
    }

    public int transformInt(Object object) {
        return (int) object;
    }

    public long transformLong(Object object) {
        return (long) object;
    }

    public short transformShort(Object object) {
        return (short) object;
    }

    public byte transformByte(Object object) {
        return (byte) object;
    }

    public float transformFloat(Object object) {
        return (float) object;
    }

    public double transformDouble(Object object) {
        return (double) object;
    }

    public int transformBoolean(Object object) {
        return (boolean) object == true ? 1 : 0;
    }

    public byte[] transformUtf8(Object object) {
        return object.toString().getBytes(StandardCharsets.UTF_8);
    }

    public byte[] transformLargeUtf8(Object object) {
        return object.toString().getBytes(StandardCharsets.UTF_8);
    }

    public byte[] transformBinary(Object object) {
        return (byte[]) object;
    }

    public byte[] transformLargeBinary(Object object) {
        return (byte[]) object;
    }

    public BigDecimal transformDecimal(Object object, int precision, int scale, int bitWidth) {
        return (BigDecimal) object;
    }

    public Timestamp transformTimestamp(Object object) {
        return (Timestamp) object;
    }

    public Date transformDate(Object object) {
        return (Date) object;
    }
}
