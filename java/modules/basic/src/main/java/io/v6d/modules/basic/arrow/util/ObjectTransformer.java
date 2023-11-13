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

    public Object defaultTransform(Object object) {
        return object;
    }

    public int intTransform(Object object) {
        return (int) object;
    }

    public long longTransform(Object object) {
        return (long) object;
    }

    public short shortTransform(Object object) {
        return (short) object;
    }

    public byte byteTransform(Object object) {
        return (byte) object;
    }

    public float floatTransform(Object object) {
        return (float) object;
    }

    public double doubleTransform(Object object) {
        return (double) object;
    }

    public int booleanTransform(Object object) {
        return (boolean) object == true ? 1 : 0;
    }

    public byte[] utf8Transform(Object object) {
        return object.toString().getBytes(StandardCharsets.UTF_8);
    }

    public byte[] largeUtf8Transform(Object object) {
        return object.toString().getBytes(StandardCharsets.UTF_8);
    }

    public byte[] binaryTransform(Object object) {
        return (byte[]) object;
    }

    public byte[] largeBinaryTransform(Object object) {
        return (byte[]) object;
    }

    public BigDecimal decimalTransform(Object object, int precision, int scale, int bitWidth) {
        return (BigDecimal) object;
    }

    public Timestamp timestampTransform(Object object) {
        return (Timestamp) object;
    }

    public Date dateTransform(Object object) {
        return (Date) object;
    }
}
