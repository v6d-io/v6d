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
import java.sql.Date;
import java.sql.Timestamp;

public class ObjectResolver {
    public ObjectResolver() {}

    public Object resolve(Object object) {
        return object;
    }

    public Object resolveInt(int object) {
        return object;
    }

    public Object resolveLong(long object) {
        return object;
    }

    public Object resolveShort(short object) {
        return object;
    }

    public byte resolveByte(byte object) {
        return object;
    }

    public float resolveFloat(float object) {
        return object;
    }

    public double resolveDouble(double object) {
        return object;
    }

    public Object resolveBoolean(int object) {
        return object == 1;
    }

    public Object resolveUtf8(String object) {
        return object;
    }

    public Object resolveLargeUtf8(String object) {
        return object;
    }

    public Object resolveBinary(byte[] object) {
        return object;
    }

    public Object resolveLargeBinary(byte[] object) {
        return object;
    }

    public Object resolveDecimal(BigDecimal object, int precision, int scale, int bitWidth) {
        return object;
    }

    public Object resolveTimestamp(Timestamp object) {
        return object;
    }

    public Object resolveDate(Date object) {
        return object;
    }
}
