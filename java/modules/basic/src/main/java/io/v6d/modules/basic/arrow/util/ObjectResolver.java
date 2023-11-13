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

    public Object defaultResolve(Object object) {
        return object;
    }

    public Object intResolve(int object) {
        return object;
    }

    public Object longResolve(long object) {
        return object;
    }

    public Object shortResolve(short object) {
        return object;
    }

    public byte byteResolve(byte object) {
        return object;
    }

    public float floatResolve(float object) {
        return object;
    }

    public double doubleResolve(double object) {
        return object;
    }

    public Object booleanResolve(int object) {
        return object == 1;
    }

    public Object utf8Resolve(String object) {
        return object;
    }

    public Object largeUtf8Resolve(String object) {
        return object;
    }

    public Object binaryResolve(byte[] object) {
        return object;
    }

    public Object largeBinaryResolve(byte[] object) {
        return object;
    }

    public Object decimalResolve(BigDecimal object, int precision, int scale, int bitWidth) {
        return object;
    }

    public Object timestampResolve(Timestamp object) {
        return object;
    }

    public Object dateResolve(Date object) {
        return object;
    }
}
