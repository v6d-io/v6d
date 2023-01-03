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
package io.v6d.core.common.util;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.val;

public class JSON {
    public static String getText(final JsonNode root, final String key) {
        return root.get(key).textValue();
    }

    public static String getTextMaybe(
            final JsonNode root, final String key, final String defaultValue) {
        val entry = root.get(key);
        if (entry == null || entry.isNull()) {
            return defaultValue;
        }
        return entry.textValue();
    }

    public static boolean getBool(final JsonNode root, final String key) {
        return root.get(key).booleanValue();
    }

    public static boolean getBoolMaybe(
            final JsonNode root, final String key, final boolean defaultValue) {
        val entry = root.get(key);
        if (entry == null || entry.isNull()) {
            return defaultValue;
        }
        return entry.booleanValue();
    }

    public static float getFloat(final JsonNode root, final String key) {
        return root.get(key).floatValue();
    }

    public static float getFloatMaybe(
            final JsonNode root, final String key, final float defaultValue) {
        val entry = root.get(key);
        if (entry == null || entry.isNull()) {
            return defaultValue;
        }
        return entry.floatValue();
    }

    public static double getDouble(final JsonNode root, final String key) {
        return root.get(key).doubleValue();
    }

    public static double getDoubleMaybe(
            final JsonNode root, final String key, final double defaultValue) {
        val entry = root.get(key);
        if (entry == null || entry.isNull()) {
            return defaultValue;
        }
        return entry.doubleValue();
    }

    public static int getInt(final JsonNode root, final String key) {
        return root.get(key).intValue();
    }

    public static int getIntMaybe(final JsonNode root, final String key, final int defaultValue) {
        val entry = root.get(key);
        if (entry == null || entry.isNull()) {
            return defaultValue;
        }
        return entry.intValue();
    }

    public static long getLong(final JsonNode root, final String key) {
        return root.get(key).longValue();
    }

    public static long getLongMaybe(
            final JsonNode root, final String key, final long defaultValue) {
        val entry = root.get(key);
        if (entry == null || entry.isNull()) {
            return defaultValue;
        }
        return entry.longValue();
    }
}
