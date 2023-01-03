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

public class Env {
    public static final String VINEYARD_IPC_SOCKET = "VINEYARD_IPC_SOCKET";

    public static final String VINEYARD_RPC_ENDPOINT = "VINEYARD_RPC_ENDPOINT";

    public static String getEnv(String key) {
        return System.getenv(key);
    }

    public static String getEnvOrNull(String key) {
        return getEnvOr(key, null);
    }

    public static String getEnvOrEmpty(String key) {
        return getEnvOr(key, "");
    }

    public static String getEnvOr(String key, String value) {
        return System.getenv().getOrDefault(key, value);
    }
}
