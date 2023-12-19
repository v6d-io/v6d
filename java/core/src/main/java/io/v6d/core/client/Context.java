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
package io.v6d.core.client;

import com.google.common.base.Stopwatch;
import com.google.common.base.StopwatchContext;
import io.v6d.core.common.util.InstanceID;
import io.v6d.core.common.util.VineyardException;
import java.text.SimpleDateFormat;

public class Context {
    public static IPCClient client = null;
    public static ThreadLocal<SimpleDateFormat> formatter =
            ThreadLocal.withInitial(() -> new SimpleDateFormat("HH:mm:ss.SSS"));

    public static synchronized IPCClient getClient() throws VineyardException {
        if (client == null) {
            Stopwatch watch = StopwatchContext.create();
            client = new IPCClient();
            Context.println(
                    "Connected to vineyard: " + client.getIPCSocket() + " uses " + watch.stop());
        }
        return client;
    }

    public static synchronized InstanceID getInstanceID() {
        if (client == null) {
            return null;
        }
        return client.getInstanceId();
    }

    public static void println(String message) {
        System.err.printf("[%s] %s\n", formatter.get().format(System.currentTimeMillis()), message);
    }

    public static void printf(String format, Object... args) {
        System.err.printf("[%s] ", formatter.get().format(System.currentTimeMillis()));
        System.err.printf(format, args);
    }
}
