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

import io.v6d.core.client.Context;
import io.v6d.core.common.util.VineyardException;
import io.v6d.core.common.util.VineyardException.NotImplemented;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.MapVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.ArrowType.ArrowTypeID;
import org.apache.arrow.vector.types.pojo.Field;

public class ArrowVectorUtils {
    private static Map<ArrowType.ArrowTypeID, ObjectTransformer> defaultTransformers;
    private static Map<ArrowType.ArrowTypeID, ObjectResolver> defaultResolvers;

    public static ArrowBuf[] getArrowBuffers(FieldVector vector) throws VineyardException {
        List<ArrowBuf> result = new ArrayList<>();
        if (vector instanceof StructVector) {
            result.add(vector.getValidityBuffer());
            for (FieldVector child : ((StructVector) vector).getChildrenFromFields()) {
                result.addAll(Arrays.asList(getArrowBuffers(child)));
            }
        } else if (vector instanceof ListVector) {
            result.add(vector.getValidityBuffer());
            result.add(vector.getOffsetBuffer());
            result.addAll(Arrays.asList(getArrowBuffers(((ListVector) vector).getDataVector())));
        } else {
            result.addAll(Arrays.asList(vector.getBuffers(false)));
        }
        return result.toArray(new ArrowBuf[result.size()]);
    }

    public static List<Integer> getValueCountOfArrowVector(FieldVector vector)
            throws VineyardException {
        List<Integer> result = new ArrayList<>();
        result.add(vector.getValueCount());
        if (vector instanceof StructVector) {
            for (FieldVector child : ((StructVector) vector).getChildrenFromFields()) {
                result.addAll(getValueCountOfArrowVector(child));
            }
        } else if (vector instanceof ListVector) {
            result.addAll(getValueCountOfArrowVector(((ListVector) vector).getDataVector()));
        }
        return result;
    }

    public static void buildArrowVector(FieldVector vector, Field field) throws VineyardException {
        List<Field> childFields = field.getChildren();
        if (vector instanceof ListVector) {
            if (vector instanceof MapVector) {
                childFields = childFields.get(0).getChildren();
            }
            if (childFields.size() != 1) {
                throw new NotImplemented("ListArrayBuilder only support one child field");
            }
            ((ListVector) vector).addOrGetVector(childFields.get(0).getFieldType());
            buildArrowVector(((ListVector) vector).getDataVector(), childFields.get(0));
        } else if (vector instanceof StructVector) {
            for (int i = 0; i < childFields.size(); i++) {
                FieldVector childVector =
                        ((StructVector) vector)
                                .addOrGet(
                                        String.valueOf(i),
                                        childFields.get(i).getFieldType(),
                                        FieldVector.class);
                buildArrowVector(childVector, childFields.get(i));
            }
        } else {
            Context.println("Primitive type. Nothing to do.");
        }
    }

    public static void buildArrowVector(
            FieldVector vector, Queue<ArrowBuf> bufs, Queue<Integer> valueCountQueue, Field field) {
        int valueCount = valueCountQueue.poll();
        List<Field> childFields = field.getChildren();
        List<ArrowBuf> currentBufs = new ArrayList<>();

        switch (field.getType().getTypeID()) {
            case Struct:
                // prepare and load buf
                currentBufs.add(bufs.poll());
                vector.loadFieldBuffers(new ArrowFieldNode(valueCount, 0), currentBufs);

                // process child vector
                for (int i = 0; i < childFields.size(); i++) {
                    FieldVector childFieldVector =
                            ((StructVector) vector)
                                    .addOrGet(
                                            String.valueOf(i),
                                            childFields.get(i).getFieldType(),
                                            FieldVector.class);
                    buildArrowVector(childFieldVector, bufs, valueCountQueue, childFields.get(i));
                }
                break;
            case Map:
                // Map type is map->list->struct
                childFields = childFields.get(0).getChildren();
            case List:
                assert childFields.size() == 1 : "ListArrayBuilder only support one child field";
                currentBufs.add(bufs.poll());
                currentBufs.add(bufs.poll());
                vector.loadFieldBuffers(new ArrowFieldNode(valueCount, 0), currentBufs);
                ((ListVector) vector).addOrGetVector(childFields.get(0).getFieldType());
                FieldVector childFieldVector = ((ListVector) vector).getDataVector();
                buildArrowVector(childFieldVector, bufs, valueCountQueue, childFields.get(0));
                break;
            default:
                switch (field.getType().getTypeID()) {
                    case Int:
                    case FloatingPoint:
                    case Bool:
                    case Binary:
                    case Date:
                    case Decimal:
                    case Timestamp:
                        currentBufs.add(bufs.poll());
                        currentBufs.add(bufs.poll());
                        vector.loadFieldBuffers(new ArrowFieldNode(valueCount, 0), currentBufs);
                        break;
                    case Utf8:
                    case LargeUtf8:
                        currentBufs.add(bufs.poll());
                        currentBufs.add(bufs.poll());
                        currentBufs.add(bufs.poll());
                        vector.loadFieldBuffers(new ArrowFieldNode(valueCount, 0), currentBufs);
                        break;
                    default:
                        assert false : "Unsupported type: " + field.getType().getTypeID().name();
                }
        }
    }

    public static void printFields(Field field, int level) {
        Context.println("--------------------------".substring(level * 3));
        Context.println("Field type:" + field.getType().getTypeID().name());
        Context.println("Field name:" + field.getName());
        switch (field.getType().getTypeID()) {
            case Int:
                Context.println("bitWidth:" + ((ArrowType.Int) field.getType()).getBitWidth());
                break;
            case FloatingPoint:
                Context.println(
                        "precision:"
                                + ((ArrowType.FloatingPoint) field.getType())
                                        .getPrecision()
                                        .name());
                break;
            case Timestamp:
                Context.println(
                        "timeUnit:" + ((ArrowType.Timestamp) field.getType()).getUnit().name());
                break;
            case Struct:
                for (int i = 0; i < field.getChildren().size(); i++) {
                    printFields(field.getChildren().get(i), level + 1);
                }
            default:
                break;
        }
        Context.println("--------------------------");
    }

    public static Map<ArrowType.ArrowTypeID, ObjectTransformer> getDefaultTransformers() {
        if (defaultTransformers == null) {
            defaultTransformers = new HashMap<>();
            ArrowTypeID[] arrowTypeIDs = ArrowTypeID.values();
            for (ArrowTypeID arrowTypeID : arrowTypeIDs) {
                defaultTransformers.put(arrowTypeID, new ObjectTransformer());
            }
        }
        return defaultTransformers;
    }

    public static Map<ArrowType.ArrowTypeID, ObjectResolver> getDefaultResolver() {
        if (defaultResolvers == null) {
            defaultResolvers = new HashMap<>();
            ArrowTypeID[] arrowTypeIDs = ArrowTypeID.values();
            for (ArrowTypeID arrowTypeID : arrowTypeIDs) {
                defaultResolvers.put(arrowTypeID, new ObjectResolver());
            }
        }
        return defaultResolvers;
    }
}
