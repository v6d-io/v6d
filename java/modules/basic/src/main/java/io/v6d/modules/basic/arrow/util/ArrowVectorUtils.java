package io.v6d.modules.basic.arrow.util;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;

import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.MapVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;
import org.apache.arrow.vector.types.pojo.ArrowType;

import org.apache.arrow.vector.types.pojo.Field;

import io.v6d.core.client.Context;
import io.v6d.core.common.util.VineyardException;
import io.v6d.core.common.util.VineyardException.NotImplemented;

public class ArrowVectorUtils {
    public static ArrowBuf[] getArrowBuffers(FieldVector vector) throws VineyardException {
        List<ArrowBuf> result = new ArrayList<>();
        if (vector instanceof StructVector) {
            result.add(vector.getValidityBuffer());
            for (FieldVector child : ((StructVector)vector).getChildrenFromFields()) {
                result.addAll(Arrays.asList(getArrowBuffers(child)));
            }
        } else if (vector instanceof ListVector) {
            result.add(vector.getValidityBuffer());
            result.add(vector.getOffsetBuffer());
            result.addAll(Arrays.asList(getArrowBuffers(((ListVector)vector).getDataVector())));
        } else {
            result.addAll(Arrays.asList(vector.getBuffers(false)));
        }
        return result.toArray(new ArrowBuf[result.size()]);
    }

    public static List<Integer> getValueCountOfArrowVector(FieldVector vector) throws VineyardException {
        List<Integer> result = new ArrayList<>();
        result.add(vector.getValueCount());
        Context.println("vector value count: " + vector.getValueCount());
        if (vector instanceof StructVector) {
            for (FieldVector child : ((StructVector)vector).getChildrenFromFields()) {
                result.addAll(getValueCountOfArrowVector(child));
            }
        } else if (vector instanceof ListVector) {
            result.addAll(getValueCountOfArrowVector(((ListVector)vector).getDataVector()));
        }
        return result;
    }

    public static void buildArrowVector(FieldVector vector, Field field) throws VineyardException {
        Context.println("=====================");
        List<Field> childFields = field.getChildren();
        if (vector instanceof ListVector) {
            if (vector instanceof MapVector) {
                childFields = childFields.get(0).getChildren();
            }
            Context.println("ListVector");
            if (childFields.size() != 1) {
                throw new NotImplemented("ListArrayBuilder only support one child field");
            }
            ((ListVector)vector).addOrGetVector(childFields.get(0).getFieldType());
            buildArrowVector(((ListVector)vector).getDataVector(), childFields.get(0));
        } else if (vector instanceof StructVector) {
            Context.println("StructVector");
            for (int i = 0; i < childFields.size(); i++) {
                FieldVector childVector = ((StructVector)vector).addOrGet(String.valueOf(i), childFields.get(i).getFieldType(), FieldVector.class);
                buildArrowVector(childVector, childFields.get(i));
            }
        } else {
            Context.println("primitive");
        }
        Context.println("=====================");
    }

    public static void buildArrowVector(FieldVector vector, Queue<ArrowBuf> bufs, Queue<Integer> valueCountQueue, Field field) throws VineyardException {
        int valueCount = valueCountQueue.poll();
        List<Field> childFields = field.getChildren();
        List<ArrowBuf> currentBufs = new ArrayList<>();
        Context.println("--------------------------");
        Context.println(field.getType().getTypeID().name());
        Context.println("value count:" + valueCount);

        switch (field.getType().getTypeID()) {
            case Struct:
                // prepare and load buf
                currentBufs.add(bufs.poll());
                vector.loadFieldBuffers(new ArrowFieldNode(valueCount, 0), currentBufs);

                // process child vector
                for (int i = 0; i < childFields.size(); i++) {
                    Context.println("stage 8");
                    FieldVector childFieldVector = ((StructVector)vector).addOrGet(String.valueOf(i), childFields.get(i).getFieldType(), FieldVector.class);
                    buildArrowVector(childFieldVector, bufs, valueCountQueue, childFields.get(i));
                }
                break;
            case Map:
                // Map type is map->list->struct
                childFields = childFields.get(0).getChildren();
            case List:
                if (childFields.size() != 1) {
                    throw new NotImplemented("ListArrayBuilder only support one child field");
                }
                currentBufs.add(bufs.poll());
                currentBufs.add(bufs.poll());
                vector.loadFieldBuffers(new ArrowFieldNode(valueCount, 0), currentBufs);
                ((ListVector)vector).addOrGetVector(childFields.get(0).getFieldType());
                FieldVector childFieldVector = ((ListVector)vector).getDataVector();
                buildArrowVector(childFieldVector, bufs, valueCountQueue, childFields.get(0));
                break;
            default:
                Context.println("stage 8");
                switch(field.getType().getTypeID()) {
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
                        Context.println("vector value count: " + vector.getValueCount());
                        break;
                    case Utf8:
                    case LargeUtf8:
                        currentBufs.add(bufs.poll());
                        currentBufs.add(bufs.poll());
                        currentBufs.add(bufs.poll());
                        vector.loadFieldBuffers(new ArrowFieldNode(valueCount, 0), currentBufs);
                        Context.println("vector value count: " + vector.getValueCount());
                        break;
                    default:
                        Context.println("Unsupported type: " + field.getType().getTypeID().name());
                        assert(false);
                }
        }
        Context.println("--------------------------");
    }

    public static void printFields(Field field) {
        Context.println("--------------------------");
        Context.println("Field type:" + field.getType().getTypeID().name());
        Context.println("Field name:" + field.getName());
        switch (field.getType().getTypeID()) {
            case Int:
                Context.println("bitWidth:" + ((ArrowType.Int)field.getType()).getBitWidth());
                break;
            case FloatingPoint:
                Context.println("precision:" + ((ArrowType.FloatingPoint)field.getType()).getPrecision().name());
                break;
            case Timestamp:
                Context.println("timeUnit:" + ((ArrowType.Timestamp)field.getType()).getUnit().name());
                break;
            case Struct:
                for (int i = 0; i < field.getChildren().size(); i++) {
                    printFields(field.getChildren().get(i));
                }
            default:
                break;
        }
        Context.println("--------------------------");
    }

    public static BigDecimal TransHiveDecimalToBigDecimal(Object obj, int scale) {
        try {
            Class<?> c = Class.forName("org.apache.hadoop.hive.common.type.HiveDecimal");
            java.lang.reflect.Method m = c.getMethod("bigDecimalValue");
            BigDecimal value = (BigDecimal)m.invoke(obj);
            if (value.scale() != scale) {
                value = value.setScale(scale);
            }
            return value;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static Object TransBigDecimalToHiveDecimal(BigDecimal obj) {
        try {
            Class<?> c = Class.forName("org.apache.hadoop.hive.common.type.HiveDecimal");
            java.lang.reflect.Method m = c.getMethod("create", BigDecimal.class);
            return m.invoke(null, obj);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
