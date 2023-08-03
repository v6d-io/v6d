package io.v6d.hive.ql.io;

import java.io.DataOutput;
import java.io.DataInput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.hive.ql.io.arrow.ArrowWrapperWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.StructTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils;
import static org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils.getStandardWritableObjectInspectorFromTypeInfo;

class VineyardRowWritable extends ArrowWrapperWritable {
    private List<Object> colValues;
    private TypeInfo[] targetTypeInfos;

    public VineyardRowWritable(final List<Object> colValues, ArrayList<TypeInfo> rowTypeInfo) {
        this.colValues = colValues;
        // StructObjectInspector rowObjectInspector = (StructObjectInspector) getStandardWritableObjectInspectorFromTypeInfo(rowTypeInfo);
        // final List<? extends StructField> fields = rowObjectInspector.getAllStructFieldRefs();
        // final int count = rowTypeInfo.size();//fields.size();
        // targetTypeInfos = new TypeInfo[count];
        // for (int i = 0; i < count; i++) {
            // final StructField field = fields.get(i);
            // final ObjectInspector fieldInspector = field.getFieldObjectInspector();
            // final TypeInfo typeInfo =
            //     TypeInfoUtils.getTypeInfoFromTypeString(fieldInspector.getTypeName());

        //     targetTypeInfos[i] = typeInfo;
        // }
        targetTypeInfos = rowTypeInfo.toArray(new TypeInfo[rowTypeInfo.size()]);
    }

    public List<Object> getValues() {
        return colValues;
    }

    public TypeInfo[] getTargetTypeInfos() {
        return targetTypeInfos;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        System.out.println("write");
        throw new IOException("write is not implemented");
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        System.out.println("readFields");
        throw new IOException("readFields is not implemented");
    }
}
