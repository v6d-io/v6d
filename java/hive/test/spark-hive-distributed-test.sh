#!/bin/bash

outdir=./spark-query/spark-out
if [ -d "$outdir" ]; then
    rm -r "$outdir"
fi

docker exec hive-metastore sh -c 'cp /hive-config-distributed/hive-site.xml $SPARK_HOME/conf/'
docker exec hive-metastore sh -c 'cp /spark-config-distributed/* $SPARK_HOME/conf/'

docker cp ./spark-query hive-metastore:/tmp/

for file in ./spark-query/*; do
    query=$(basename "$file")
    docker exec hive-metastore spark-shell --master yarn -i /tmp/spark-query/"$query"
done

docker cp hive-metastore:/tmp/spark-out ./spark-query/
for dir in ./spark-query/spark-out/*; do
    cat $dir/part-* > ./spark-query/spark-out/$(basename "$dir").q.out
    rm -r $dir
done

filecount=$(find ./spark-query/ -name "*.scala" | wc -l)
testedcount=$(find ./spark-query/spark-out/ -name "*.out" | wc -l)
successcount=0
failedcount=0

for file in ./spark-query/spark-out/*; do
    if [ -f "$file" ]; then
        echo "Diff $file with expected/$(basename "$file")"
        if diff -a "$file" ./expected/$(basename "$file"); then
            successcount=$((successcount+1))
        else
            failedcount=$((failedcount+1))
        fi
    fi
done

echo "Total test: $filecount Success: $successcount Failed: $failedcount Skipped: $((filecount-testedcount))"

if [ $successcount -eq $filecount ]; then
    exit 0
else
    exit 1
fi