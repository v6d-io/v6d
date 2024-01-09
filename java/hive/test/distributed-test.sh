outdir=./query/out
if [ -d "$outdir" ]; then
    rm -r "$outdir"
fi
docker cp ./query hive-hiveserver2:/tmp/

for file in ./query/*; do
    query=$(basename "$file")
    docker exec hive-hiveserver2 beeline -u 'jdbc:hive2://localhost:10000/;transportMode=https;httpPath=cliservice' \
                -f /tmp/query/"$query" -n root
done

docker cp hive-hiveserver2:/tmp/out ./query/
for dir in ./query/out/*; do
    cat $dir/* > ./query/out/$(basename "$dir").q.out
    rm -r $dir
done

filecount=$(find ./query/  -maxdepth 1  -name "*.q" | wc -l)
testedcount=$(find ./query/out/ -maxdepth 1 -name "*.q.out"| wc -l)
successcount=0
failedcount=0

for file in ./query/out/*; do
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
