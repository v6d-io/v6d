outdir=./query/out
if [ -d "$outdir" ]; then
    rm -r "$outdir"
fi
docker cp ./query hive:/tmp/

for file in ./query/*; do
    query=$(basename "$file")
    docker exec hive beeline -u 'jdbc:hive2://localhost:10000/;transportMode=http;httpPath=cliservice' \
                -f /tmp/query/"$query"
done

docker cp hive:/tmp/out ./query/
for dir in ./query/out/*; do
    cat $dir/* > ./query/out/$(basename "$dir").q.out
    rm -r $dir
done

filecount=$(find ./query/ -name "*.q" | wc -l)
testedcount=$(find ./query/out/ -name "*.out" | wc -l)
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
