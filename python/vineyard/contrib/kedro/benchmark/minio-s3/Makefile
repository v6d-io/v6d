all:  build-images-with-1m-data build-images-with-10m-data build-images-with-100m-data build-images-with-500m-data

# the original dataset size is 450KB
# 1M dataset size is around 450KB * 3
build-images-with-1m-data:
	docker build -t minio-s3-benchmark-with-1m-data:latest -f Dockerfile --build-arg DATASET_COUNT=3 .

# 10M dataset size is around 450KB * 22
build-images-with-10m-data:
	docker build -t minio-s3-benchmark-with-10m-data:latest -f Dockerfile --build-arg DATASET_COUNT=22 .

# 100M dataset size is around 450KB * 222
build-images-with-100m-data:
	docker build -t minio-s3-benchmark-with-100m-data:latest -f Dockerfile --build-arg DATASET_COUNT=222 .

# 500M dataset size is around 450KB * 1111
build-images-with-500m-data:
	docker build -t minio-s3-benchmark-with-500m-data:latest -f Dockerfile --build-arg DATASET_COUNT=1111 .

