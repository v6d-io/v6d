docker-build:
	docker build prepare-data/ -f Dockerfile \
		--build-arg APP=prepare-data.py \
		-t prepare-data

	docker build preprocess/ -f Dockerfile \
		--build-arg APP=preprocess.py \
		-t preprocess-data

	docker build train/ -f Dockerfile \
		--build-arg APP=train.py \
		-t train-data
	
	docker build test/ -f Dockerfile \
		--build-arg APP=test.py \
		-t test-data

load-images:
	kind load docker-image prepare-data
	kind load docker-image preprocess-data
	kind load docker-image train-data
	kind load docker-image test-data