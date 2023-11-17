REGISTRY := "ghcr.io/v6d-io/v6d/csidriver-example"
docker-build:
	docker build prepare-data/ -f Dockerfile \
		--build-arg APP=prepare-data.py \
		-t $(REGISTRY)/prepare-data

	docker build preprocess/ -f Dockerfile \
		--build-arg APP=preprocess.py \
		-t $(REGISTRY)/preprocess-data

	docker build train/ -f Dockerfile \
		--build-arg APP=train.py \
		-t $(REGISTRY)/train-data
	
	docker build test/ -f Dockerfile \
		--build-arg APP=test.py \
		-t $(REGISTRY)/test-data

push-images:
	docker push $(REGISTRY)/prepare-data
	docker push $(REGISTRY)/preprocess-data
	docker push $(REGISTRY)/train-data
	docker push $(REGISTRY)/test-data
