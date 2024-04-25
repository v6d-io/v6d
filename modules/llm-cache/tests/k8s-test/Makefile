registry = registry.cn-wulanchabu.aliyuncs.com/vineyard
build-images:
	docker build -t ${registry}/fs-llm-master:latest -f ./Dockerfile.master .
	docker build -t ${registry}/fs-llm-worker:latest -f ./Dockerfile.worker .
push-images:
	docker push ${registry}/fs-llm-master:latest
	docker push ${registry}/fs-llm-worker:latest