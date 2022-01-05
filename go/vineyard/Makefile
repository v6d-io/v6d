# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

CC				?= gcc
AR				?= ar

all: cgo fmt vet test

pkg/common/memory/libfling.a: pkg/common/memory/fling.cc
	${CC} -O3 -fPIC -c pkg/common/memory/fling.cc -o pkg/common/memory/fling.o
	${AR} rcs pkg/common/memory/libfling.a pkg/common/memory/fling.o

cgo: pkg/common/memory/libfling.a
.PHONY: cgo

# Run tests
test: cgo
	go test ./...
.PHONY: test

# Run go fmt against code
fmt:
	go fmt ./...
.PHONY: fmt

# Run go vet against code
vet:
	go vet ./...
.PHONY: vet

vendor:
	go mod tidy
	go mod vendor
.PHONY: vendor
