# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

GOLINT := $(shell command -v ${GOBIN}/golangci-lint --version 2>/dev/null)

## Tool Versions
GOLANGCI_LINT_VERSION ?= v1.49.0

all: fmt test

# Run golangci-lint
lint:
    ifeq (${GOLINT},)
		@echo "golangci-lint not exist, installing it..."
		go install github.com/golangci/golangci-lint/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)
    endif
	${GOBIN}/golangci-lint run ./...
.PHONY: golint

# Run tests
test:
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
