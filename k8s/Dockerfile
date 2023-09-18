# Build the manager binary
FROM golang:1.19 as builder

ENV GO111MODULE=on
ENV CGO_ENABLED=0
ENV GOOS=linux

WORKDIR /workspace

# Copy the vineyard Go SDK
COPY go/ go/

# Copy the Go source for vineyardctl
COPY k8s/ k8s/

# Update the working directory
WORKDIR /workspace/k8s

# Build
RUN go mod download
RUN go build -a -o vineyardctl cmd/main.go && \
    strip vineyardctl || true


# Use distroless as minimal base image to package the manager binary
# Refer to https://github.com/GoogleContainerTools/distroless for more details
FROM gcr.io/distroless/base:debug
WORKDIR /

COPY k8s/config/scheduler/config.yaml /etc/kubernetes/scheduler.yaml

COPY --from=builder /workspace/k8s/vineyardctl /vineyardctl
