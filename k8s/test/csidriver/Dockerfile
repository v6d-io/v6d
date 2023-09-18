# Build the manager binary
FROM golang:1.19-buster as builder

ENV GO111MODULE=on
ENV CGO_ENABLED=0
ENV GOOS=linux

WORKDIR /workspace

# Copy the vineyard Go SDK
COPY go/ go/

# Copy the Go source for vineyardctl
COPY k8s/ k8s/

WORKDIR /workspace/k8s

RUN go mod download 

# Update the working directory
WORKDIR /workspace/k8s/cmd/commands/csi

USER root

CMD ["go", "test", "-run", "./..."]
