# Stage 1: Build the Go binary
FROM golang:1.23-bookworm AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# We need CGO enabled for ONNX Runtime bindings
RUN CGO_ENABLED=1 GOOS=linux go build -o l7-lb .

# Stage 2: Create the minimal production image
FROM debian:bookworm-slim
WORKDIR /app

# Install required shared libraries for ONNX (e.g. libc)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime shared library
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz && \
    tar xzf onnxruntime-linux-x64-1.20.0.tgz && \
    cp onnxruntime-linux-x64-1.20.0/lib/libonnxruntime.so* /usr/lib/ && \
    rm -rf onnxruntime-linux-x64-1.20.0*

# Copy compiled binary and models
COPY --from=builder /app/l7-lb /app/l7-lb
COPY models/ /app/models/

EXPOSE 8080

CMD ["./l7-lb", "-model", "models/laya_l7_fp32.onnx", "-ort-lib", "/usr/lib/libonnxruntime.so", "-ai=true"]
