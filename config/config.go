// Package config defines configuration structures and defaults for the L7 load balancer.
package config

import (
	"time"
)

// Backend represents a single upstream server target, including semantic schema
// descriptions used dynamically by the Laya ONNX engine at inference time.
type Backend struct {
	Name        string `json:"name"`
	URL         string `json:"url"`
	Description string `json:"description"` // Semantic meaning used for Dynamic Schema Passing
}

// Config holds the complete server configuration.
type Config struct {
	// ListenAddr is the address the proxy listens on (e.g., ":8080").
	ListenAddr string `json:"listen_addr"`

	// Backends is an ordered list of upstream servers.
	Backends []Backend `json:"backends"`

	// ModelPath is the filesystem path to the quantized ONNX model.
	ModelPath string `json:"model_path"`

	// ONNXLibPath is the path to the ONNX Runtime shared library.
	ONNXLibPath string `json:"onnx_lib_path"`

	// InferenceWorkers controls the size of the ONNX evaluation worker pool.
	InferenceWorkers int `json:"inference_workers"`

	// InferenceTimeout is the maximum wall-clock time allowed for a single inference.
	InferenceTimeout time.Duration `json:"inference_timeout"`

	// ConfidenceThreshold is the minimum top-class probability to trust the AI decision.
	// Below this threshold, routing falls back to Round-Robin.
	ConfidenceThreshold float64 `json:"confidence_threshold"`

	// Temperature is the softmax temperature scaling parameter (T).
	// T < 1.0 sharpens, T > 1.0 smooths the distribution. T = 1.0 is neutral.
	Temperature float64 `json:"temperature"`

	// MaxIdleConnsPerHost controls Keep-Alive connection pooling to each backend.
	MaxIdleConnsPerHost int `json:"max_idle_conns_per_host"`

	// GracefulShutdownTimeout is the maximum duration to wait for in-flight requests during shutdown.
	GracefulShutdownTimeout time.Duration `json:"graceful_shutdown_timeout"`

	// EnableAIRouting toggles AI-based routing. When false, pure Round-Robin is used.
	EnableAIRouting bool `json:"enable_ai_routing"`
}

// DefaultConfig returns a production-grade default configuration.
func DefaultConfig() *Config {
	return &Config{
		ListenAddr: ":8080",
		Backends: []Backend{
			{Name: "gpu_cluster", URL: "http://host.docker.internal:9001"},
			{Name: "standard_nodes", URL: "http://host.docker.internal:9002"},
			{Name: "quarantine", URL: "http://host.docker.internal:9003"},
		},
		ModelPath:               "models/laya_l7_quantized.onnx",
		ONNXLibPath:             "",
		InferenceWorkers:        4,
		InferenceTimeout:        10 * time.Millisecond,
		ConfidenceThreshold:     0.85,
		Temperature:             1.0,
		MaxIdleConnsPerHost:     1000,
		GracefulShutdownTimeout: 30 * time.Second,
		EnableAIRouting:         true,
	}
}
