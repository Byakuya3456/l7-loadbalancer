//go:build !cgo
// +build !cgo

package inference

import (
	"fmt"
)

// In non-cgo environments, ONNX runtime is unavailable.
func (e *LayaEngine) initORT() error {
	return fmt.Errorf("CGO is disabled in this build; real ONNX inference is unavailable")
}

func (e *LayaEngine) ortInfer(features []float32) InferenceResult {
	return InferenceResult{Err: fmt.Errorf("CGO is disabled; cannot run ONNX inference")}
}

func (e *LayaEngine) cleanupORT() {
	// No-op
}
