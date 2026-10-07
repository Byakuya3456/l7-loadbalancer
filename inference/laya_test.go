package inference

import (
	"context"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/l7-loadbalancer/config"
)

// ────────────────────────────────────────────────
// EncodeRequest Tests
// ────────────────────────────────────────────────

func TestEncodeRequest_MethodCodes(t *testing.T) {
	methods := map[string]float32{
		http.MethodGet:    1,
		http.MethodPost:   2,
		http.MethodPut:    3,
		http.MethodDelete: 4,
		http.MethodPatch:  5,
		"OPTIONS":         0,
	}

	for method, expected := range methods {
		req := httptest.NewRequest(method, "/test", nil)
		vec := *EncodeRequest(req)

		if vec[0] != expected {
			t.Errorf("method %s: got %.0f, want %.0f", method, vec[0], expected)
		}
	}
}

func TestEncodeRequest_ContentLength(t *testing.T) {
	// Large body → high content-length feature.
	body := strings.NewReader(strings.Repeat("x", 1<<20)) // 1 MB
	req := httptest.NewRequest(http.MethodPost, "/upload", body)
	vec := *EncodeRequest(req)

	if vec[1] <= 0 {
		t.Errorf("content-length feature should be > 0 for 1MB body, got %f", vec[1])
	}
	if vec[1] > 1.0 {
		t.Errorf("content-length feature should be clamped to [0,1], got %f", vec[1])
	}
}

func TestEncodeRequest_Authorization(t *testing.T) {
	// Without auth.
	reqNoAuth := httptest.NewRequest(http.MethodGet, "/", nil)
	vecNoAuth := *EncodeRequest(reqNoAuth)
	if vecNoAuth[2] != 0.0 {
		t.Errorf("auth feature should be 0 without header, got %f", vecNoAuth[2])
	}

	// With auth.
	reqAuth := httptest.NewRequest(http.MethodGet, "/", nil)
	reqAuth.Header.Set("Authorization", "Bearer token123")
	vecAuth := *EncodeRequest(reqAuth)
	if vecAuth[2] != 1.0 {
		t.Errorf("auth feature should be 1.0 with header, got %f", vecAuth[2])
	}
}

func TestEncodeRequest_PathHashing(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/v2/users/profile", nil)
	vec := *EncodeRequest(req)

	// Path segments start at index 7 now.
	// We should have 4 segments hashed.
	for i := 7; i < 11; i++ {
		if vec[i] < 0 || vec[i] > 1.0 {
			t.Errorf("path hash at [%d] out of [0,1]: %f", i, vec[i])
		}
	}
}

func TestEncodeRequest_VectorLength(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	vec := *EncodeRequest(req)

	if len(vec) != MaxTokenLen {
		t.Errorf("vector length: got %d, want %d", len(vec), MaxTokenLen)
	}
}

// ────────────────────────────────────────────────
// CalibratedSoftmax Tests
// ────────────────────────────────────────────────

func TestCalibratedSoftmax_Uniform(t *testing.T) {
	logits := []float32{0, 0, 0}
	probs, _, topProb := CalibratedSoftmax(logits, 1.0)

	expected := float32(1.0 / 3.0)
	for i, p := range probs {
		if math.Abs(float64(p-expected)) > 1e-5 {
			t.Errorf("uniform logits: prob[%d] = %f, want ~%f", i, p, expected)
		}
	}

	if math.Abs(topProb-float64(expected)) > 1e-5 {
		t.Errorf("uniform logits: topProb = %f, want ~%f", topProb, expected)
	}
}

func TestCalibratedSoftmax_StrongSignal(t *testing.T) {
	logits := []float32{10, 0, 0}
	_, topIdx, topProb := CalibratedSoftmax(logits, 1.0)

	if topIdx != 0 {
		t.Errorf("strong signal: topIdx = %d, want 0", topIdx)
	}
	if topProb < 0.99 {
		t.Errorf("strong signal: topProb = %f, should be > 0.99", topProb)
	}
}

func TestCalibratedSoftmax_TemperatureSharpens(t *testing.T) {
	logits := []float32{2, 1, 0}

	_, _, probT1 := CalibratedSoftmax(logits, 1.0)
	_, _, probT05 := CalibratedSoftmax(logits, 0.5) // sharper

	if probT05 <= probT1 {
		t.Errorf("lower temperature should sharpen: T=0.5 prob=%f, T=1.0 prob=%f",
			probT05, probT1)
	}
}

func TestCalibratedSoftmax_TemperatureSmooths(t *testing.T) {
	logits := []float32{2, 1, 0}

	_, _, probT1 := CalibratedSoftmax(logits, 1.0)
	_, _, probT2 := CalibratedSoftmax(logits, 2.0) // smoother

	if probT2 >= probT1 {
		t.Errorf("higher temperature should smooth: T=2.0 prob=%f, T=1.0 prob=%f",
			probT2, probT1)
	}
}

func TestCalibratedSoftmax_ZeroTemperature(t *testing.T) {
	// Zero temperature should default to T=1.0 (no division by zero).
	logits := []float32{1, 2, 3}
	probs, _, _ := CalibratedSoftmax(logits, 0)

	var sum float64
	for _, p := range probs {
		sum += float64(p)
	}
	if math.Abs(sum-1.0) > 1e-4 {
		t.Errorf("probabilities should sum to 1.0, got %f", sum)
	}
}

func TestCalibratedSoftmax_ProbsSumToOne(t *testing.T) {
	logits := []float32{-1.5, 3.2, 0.8, -0.3}
	probs, _, _ := CalibratedSoftmax(logits, 1.2)

	var sum float64
	for _, p := range probs {
		sum += float64(p)
	}
	if math.Abs(sum-1.0) > 1e-4 {
		t.Errorf("probabilities should sum to 1.0, got %f", sum)
	}
}

// ────────────────────────────────────────────────
// LayaEngine Mock Inference Tests
// ────────────────────────────────────────────────

func newTestEngine(t *testing.T) *LayaEngine {
	t.Helper()
	cfg := config.DefaultConfig()
	cfg.EnableAIRouting = false // forces mock mode
	cfg.InferenceWorkers = 2
	cfg.InferenceTimeout = 100 * time.Millisecond

	engine, err := NewLayaEngine(cfg, 3)
	if err != nil {
		t.Fatalf("NewLayaEngine: %v", err)
	}
	if !engine.IsMock() {
		t.Fatal("expected mock mode")
	}
	return engine
}

func TestLayaEngine_MockInfer_ReturnsLogits(t *testing.T) {
	engine := newTestEngine(t)
	defer engine.Shutdown()

	features := make([]float32, MaxTokenLen)
	features[0] = 1 // GET

	logits, err := engine.Infer(context.Background(), features)
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}
	if len(logits) != 3 {
		t.Errorf("expected 3 logits, got %d", len(logits))
	}
}

func TestLayaEngine_MockInfer_LargePayloadRoutesToGPU(t *testing.T) {
	engine := newTestEngine(t)
	defer engine.Shutdown()

	features := make([]float32, MaxTokenLen)
	features[0] = 2   // POST
	features[1] = 0.8 // high content-length

	logits, err := engine.Infer(context.Background(), features)
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}

	// GPU cluster (index 0) should have the highest logit.
	if logits[0] <= logits[1] || logits[0] <= logits[2] {
		t.Errorf("expected GPU cluster to be preferred: logits=%v", logits)
	}
}

func TestLayaEngine_MockInfer_DeleteRoutesToQuarantine(t *testing.T) {
	engine := newTestEngine(t)
	defer engine.Shutdown()

	features := make([]float32, MaxTokenLen)
	features[0] = 4 // DELETE

	logits, err := engine.Infer(context.Background(), features)
	if err != nil {
		t.Fatalf("Infer: %v", err)
	}

	// Quarantine (index 2) should have the highest logit.
	if logits[2] <= logits[0] || logits[2] <= logits[1] {
		t.Errorf("expected quarantine to be preferred: logits=%v", logits)
	}
}

func TestLayaEngine_ConcurrentInference(t *testing.T) {
	engine := newTestEngine(t)
	defer engine.Shutdown()

	const numRequests = 100
	done := make(chan error, numRequests)

	for i := 0; i < numRequests; i++ {
		go func() {
			features := make([]float32, MaxTokenLen)
			features[0] = float32(i%5 + 1)
			_, err := engine.Infer(context.Background(), features)
			done <- err
		}()
	}

	for i := 0; i < numRequests; i++ {
		if err := <-done; err != nil {
			t.Errorf("concurrent inference %d failed: %v", i, err)
		}
	}
}

func TestLayaEngine_Shutdown_Idempotent(t *testing.T) {
	engine := newTestEngine(t)
	// Multiple shutdowns should not panic.
	engine.Shutdown()
	engine.Shutdown()
	engine.Shutdown()
}
