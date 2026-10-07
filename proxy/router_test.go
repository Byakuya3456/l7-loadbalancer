package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/l7-loadbalancer/config"
	"github.com/l7-loadbalancer/inference"
)

// ────────────────────────────────────────────────
// Test Helpers
// ────────────────────────────────────────────────

// newTestRouter creates a router with real HTTP test backends.
func newTestRouter(t *testing.T, enableAI bool) (*Router, []*httptest.Server) {
	t.Helper()

	// Spin up 3 dummy backends that echo their name.
	names := []string{"gpu_cluster", "standard_nodes", "quarantine"}
	var servers []*httptest.Server
	var backends []config.Backend

	for _, name := range names {
		n := name // capture
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("X-Backend", n)
			w.WriteHeader(http.StatusOK)
			fmt.Fprintf(w, "served by %s", n)
		}))
		servers = append(servers, srv)
		backends = append(backends, config.Backend{Name: n, URL: srv.URL})
	}

	cfg := config.DefaultConfig()
	cfg.Backends = backends
	cfg.EnableAIRouting = enableAI
	cfg.InferenceWorkers = 2
	cfg.InferenceTimeout = 100 * time.Millisecond
	cfg.ConfidenceThreshold = 0.85

	engine, err := inference.NewLayaEngine(cfg, len(backends))
	if err != nil {
		t.Fatalf("NewLayaEngine: %v", err)
	}

	router, err := NewRouter(cfg, engine)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	return router, servers
}

func closeServers(servers []*httptest.Server) {
	for _, s := range servers {
		s.Close()
	}
}

// ────────────────────────────────────────────────
// Round-Robin Tests
// ────────────────────────────────────────────────

func TestRoundRobin_Distribution(t *testing.T) {
	router, servers := newTestRouter(t, false) // AI disabled
	defer closeServers(servers)

	counts := make(map[string]int)
	const numReqs = 30

	for i := 0; i < numReqs; i++ {
		req := httptest.NewRequest(http.MethodGet, "/test", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		body, _ := io.ReadAll(w.Result().Body)
		counts[string(body)]++
	}

	// gpu_cluster and standard_nodes should each get 10 requests.
	// quarantine should get blocked and return a 403, so it gets 0 successful proxy requests.
	for _, name := range []string{"gpu_cluster", "standard_nodes"} {
		key := fmt.Sprintf("served by %s", name)
		if counts[key] != 10 {
			t.Errorf("backend %s: got %d requests, want 10", name, counts[key])
		}
	}

	wafBlocks := counts["403 Forbidden: Blocked by AI Security Policy\n"]
	if wafBlocks != 10 {
		t.Errorf("expected 10 WAF blocks for quarantine via round-robin, got %d", wafBlocks)
	}
}

// ────────────────────────────────────────────────
// AI Routing Tests (Mock Mode)
// ────────────────────────────────────────────────

func TestAIRouting_LargePostToGPU(t *testing.T) {
	router, servers := newTestRouter(t, true) // AI enabled (mock)
	defer closeServers(servers)

	// POST with large body should route to gpu_cluster.
	body := strings.NewReader(strings.Repeat("x", 1<<20))
	req := httptest.NewRequest(http.MethodPost, "/upload", body)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	backendHeader := w.Result().Header.Get("X-Backend")
	if backendHeader != "gpu_cluster" {
		t.Errorf("large POST should route to gpu_cluster, got %s", backendHeader)
	}
}

func TestAIRouting_DeleteToQuarantine(t *testing.T) {
	router, servers := newTestRouter(t, true)
	defer closeServers(servers)

	req := httptest.NewRequest(http.MethodDelete, "/resource/123", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusForbidden {
		t.Errorf("DELETE should be blocked by AI WAF (403), got %d", w.Code)
	}

	snap := router.Metrics.Snapshot()
	if snap["security_blocks"] != 1 {
		t.Errorf("expected 1 security block, got %d", snap["security_blocks"])
	}
}

// ────────────────────────────────────────────────
// Fallback Tests
// ────────────────────────────────────────────────

func TestFallback_LowConfidence(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.EnableAIRouting = true
	cfg.ConfidenceThreshold = 0.99 // Extremely high — should force fallback
	cfg.InferenceWorkers = 1
	cfg.InferenceTimeout = 100 * time.Millisecond

	names := []string{"a", "b", "c"}
	var servers []*httptest.Server
	var backends []config.Backend
	for _, name := range names {
		n := name
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("X-Backend", n)
			fmt.Fprintf(w, "served by %s", n)
		}))
		servers = append(servers, srv)
		backends = append(backends, config.Backend{Name: n, URL: srv.URL})
	}
	defer closeServers(servers)

	cfg.Backends = backends
	engine, _ := inference.NewLayaEngine(cfg, 3)
	router, _ := NewRouter(cfg, engine)

	// With very high confidence threshold, the mock should fail to meet it,
	// triggering round-robin fallback.
	req := httptest.NewRequest(http.MethodGet, "/something", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	snap := router.Metrics.Snapshot()
	if snap["fallback_routed"] == 0 {
		t.Error("expected fallback routing due to low confidence")
	}
}

func TestFallback_InferenceTimeout(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.EnableAIRouting = true
	cfg.InferenceTimeout = 1 * time.Nanosecond // absurdly short — forces timeout
	cfg.InferenceWorkers = 1

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "ok")
	}))
	defer srv.Close()

	cfg.Backends = []config.Backend{{Name: "test", URL: srv.URL}}
	engine, _ := inference.NewLayaEngine(cfg, 1)
	router, _ := NewRouter(cfg, engine)

	// Flood with requests to increase chance of timeout.
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)
		}()
	}
	wg.Wait()

	// At least some should have been routed, even if via fallback.
	snap := router.Metrics.Snapshot()
	if snap["total_requests"] != 50 {
		t.Errorf("expected 50 total requests, got %d", snap["total_requests"])
	}
}

// ────────────────────────────────────────────────
// Healthz / Metrics Endpoints
// ────────────────────────────────────────────────

func TestHealthzEndpoint(t *testing.T) {
	router, servers := newTestRouter(t, false)
	defer closeServers(servers)

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("healthz: got %d, want 200", w.Code)
	}
	if body := w.Body.String(); body != "ok" {
		t.Errorf("healthz body: got %q, want \"ok\"", body)
	}
}

func TestMetricsEndpoint(t *testing.T) {
	router, servers := newTestRouter(t, false)
	defer closeServers(servers)

	// Generate some traffic.
	for i := 0; i < 5; i++ {
		req := httptest.NewRequest(http.MethodGet, "/test", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
	}

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("metrics: got %d, want 200", w.Code)
	}

	var data map[string]uint64
	if err := json.Unmarshal(w.Body.Bytes(), &data); err != nil {
		t.Fatalf("metrics JSON parse: %v", err)
	}

	// 5 test requests + 1 metrics request = 6 total.
	if data["total_requests"] != 6 {
		t.Errorf("total_requests: got %d, want 6", data["total_requests"])
	}
}

// ────────────────────────────────────────────────
// Concurrent Proxy Test
// ────────────────────────────────────────────────

func TestConcurrentProxying(t *testing.T) {
	router, servers := newTestRouter(t, true)
	defer closeServers(servers)

	const numReqs = 200
	var wg sync.WaitGroup
	errors := make(chan error, numReqs)

	for i := 0; i < numReqs; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			method := http.MethodGet
			if idx%3 == 0 {
				method = http.MethodPost
			}

			req := httptest.NewRequest(method, "/api/test", nil).WithContext(ctx)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			if w.Code != http.StatusOK {
				errors <- fmt.Errorf("request %d: got %d", idx, w.Code)
			}
		}(i)
	}
	wg.Wait()
	close(errors)

	for err := range errors {
		t.Error(err)
	}

	snap := router.Metrics.Snapshot()
	if snap["total_requests"] != numReqs {
		t.Errorf("total_requests: got %d, want %d", snap["total_requests"], numReqs)
	}
}
