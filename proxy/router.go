// Package proxy implements the HTTP reverse proxy handler with AI-assisted
// routing and automatic Round-Robin fallback.
package proxy

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/l7-loadbalancer/config"
	"github.com/l7-loadbalancer/inference"
)

// ────────────────────────────────────────────────
// Shadow Mode Logger (Continuous Feedback Loop)
// ────────────────────────────────────────────────

type shadowRequest struct {
	Features    []float32
	TopProb     float64
	Timestamp   time.Time
}

var (
	shadowQueue = make(chan shadowRequest, 1000)
)

func init() {
	go func() {
		f, err := os.OpenFile("shadow_mode_low_confidence.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			log.Printf("[shadow] WARN: could not open shadow log: %v", err)
			return
		}
		defer f.Close()

		for req := range shadowQueue {
			// Write the feature vector to disk for the ML team
			strFeatures := make([]string, len(req.Features))
			for i, v := range req.Features {
				strFeatures[i] = fmt.Sprintf("%.4f", v)
			}
			line := fmt.Sprintf(`{"ts":"%s","prob":%.4f,"features":[%s]}`+"\n",
				req.Timestamp.Format(time.RFC3339),
				req.TopProb,
				strings.Join(strFeatures, ","),
			)
			_, _ = f.WriteString(line)
		}
	}()
}

// ────────────────────────────────────────────────
// Metrics (lock-free atomic counters)
// ────────────────────────────────────────────────

// RouterMetrics tracks routing statistics for observability.
type RouterMetrics struct {
	TotalRequests    atomic.Uint64
	AIRouted         atomic.Uint64
	FallbackRouted   atomic.Uint64
	InferenceErrors  atomic.Uint64
	InferenceTimeouts atomic.Uint64
	LowConfidence    atomic.Uint64
	SecurityBlocks   atomic.Uint64 // New: AI WAF Blocks
}

// Snapshot returns a point-in-time copy of the metrics.
func (m *RouterMetrics) Snapshot() map[string]uint64 {
	return map[string]uint64{
		"total_requests":     m.TotalRequests.Load(),
		"ai_routed":          m.AIRouted.Load(),
		"fallback_routed":    m.FallbackRouted.Load(),
		"inference_errors":   m.InferenceErrors.Load(),
		"inference_timeouts": m.InferenceTimeouts.Load(),
		"low_confidence":     m.LowConfidence.Load(),
		"security_blocks":    m.SecurityBlocks.Load(),
	}
}

// ────────────────────────────────────────────────
// Router
// ────────────────────────────────────────────────

// Router is the core HTTP handler that selects backends via AI inference
// with automatic fallback to Round-Robin.
type Router struct {
	engine      *inference.LayaEngine
	cfg         *config.Config
	backends    []*BackendProxy
	rrCounter   atomic.Uint64 // Round-Robin counter
	Metrics     RouterMetrics
}

// BackendProxy pairs a backend config with a pre-configured reverse proxy.
type BackendProxy struct {
	Name  string
	URL   *url.URL
	Proxy *httputil.ReverseProxy
}

// NewRouter initializes the router, creating a connection-pooled reverse proxy
// per backend.
func NewRouter(cfg *config.Config, engine *inference.LayaEngine) (*Router, error) {
	r := &Router{
		engine: engine,
		cfg:    cfg,
	}

	for _, b := range cfg.Backends {
		parsed, err := url.Parse(b.URL)
		if err != nil {
			return nil, fmt.Errorf("invalid backend URL %q: %w", b.URL, err)
		}

		transport := &http.Transport{
			MaxIdleConnsPerHost: cfg.MaxIdleConnsPerHost,
			MaxIdleConns:        cfg.MaxIdleConnsPerHost * 2,
			IdleConnTimeout:     90 * time.Second,
			DisableCompression:  false,
			ForceAttemptHTTP2:   true,
		}

		proxy := httputil.NewSingleHostReverseProxy(parsed)
		proxy.Transport = transport
		proxy.ErrorHandler = r.proxyErrorHandler

		r.backends = append(r.backends, &BackendProxy{
			Name:  b.Name,
			URL:   parsed,
			Proxy: proxy,
		})
	}

	if len(r.backends) == 0 {
		return nil, fmt.Errorf("no backends configured")
	}

	return r, nil
}

// ServeHTTP is the main request entry point.
func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	r.Metrics.TotalRequests.Add(1)

	// Health-check endpoint.
	if req.URL.Path == "/healthz" {
		r.handleHealthz(w)
		return
	}

	// Metrics endpoint.
	if req.URL.Path == "/metrics" {
		r.handleMetrics(w)
		return
	}

	backend := r.selectBackend(req)
	
	// AI Web Application Firewall (WAF) interception
	if backend.Name == "quarantine" {
		r.Metrics.SecurityBlocks.Add(1)
		http.Error(w, "403 Forbidden: Blocked by AI Security Policy", http.StatusForbidden)
		return
	}

	backend.Proxy.ServeHTTP(w, req)
}

// selectBackend uses AI inference when enabled and confident; otherwise falls
// back to Round-Robin.
func (r *Router) selectBackend(req *http.Request) *BackendProxy {
	if r.cfg.EnableAIRouting && r.engine != nil && !r.engine.IsMock() {
		if backend := r.aiRoute(req); backend != nil {
			return backend
		}
	} else if r.cfg.EnableAIRouting && r.engine != nil && r.engine.IsMock() {
		// Mock mode — still exercise the inference path for testing.
		if backend := r.aiRoute(req); backend != nil {
			return backend
		}
	}

	return r.roundRobin()
}

// aiRoute encodes the request, runs inference, applies calibrated softmax,
// and checks the confidence threshold.
func (r *Router) aiRoute(req *http.Request) *BackendProxy {
	featuresPtr := inference.EncodeRequest(req)
	features := *featuresPtr
	defer inference.PutFeatureVector(featuresPtr)

	logits, err := r.engine.Infer(req.Context(), features)
	if err != nil {
		if context.DeadlineExceeded.Error() == err.Error() || isTimeout(err) {
			r.Metrics.InferenceTimeouts.Add(1)
		} else {
			r.Metrics.InferenceErrors.Add(1)
		}
		r.Metrics.FallbackRouted.Add(1)
		return nil // caller uses Round-Robin
	}

	_, topIdx, topProb := inference.CalibratedSoftmax(logits, r.cfg.Temperature)

	if topProb < r.cfg.ConfidenceThreshold {
		r.Metrics.LowConfidence.Add(1)
		r.Metrics.FallbackRouted.Add(1)
		
		// [New Upgrade] Continuous Feedback Loop: Log to Shadow Mode asynchronously
		// Copy features so they don't get zeroed out by the sync.Pool
		shadowFeatures := make([]float32, len(features))
		copy(shadowFeatures, features)
		
		select {
		case shadowQueue <- shadowRequest{
			Features:  shadowFeatures,
			TopProb:   topProb,
			Timestamp: time.Now(),
		}:
		default:
			// Queue is full, drop the log so we don't block the proxy
		}
		
		return nil
	}

	if topIdx < 0 || topIdx >= len(r.backends) {
		r.Metrics.InferenceErrors.Add(1)
		r.Metrics.FallbackRouted.Add(1)
		return nil
	}

	r.Metrics.AIRouted.Add(1)
	return r.backends[topIdx]
}

// roundRobin performs a simple atomic counter-based round-robin selection.
func (r *Router) roundRobin() *BackendProxy {
	idx := r.rrCounter.Add(1) - 1
	r.Metrics.FallbackRouted.Add(1)
	return r.backends[idx%uint64(len(r.backends))]
}

// isTimeout checks if an error contains timeout-related messages.
func isTimeout(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return contains(s, "timeout") || contains(s, "deadline")
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// proxyErrorHandler handles backend connection failures.
func (r *Router) proxyErrorHandler(w http.ResponseWriter, req *http.Request, err error) {
	log.Printf("[proxy] backend error: %v", err)
	http.Error(w, "Service Unavailable", http.StatusServiceUnavailable)
}

// ────────────────────────────────────────────────
// Introspection Endpoints
// ────────────────────────────────────────────────

var metricsPool = sync.Pool{
	New: func() interface{} {
		b := make([]byte, 0, 512)
		return &b
	},
}

func (r *Router) handleHealthz(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/plain")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func (r *Router) handleMetrics(w http.ResponseWriter) {
	snap := r.Metrics.Snapshot()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w,
		`{"total_requests":%d,"ai_routed":%d,"fallback_routed":%d,"inference_errors":%d,"inference_timeouts":%d,"low_confidence":%d,"security_blocks":%d}`,
		snap["total_requests"],
		snap["ai_routed"],
		snap["fallback_routed"],
		snap["inference_errors"],
		snap["inference_timeouts"],
		snap["low_confidence"],
		snap["security_blocks"],
	)
}
