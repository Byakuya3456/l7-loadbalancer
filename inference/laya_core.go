package inference

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/l7-loadbalancer/config"
)

// ────────────────────────────────────────────────
// Request Feature Encoding
// ────────────────────────────────────────────────

const MaxTokenLen = 64

// ────────────────────────────────────────────────
// Zero-Allocation Pools & Inference Cache
// ────────────────────────────────────────────────

var (
	// featurePool drastically reduces GC pressure by recycling the 64-float arrays
	featurePool = sync.Pool{
		New: func() interface{} {
			v := make([]float32, MaxTokenLen)
			return &v
		},
	}

	// inferenceCache provides ~1 microsecond LRU caching for identical HTTP requests
	inferenceCache sync.Map
)

type cacheEntry struct {
	Logits [3]float32
	Expiry time.Time
}

// GetFeatureVector retrieves a recycled float32 slice from the sync.Pool
func GetFeatureVector() *[]float32 {
	return featurePool.Get().(*[]float32)
}

// PutFeatureVector returns a used slice to the sync.Pool
func PutFeatureVector(vec *[]float32) {
	// Zero out the vector before returning
	for i := range *vec {
		(*vec)[i] = 0
	}
	featurePool.Put(vec)
}

func EncodeRequest(r *http.Request) *[]float32 {
	vecPtr := GetFeatureVector()
	vec := *vecPtr
	switch r.Method {
	case http.MethodGet:
		vec[0] = 1
	case http.MethodPost:
		vec[0] = 2
	case http.MethodPut:
		vec[0] = 3
	case http.MethodDelete:
		vec[0] = 4
	case http.MethodPatch:
		vec[0] = 5
	default:
		vec[0] = 0
	}
	cl := float64(r.ContentLength)
	if cl < 0 {
		cl = 0
	}
	vec[1] = float32(math.Min(math.Log2(1+cl)/30.0, 1.0))
	
	if r.Header.Get("Authorization") != "" {
		vec[2] = 1.0
	}
	
	pathStr := r.URL.Path
	vec[3] = float32(math.Min(float64(len(pathStr))/512.0, 1.0))
	
	// [New Feature] IP Subnet Hash (internal vs external heuristics)
	ip := strings.Split(r.RemoteAddr, ":")[0]
	vec[4] = float32(fnv1a(ip)%256) / 255.0

	// [New Feature] User-Agent Categorical Hash (Bot/Mobile/Browser)
	ua := r.Header.Get("User-Agent")
	vec[5] = float32(fnv1a(ua)%256) / 255.0

	// [New Feature] Query Parameter density
	queries := r.URL.Query()
	vec[6] = float32(math.Min(float64(len(queries))/20.0, 1.0))

	// Hashing path segments + query keys
	var segments []string
	if pathTrimmed := strings.Trim(pathStr, "/"); pathTrimmed != "" {
		segments = strings.Split(pathTrimmed, "/")
	}
	for k := range queries {
		segments = append(segments, k)
	}

	idx := 7
	for _, seg := range segments {
		if idx >= MaxTokenLen {
			break
		}
		h := fnv1a(seg)
		vec[idx] = float32(h%256) / 255.0
		idx++
	}
	return vecPtr
}

func fnv1a(s string) uint32 {
	const offset32 = uint32(2166136261)
	const prime32 = uint32(16777619)
	h := offset32
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= prime32
	}
	return h
}

// ────────────────────────────────────────────────
// Calibrated Softmax
// ────────────────────────────────────────────────

func CalibratedSoftmax(logits []float32, temperature float64) (probs []float32, topIdx int, topProb float64) {
	if temperature <= 0 {
		temperature = 1.0
	}
	// LOGGING FOR DEBUGGING
	if true { // Log EVERY request for debugging
		fmt.Printf("[debug] Raw logits from ONNX: %v\n", logits)
	}

	n := len(logits)
	probs = make([]float32, n)
	maxLogit := float64(logits[0])
	for i := 1; i < n; i++ {
		if v := float64(logits[i]); v > maxLogit {
			maxLogit = v
		}
	}
	var sumExp float64
	for i := 0; i < n; i++ {
		scaled := (float64(logits[i]) - maxLogit) / temperature
		e := math.Exp(scaled)
		probs[i] = float32(e)
		sumExp += e
	}
	topIdx = 0
	topProb = 0
	for i := 0; i < n; i++ {
		p := float64(probs[i]) / sumExp
		probs[i] = float32(p)
		if p > topProb {
			topProb = p
			topIdx = i
		}
	}
	if true {
		fmt.Printf("[debug] Temperature: %f, topIdx: %d, topProb: %f\n", temperature, topIdx, topProb)
	}
	return probs, topIdx, topProb
}

type InferenceRequest struct {
	Features []float32
	Result   chan<- InferenceResult
}

type InferenceResult struct {
	Logits []float32
	Err    error
}

// LayaEngine wraps the ONNX Runtime session or mock.
type LayaEngine struct {
	cfg         *config.Config
	numBackends int
	reqCh       chan InferenceRequest
	wg          sync.WaitGroup
	closeOnce   sync.Once

	ortInitialized bool
	useMock        bool
}

func NewLayaEngine(cfg *config.Config, numBackends int) (*LayaEngine, error) {
	e := &LayaEngine{
		cfg:         cfg,
		numBackends: numBackends,
		reqCh:       make(chan InferenceRequest, cfg.InferenceWorkers*4),
	}

	if !cfg.EnableAIRouting {
		e.useMock = true
		e.startWorkers()
		return e, nil
	}

	if err := e.initORT(); err != nil {
		fmt.Printf("[laya] WARN: ONNX init failed (%v), falling back to mock inference\n", err)
		e.useMock = true
	}

	e.startWorkers()
	return e, nil
}

func (e *LayaEngine) startWorkers() {
	workers := e.cfg.InferenceWorkers
	if workers < 1 {
		workers = 1
	}
	for i := 0; i < workers; i++ {
		e.wg.Add(1)
		go e.worker(i)
	}
}

func (e *LayaEngine) worker(id int) {
	defer e.wg.Done()
	for req := range e.reqCh {
		var res InferenceResult
		if e.useMock {
			res = e.mockInfer(req.Features)
		} else {
			res = e.ortInfer(req.Features)
		}
		req.Result <- res
	}
}

func (e *LayaEngine) mockInfer(features []float32) InferenceResult {
	logits := make([]float32, e.numBackends)
	methodCode := features[0]
	contentWeight := features[1]

	switch {
	case methodCode >= 4:
		if e.numBackends >= 3 {
			logits[2] = 3.0
		}
	case contentWeight > 0.5:
		logits[0] = 3.0
	default:
		if e.numBackends >= 2 {
			logits[1] = 3.0
		}
	}
	return InferenceResult{Logits: logits}
}

func (e *LayaEngine) Infer(ctx context.Context, features []float32) ([]float32, error) {
	// 1. Check Inference Cache (O(1) fast-path)
	var key [MaxTokenLen]float32
	copy(key[:], features)
	
	if val, ok := inferenceCache.Load(key); ok {
		entry := val.(cacheEntry)
		if time.Now().Before(entry.Expiry) {
			// Cache HIT
			cachedLogits := make([]float32, len(entry.Logits))
			copy(cachedLogits, entry.Logits[:])
			return cachedLogits, nil
		} else {
			inferenceCache.Delete(key) // Evict expired
		}
	}

	// 2. Lock-free Submission
	resultCh := make(chan InferenceResult, 1)
	req := InferenceRequest{Features: features, Result: resultCh}
	
	// Immediate fallback if all workers are saturated (no waiting)
	select {
	case e.reqCh <- req:
	default:
		return nil, fmt.Errorf("worker pool saturated, immediate fallback")
	}

	// 3. Wait for result (with hard timeout to prevent deadlocks)
	timeout := e.cfg.InferenceTimeout
	if timeout <= 0 {
		timeout = 10 * time.Millisecond
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case res := <-resultCh:
		if res.Err == nil && len(res.Logits) == e.numBackends && e.numBackends <= 3 {
			// Save to Cache for future requests (10-second TTL)
			var logitsArr [3]float32
			copy(logitsArr[:], res.Logits)
			inferenceCache.Store(key, cacheEntry{
				Logits: logitsArr,
				Expiry: time.Now().Add(10 * time.Second),
			})
		}
		return res.Logits, res.Err
	case <-timer.C:
		return nil, fmt.Errorf("inference result timeout")
	case <-ctx.Done():
		return nil, fmt.Errorf("request context canceled: %w", ctx.Err())
	}
}

func (e *LayaEngine) Shutdown() {
	e.closeOnce.Do(func() {
		close(e.reqCh)
		e.wg.Wait()
		e.cleanupORT()
	})
}

func (e *LayaEngine) IsMock() bool {
	return e.useMock
}
