// L7 Load Balancer — main entry point.
//
// Initializes the ONNX inference engine, configures the proxy router,
// starts the HTTP server, and handles graceful shutdown on SIGINT/SIGTERM.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/l7-loadbalancer/config"
	"github.com/l7-loadbalancer/inference"
	"github.com/l7-loadbalancer/proxy"
)

func main() {
	// ── CLI flags ─────────────────────────────────
	listenAddr := flag.String("listen", ":8080", "Proxy listen address")
	modelPath := flag.String("model", "models/laya_l7_quantized.onnx", "Path to quantized ONNX model")
	ortLib := flag.String("ort-lib", "", "Path to ONNX Runtime shared library")
	workers := flag.Int("workers", 4, "Inference worker pool size")
	timeout := flag.Duration("timeout", 10*time.Millisecond, "Inference timeout per request")
	confidence := flag.Float64("confidence", 0.85, "Minimum AI confidence threshold")
	temperature := flag.Float64("temperature", 1.0, "Softmax temperature scaling")
	enableAI := flag.Bool("ai", true, "Enable AI-based routing (false = pure Round-Robin)")
	flag.Parse()

	// ── Build config ──────────────────────────────
	cfg := config.DefaultConfig()
	cfg.ListenAddr = *listenAddr
	cfg.ModelPath = *modelPath
	cfg.ONNXLibPath = *ortLib
	cfg.InferenceWorkers = *workers
	cfg.InferenceTimeout = *timeout
	cfg.ConfidenceThreshold = *confidence
	cfg.Temperature = *temperature
	cfg.EnableAIRouting = *enableAI

	// Allow backend override via environment (comma-separated).
	if envBackends := os.Getenv("LB_BACKENDS"); envBackends != "" {
		cfg.Backends = parseEnvBackends(envBackends)
	}

	// [Architecture Constraint] Cap backends at 15 to fit within token head budget
	// of the single-forward-pass transformer and avoid output layer allocation faults.
	if len(cfg.Backends) > 15 {
		log.Printf("[main] WARN: Truncating backends to 15. The Laya decision engine supports a strict maximum of 15 dynamically passed schemas.")
		cfg.Backends = cfg.Backends[:15]
	}

	// ── Initialize inference engine ───────────────
	engine, err := inference.NewLayaEngine(cfg, len(cfg.Backends))
	if err != nil {
		log.Fatalf("[main] failed to initialize inference engine: %v", err)
	}

	modeStr := "AI-routed"
	if engine.IsMock() {
		modeStr = "mock-inference (Round-Robin dominant)"
	}
	if !cfg.EnableAIRouting {
		modeStr = "pure Round-Robin"
	}

	// ── Initialize proxy router ───────────────────
	router, err := proxy.NewRouter(cfg, engine)
	if err != nil {
		log.Fatalf("[main] failed to initialize router: %v", err)
	}

	// ── HTTP server with timeouts ─────────────────
	srv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           router,
		ReadHeaderTimeout: 5 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    1 << 20, // 1 MB
	}

	// ── Start server ──────────────────────────────
	go func() {
		log.Printf("[main] L7 Load Balancer starting on %s [mode: %s]", cfg.ListenAddr, modeStr)
		log.Printf("[main] Backends: %d | Workers: %d | Confidence: %.2f | Temperature: %.2f",
			len(cfg.Backends), cfg.InferenceWorkers, cfg.ConfidenceThreshold, cfg.Temperature)

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[main] server error: %v", err)
		}
	}()

	// ── Graceful shutdown ─────────────────────────
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	log.Printf("[main] received signal %v, shutting down...", sig)

	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.GracefulShutdownTimeout)
	defer cancel()

	// Drain HTTP connections.
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("[main] HTTP shutdown error: %v", err)
	}

	// Drain inference workers.
	engine.Shutdown()

	log.Println("[main] shutdown complete")
}

// parseEnvBackends parses "name=url,name=url,..." into backend configs.
func parseEnvBackends(raw string) []config.Backend {
	var backends []config.Backend
	entries := splitNonEmpty(raw, ',')
	for _, entry := range entries {
		parts := splitNonEmpty(entry, '=')
		if len(parts) == 2 {
			backends = append(backends, config.Backend{
				Name: parts[0],
				URL:  parts[1],
			})
		}
	}
	return backends
}

func splitNonEmpty(s string, sep byte) []string {
	var result []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == sep {
			part := s[start:i]
			if len(part) > 0 {
				result = append(result, part)
			}
			start = i + 1
		}
	}
	return result
}

func init() {
	// Structured log prefix for all output.
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)
	log.SetPrefix("")
	fmt.Println(`
 ╔═══════════════════════════════════════════════════╗
 ║  L7 Load Balancer — AI-Assisted Reverse Proxy     ║
 ║  Engine: Laya Typed Decisions (ONNX INT8)         ║
 ╚═══════════════════════════════════════════════════╝`)
}
