// Dummy backend server for testing and benchmarking.
// Returns 200 OK with a small JSON payload identifying itself.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	port := flag.Int("port", 9001, "Port to listen on")
	flag.Parse()

	name := fmt.Sprintf("backend-%d", *port)

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Backend", name)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"backend":"%s","path":"%s","method":"%s"}`, name, r.URL.Path, r.Method)
	})

	addr := fmt.Sprintf(":%d", *port)
	srv := &http.Server{Addr: addr, Handler: mux}

	go func() {
		log.Printf("[backend] %s listening on %s", name, addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[backend] %s error: %v", name, err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Printf("[backend] %s shutting down", name)
}
