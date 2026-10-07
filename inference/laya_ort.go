//go:build cgo
// +build cgo

package inference

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

func (e *LayaEngine) initORT() error {
	libPath := e.cfg.ONNXLibPath
	if libPath == "" {
		libPath = "onnxruntime.dll"
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("ort env init: %w", err)
	}

	workers := e.cfg.InferenceWorkers
	if workers < 1 {
		workers = 1
	}
	
	// Create a channel to pool ortContexts for workers
	ortContexts = make(chan *ortContext, workers)
	for i := 0; i < workers; i++ {
		ctx, err := createOrtContext(e)
		if err != nil {
			return fmt.Errorf("failed to create ort context: %w", err)
		}
		ortContexts <- ctx
	}

	e.ortInitialized = true
	return nil
}

type ortContext struct {
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	session      *ort.AdvancedSession
}

var ortContexts chan *ortContext

func createOrtContext(e *LayaEngine) (*ortContext, error) {
	inputShape := ort.NewShape(1, int64(MaxTokenLen))
	inputData := make([]float32, MaxTokenLen)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("input tensor: %w", err)
	}

	outputShape := ort.NewShape(1, int64(e.numBackends))
	outputData := make([]float32, e.numBackends)
	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		e.cfg.ModelPath,
		[]string{"input"},
		[]string{"output"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		nil,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("session create: %w", err)
	}

	return &ortContext{
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
		session:      session,
	}, nil
}

func (e *LayaEngine) ortInfer(features []float32) InferenceResult {
	// Grab an available ORT context from the pool
	ctx := <-ortContexts
	defer func() { ortContexts <- ctx }() // return it when done

	// Copy new features into the pre-allocated input tensor
	inData := ctx.inputTensor.GetData()
	copy(inData, features)

	if err := ctx.session.Run(); err != nil {
		return InferenceResult{Err: fmt.Errorf("session run: %w", err)}
	}

	// Copy out the results from the output tensor
	logits := make([]float32, e.numBackends)
	copy(logits, ctx.outputTensor.GetData())
	return InferenceResult{Logits: logits}
}

func (e *LayaEngine) cleanupORT() {
	if e.ortInitialized {
		close(ortContexts)
		for ctx := range ortContexts {
			ctx.session.Destroy()
			ctx.inputTensor.Destroy()
			ctx.outputTensor.Destroy()
		}
		_ = ort.DestroyEnvironment()
	}
}
