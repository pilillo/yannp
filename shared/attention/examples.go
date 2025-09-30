package attention

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// ExampleScaledDotProductAttention demonstrates basic scaled dot-product attention
func ExampleScaledDotProductAttention() {
	fmt.Println("=== Scaled Dot-Product Attention Example ===")

	// Create sample data
	Q := mat.NewDense(3, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	K := mat.NewDense(3, 4, []float64{
		1, 1, 1, 1,
		2, 2, 2, 2,
		3, 3, 3, 3,
	})
	V := mat.NewDense(3, 4, []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	})

	// Create attention layer
	attention := NewScaledDotProductAttention(0.1)
	attention.SetTraining(false)

	// Forward pass
	output := attention.Forward(Q, K, V, nil)

	qr, qc := Q.Dims()
	fmt.Printf("Query shape: (%d, %d)\n", qr, qc)
	kr, kc := K.Dims()
	fmt.Printf("Key shape: (%d, %d)\n", kr, kc)
	vr, vc := V.Dims()
	fmt.Printf("Value shape: (%d, %d)\n", vr, vc)
	or, oc := output.Dims()
	fmt.Printf("Output shape: (%d, %d)\n", or, oc)
	fmt.Println("Output:")
	fmt.Println(mat.Formatted(output, mat.Prefix("  ")))
}

// ExampleMultiHeadAttention demonstrates multi-head attention
func ExampleMultiHeadAttention() {
	fmt.Println("=== Multi-Head Attention Example ===")

	// Create sample data
	X := mat.NewDense(2, 8, nil)
	randGen := rand.New(rand.NewSource(42))
	for i := 0; i < 2; i++ {
		for j := 0; j < 8; j++ {
			X.Set(i, j, randGen.NormFloat64())
		}
	}

	// Create multi-head attention
	mha := NewMultiHeadAttention(4, 8, 2, 2, 0.1)
	mha.SetTraining(false)

	// Forward pass
	output := mha.Forward(X, nil)

	xr, xc := X.Dims()
	fmt.Printf("Input shape: (%d, %d)\n", xr, xc)
	or, oc := output.Dims()
	fmt.Printf("Output shape: (%d, %d)\n", or, oc)
	fmt.Println("Number of heads: 4")
	fmt.Println("Model dimension: 8")
}

// ExamplePositionalEncoding demonstrates positional encoding
func ExamplePositionalEncoding() {
	fmt.Println("=== Positional Encoding Example ===")

	// Create positional encoding
	pe := NewPositionalEncoding(4, 10)

	// Create sample input
	X := mat.NewDense(3, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})

	// Add positional encoding
	output := pe.Forward(X)

	xr, xc := X.Dims()
	fmt.Printf("Input shape: (%d, %d)\n", xr, xc)
	or, oc := output.Dims()
	fmt.Printf("Output shape: (%d, %d)\n", or, oc)
	fmt.Println("Positional encoding added to input")
}

// ExampleAttentionMasks demonstrates different attention masks
func ExampleAttentionMasks() {
	fmt.Println("=== Attention Masks Example ===")

	seqLen := 5

	// Causal mask
	causalMask := NewCausalMask()
	causalMatrix := causalMask.GetMask(seqLen)
	fmt.Println("Causal mask:")
	fmt.Println(mat.Formatted(causalMatrix, mat.Prefix("  ")))

	// Padding mask
	lengths := []int{3, 5, 2}
	paddingMask := NewPaddingMask(lengths)
	paddingMatrix := paddingMask.GetMask(seqLen)
	fmt.Println("Padding mask (lengths:", lengths, "):")
	fmt.Println(mat.Formatted(paddingMatrix, mat.Prefix("  ")))

	// Look-ahead mask
	lookAheadMask := NewLookAheadMask(2)
	lookAheadMatrix := lookAheadMask.GetMask(seqLen)
	fmt.Println("Look-ahead mask (steps=2):")
	fmt.Println(mat.Formatted(lookAheadMatrix, mat.Prefix("  ")))
}

// ExampleAttentionBlock demonstrates a complete attention block
func ExampleAttentionBlock() {
	fmt.Println("=== Attention Block Example ===")

	// Create self-attention layer
	selfAttention := NewSelfAttentionLayer(4, 8, 2, 2, 0.1)

	// Create attention block
	block := NewAttentionBlock(selfAttention, 8, 16, 0.1)
	block.SetTraining(false)

	// Create sample input
	X := mat.NewDense(2, 8, nil)
	randGen := rand.New(rand.NewSource(42))
	for i := 0; i < 2; i++ {
		for j := 0; j < 8; j++ {
			X.Set(i, j, randGen.NormFloat64())
		}
	}

	// Forward pass
	output := block.Forward(X, nil)

	xr, xc := X.Dims()
	fmt.Printf("Input shape: (%d, %d)\n", xr, xc)
	or, oc := output.Dims()
	fmt.Printf("Output shape: (%d, %d)\n", or, oc)
	fmt.Println("Attention block with residual connections and layer norm")
}

// ExampleAttentionVisualization demonstrates attention visualization
func ExampleAttentionVisualization() {
	fmt.Println("=== Attention Visualization Example ===")

	// Create sample attention weights
	attentionWeights := mat.NewDense(4, 4, []float64{
		0.1, 0.2, 0.3, 0.4,
		0.2, 0.3, 0.3, 0.2,
		0.3, 0.3, 0.2, 0.2,
		0.4, 0.2, 0.2, 0.2,
	})

	// Create visualization utils
	viz := NewAttentionVisualization()

	// Get heatmap
	heatmap := viz.GetAttentionHeatmap(attentionWeights)
	fmt.Println("Attention heatmap:")
	for i, row := range heatmap {
		fmt.Printf("  Row %d: %v\n", i, row)
	}

	// Get statistics
	stats := viz.GetAttentionStats(attentionWeights)
	fmt.Println("Attention statistics:")
	for key, value := range stats {
		fmt.Printf("  %s: %.4f\n", key, value)
	}
}

// ExampleAttentionConfig demonstrates attention configuration
func ExampleAttentionConfig() {
	fmt.Println("=== Attention Configuration Example ===")

	// Create configuration
	config := NewAttentionConfig(8, 512)
	config.DropoutRate = 0.1
	config.MaxSeqLen = 1024

	// Validate configuration
	err := config.Validate()
	if err != nil {
		fmt.Printf("Configuration error: %v\n", err)
		return
	}

	fmt.Println("Configuration:")
	fmt.Printf("  Number of heads: %d\n", config.NumHeads)
	fmt.Printf("  Model dimension: %d\n", config.DModel)
	fmt.Printf("  Key dimension: %d\n", config.DK)
	fmt.Printf("  Value dimension: %d\n", config.DV)
	fmt.Printf("  Dropout rate: %.2f\n", config.DropoutRate)
	fmt.Printf("  Max sequence length: %d\n", config.MaxSeqLen)
	fmt.Printf("  Scale factor: %.4f\n", config.ScaleFactor)
}

// ExampleAttentionUtils demonstrates attention utilities
func ExampleAttentionUtils() {
	fmt.Println("=== Attention Utils Example ===")

	utils := NewAttentionUtils()

	// Create sample data
	Q := mat.NewDense(3, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	K := mat.NewDense(3, 4, []float64{
		1, 1, 1, 1,
		2, 2, 2, 2,
		3, 3, 3, 3,
	})
	V := mat.NewDense(3, 4, []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	})

	// Compute attention scores
	scale := 1.0 / math.Sqrt(4.0)
	scores := utils.ComputeAttentionScores(Q, K, scale)
	sr, sc := scores.Dims()
	fmt.Printf("Attention scores shape: (%d, %d)\n", sr, sc)

	// Apply softmax
	attentionWeights := utils.ApplySoftmax(scores)
	awr, awc := attentionWeights.Dims()
	fmt.Printf("Attention weights shape: (%d, %d)\n", awr, awc)

	// Compute final output
	output := utils.ComputeAttentionOutput(attentionWeights, V)
	or, oc := output.Dims()
	fmt.Printf("Final output shape: (%d, %d)\n", or, oc)
}

// ExampleCompleteAttentionPipeline demonstrates a complete attention pipeline
func ExampleCompleteAttentionPipeline() {
	fmt.Println("=== Complete Attention Pipeline Example ===")

	// Create sample data
	X := mat.NewDense(2, 8, nil)
	randGen := rand.New(rand.NewSource(42))
	for i := 0; i < 2; i++ {
		for j := 0; j < 8; j++ {
			X.Set(i, j, randGen.NormFloat64())
		}
	}

	// Create positional encoding
	pe := NewPositionalEncoding(8, 100)
	X = pe.Forward(X)

	// Create multi-head attention
	mha := NewMultiHeadAttention(4, 8, 2, 2, 0.1)
	mha.SetTraining(false)

	// Create causal mask
	mask := NewCausalMask()
	attentionMask := mask.GetMask(2)

	// Forward pass
	output := mha.Forward(X, attentionMask)

	xr, xc := X.Dims()
	fmt.Printf("Input shape: (%d, %d)\n", xr, xc)
	or, oc := output.Dims()
	fmt.Printf("Output shape: (%d, %d)\n", or, oc)
	fmt.Println("Pipeline: Input -> Positional Encoding -> Multi-Head Attention -> Output")
}
