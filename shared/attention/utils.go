package attention

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// AttentionUtils provides utility functions for attention mechanisms
type AttentionUtils struct{}

// NewAttentionUtils creates a new attention utils instance
func NewAttentionUtils() *AttentionUtils {
	return &AttentionUtils{}
}

// ComputeAttentionScores computes attention scores between queries and keys
func (au *AttentionUtils) ComputeAttentionScores(Q, K *mat.Dense, scale float64) *mat.Dense {
	var scores mat.Dense
	scores.Mul(Q, K.T())
	scores.Scale(scale, &scores)
	return &scores
}

// ApplySoftmax applies softmax to the attention scores
func (au *AttentionUtils) ApplySoftmax(scores *mat.Dense) *mat.Dense {
	rows, cols := scores.Dims()
	output := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		// Get row
		row := mat.Row(nil, i, scores)

		// Find max for numerical stability
		maxVal := row[0]
		for _, val := range row {
			if val > maxVal {
				maxVal = val
			}
		}

		// Compute exp and sum
		expSum := 0.0
		expVals := make([]float64, len(row))
		for j, val := range row {
			expVal := math.Exp(val - maxVal)
			expVals[j] = expVal
			expSum += expVal
		}

		// Normalize
		for j, expVal := range expVals {
			output.Set(i, j, expVal/expSum)
		}
	}

	return output
}

// ApplyMask applies a mask to attention scores
func (au *AttentionUtils) ApplyMask(scores, mask *mat.Dense) *mat.Dense {
	var masked mat.Dense
	masked.MulElem(scores, mask)
	return &masked
}

// ComputeAttentionOutput computes the final attention output
func (au *AttentionUtils) ComputeAttentionOutput(attentionWeights, V *mat.Dense) *mat.Dense {
	var output mat.Dense
	output.Mul(attentionWeights, V)
	return &output
}

// ReshapeForMultiHead reshapes input for multi-head attention
func (au *AttentionUtils) ReshapeForMultiHead(X *mat.Dense, numHeads, headDim int) *mat.Dense {
	// This is a simplified implementation
	// In practice, you'd want to properly reshape the tensor
	return X
}

// ConcatenateHeads concatenates attention heads
func (au *AttentionUtils) ConcatenateHeads(headOutputs []*mat.Dense) *mat.Dense {
	// This is a simplified implementation
	// In practice, you'd want to properly concatenate the heads
	return headOutputs[0]
}

// SplitHeads splits input into multiple attention heads
func (au *AttentionUtils) SplitHeads(X *mat.Dense, numHeads, headDim int) []*mat.Dense {
	// This is a simplified implementation
	// In practice, you'd want to properly split the tensor
	heads := make([]*mat.Dense, numHeads)
	for i := 0; i < numHeads; i++ {
		heads[i] = X
	}
	return heads
}

// AttentionVisualization provides utilities for visualizing attention
type AttentionVisualization struct{}

// NewAttentionVisualization creates a new attention visualization instance
func NewAttentionVisualization() *AttentionVisualization {
	return &AttentionVisualization{}
}

// GetAttentionHeatmap creates a heatmap of attention weights
func (av *AttentionVisualization) GetAttentionHeatmap(attentionWeights *mat.Dense) [][]float64 {
	rows, cols := attentionWeights.Dims()
	heatmap := make([][]float64, rows)

	for i := 0; i < rows; i++ {
		heatmap[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			heatmap[i][j] = attentionWeights.At(i, j)
		}
	}

	return heatmap
}

// GetAttentionStats computes statistics about attention weights
func (av *AttentionVisualization) GetAttentionStats(attentionWeights *mat.Dense) map[string]float64 {
	rows, cols := attentionWeights.Dims()
	stats := make(map[string]float64)

	// Compute mean attention weight
	sum := 0.0
	count := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			sum += attentionWeights.At(i, j)
			count++
		}
	}
	stats["mean"] = sum / float64(count)

	// Compute max attention weight
	maxVal := attentionWeights.At(0, 0)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if attentionWeights.At(i, j) > maxVal {
				maxVal = attentionWeights.At(i, j)
			}
		}
	}
	stats["max"] = maxVal

	// Compute min attention weight
	minVal := attentionWeights.At(0, 0)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if attentionWeights.At(i, j) < minVal {
				minVal = attentionWeights.At(i, j)
			}
		}
	}
	stats["min"] = minVal

	// Compute attention entropy (measure of attention spread)
	entropy := 0.0
	for i := 0; i < rows; i++ {
		row := mat.Row(nil, i, attentionWeights)
		for _, val := range row {
			if val > 0 {
				entropy -= val * math.Log2(val)
			}
		}
	}
	stats["entropy"] = entropy / float64(rows)

	return stats
}

// AttentionConfig holds configuration for attention mechanisms
type AttentionConfig struct {
	NumHeads    int
	DModel      int
	DK          int
	DV          int
	DropoutRate float64
	MaxSeqLen   int
	UseBias     bool
	ScaleFactor float64
}

// NewAttentionConfig creates a new attention configuration
func NewAttentionConfig(numHeads, dModel int) *AttentionConfig {
	return &AttentionConfig{
		NumHeads:    numHeads,
		DModel:      dModel,
		DK:          dModel / numHeads,
		DV:          dModel / numHeads,
		DropoutRate: 0.1,
		MaxSeqLen:   512,
		UseBias:     true,
		ScaleFactor: 1.0 / math.Sqrt(float64(dModel/numHeads)),
	}
}

// Validate validates the attention configuration
func (ac *AttentionConfig) Validate() error {
	if ac.NumHeads <= 0 {
		return &AttentionError{Message: "num_heads must be positive"}
	}
	if ac.DModel <= 0 {
		return &AttentionError{Message: "d_model must be positive"}
	}
	if ac.DModel%ac.NumHeads != 0 {
		return &AttentionError{Message: "d_model must be divisible by num_heads"}
	}
	if ac.DropoutRate < 0 || ac.DropoutRate >= 1 {
		return &AttentionError{Message: "dropout_rate must be in [0, 1)"}
	}
	return nil
}

// AttentionError represents an error in attention operations
type AttentionError struct {
	Message string
}

func (ae *AttentionError) Error() string {
	return "AttentionError: " + ae.Message
}

// CreateAttentionMask creates various types of attention masks
func CreateAttentionMask(maskType string, seqLen int, args ...interface{}) (*mat.Dense, error) {
	switch maskType {
	case "causal":
		return NewCausalMask().GetMask(seqLen), nil
	case "padding":
		if len(args) > 0 {
			if lengths, ok := args[0].([]int); ok {
				return NewPaddingMask(lengths).GetMask(seqLen), nil
			}
		}
		return NewNoMask().GetMask(seqLen), nil
	case "lookahead":
		lookAheadSteps := 1
		if len(args) > 0 {
			if steps, ok := args[0].(int); ok {
				lookAheadSteps = steps
			}
		}
		return NewLookAheadMask(lookAheadSteps).GetMask(seqLen), nil
	case "none":
		return NewNoMask().GetMask(seqLen), nil
	default:
		return nil, &AttentionError{Message: "unknown mask type: " + maskType}
	}
}
