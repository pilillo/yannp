package attention

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// ScaledDotProductAttention implements the scaled dot-product attention mechanism
type ScaledDotProductAttention struct {
	dropoutRate float64
	training    bool
}

// NewScaledDotProductAttention creates a new scaled dot-product attention layer
func NewScaledDotProductAttention(dropoutRate float64) *ScaledDotProductAttention {
	return &ScaledDotProductAttention{
		dropoutRate: dropoutRate,
		training:    true,
	}
}

// SetTraining sets the training mode
func (s *ScaledDotProductAttention) SetTraining(training bool) {
	s.training = training
}

// Forward performs the forward pass of scaled dot-product attention
// Q: query matrix (batch_size, seq_len, d_k)
// K: key matrix (batch_size, seq_len, d_k)
// V: value matrix (batch_size, seq_len, d_v)
// mask: optional attention mask (batch_size, seq_len, seq_len)
func (s *ScaledDotProductAttention) Forward(Q, K, V, mask *mat.Dense) *mat.Dense {
	// Get dimensions
	_, dK := Q.Dims()

	// Compute attention scores: Q * K^T / sqrt(d_k)
	var scores mat.Dense
	scores.Mul(Q, K.T())
	scores.Scale(1.0/math.Sqrt(float64(dK)), &scores)

	// Apply mask if provided (add mask to scores for causal masking)
	if mask != nil {
		// Mask should contain 0 for allowed positions and -Inf for masked positions
		// We add the mask to scores so that masked positions become -Inf
		rows, cols := scores.Dims()
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				maskVal := mask.At(i, j)
				// If mask is 1.0, keep score as is (allowed position)
				// If mask is -Inf, set score to -Inf (masked position)
				if math.IsInf(maskVal, -1) {
					scores.Set(i, j, maskVal)
				}
			}
		}
	}

	// Apply softmax to get attention weights
	attentionWeights := s.softmax(&scores)

	// Apply dropout during training
	if s.training && s.dropoutRate > 0 {
		attentionWeights = s.applyDropout(attentionWeights)
	}

	// Compute weighted sum: attention_weights * V
	var output mat.Dense
	output.Mul(attentionWeights, V)

	return &output
}

// softmax applies softmax to the last dimension of the input matrix
func (s *ScaledDotProductAttention) softmax(x *mat.Dense) *mat.Dense {
	rows, cols := x.Dims()
	output := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		// Get row
		row := mat.Row(nil, i, x)

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

// applyDropout applies dropout to the attention weights
func (s *ScaledDotProductAttention) applyDropout(x *mat.Dense) *mat.Dense {
	// This is a simplified dropout implementation
	// In practice, you'd want to use a proper random number generator
	rows, cols := x.Dims()
	output := mat.NewDense(rows, cols, nil)

	output.Apply(func(i, j int, v float64) float64 {
		// Simplified: just scale by (1 - dropout_rate)
		return v * (1.0 - s.dropoutRate)
	}, x)

	return output
}

// GetAttentionWeights returns the attention weights for visualization
func (s *ScaledDotProductAttention) GetAttentionWeights(Q, K, mask *mat.Dense) *mat.Dense {
	_, dK := Q.Dims()

	var scores mat.Dense
	scores.Mul(Q, K.T())
	scores.Scale(1.0/math.Sqrt(float64(dK)), &scores)

	if mask != nil {
		rows, cols := scores.Dims()
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				maskVal := mask.At(i, j)
				if math.IsInf(maskVal, -1) {
					scores.Set(i, j, maskVal)
				}
			}
		}
	}

	return s.softmax(&scores)
}
