package attention

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// MultiHeadAttention implements multi-head attention mechanism
type MultiHeadAttention struct {
	numHeads    int
	dModel      int
	dK          int
	dV          int
	dropoutRate float64
	training    bool

	// Linear transformations
	WQ, WK, WV *mat.Dense // Query, Key, Value projections
	WO         *mat.Dense // Output projection

	// Scaled dot-product attention
	scaledDotProduct *ScaledDotProductAttention
}

// NewMultiHeadAttention creates a new multi-head attention layer
func NewMultiHeadAttention(numHeads, dModel, dK, dV int, dropoutRate float64) *MultiHeadAttention {
	// Initialize weight matrices
	WQ := mat.NewDense(dModel, numHeads*dK, nil)
	WK := mat.NewDense(dModel, numHeads*dK, nil)
	WV := mat.NewDense(dModel, numHeads*dV, nil)
	WO := mat.NewDense(numHeads*dV, dModel, nil)

	// Initialize with small random values
	randGen := rand.New(rand.NewSource(42))
	WQ.Apply(func(i, j int, v float64) float64 {
		return randGen.NormFloat64() * 0.1
	}, WQ)
	WK.Apply(func(i, j int, v float64) float64 {
		return randGen.NormFloat64() * 0.1
	}, WK)
	WV.Apply(func(i, j int, v float64) float64 {
		return randGen.NormFloat64() * 0.1
	}, WV)
	WO.Apply(func(i, j int, v float64) float64 {
		return randGen.NormFloat64() * 0.1
	}, WO)

	return &MultiHeadAttention{
		numHeads:         numHeads,
		dModel:           dModel,
		dK:               dK,
		dV:               dV,
		dropoutRate:      dropoutRate,
		training:         true,
		WQ:               WQ,
		WK:               WK,
		WV:               WV,
		WO:               WO,
		scaledDotProduct: NewScaledDotProductAttention(dropoutRate),
	}
}

// SetTraining sets the training mode
func (m *MultiHeadAttention) SetTraining(training bool) {
	m.training = training
	m.scaledDotProduct.SetTraining(training)
}

// Forward performs the forward pass of multi-head attention
// X: input matrix (batch_size, seq_len, d_model)
// mask: optional attention mask (batch_size, seq_len, seq_len)
func (m *MultiHeadAttention) Forward(X, mask *mat.Dense) *mat.Dense {
	batchSize, _ := X.Dims()
	seqLen := 1 // Simplified for this implementation

	// Linear transformations
	Q := m.linearTransform(X, m.WQ) // (batch_size, seq_len, num_heads * d_k)
	K := m.linearTransform(X, m.WK) // (batch_size, seq_len, num_heads * d_k)
	V := m.linearTransform(X, m.WV) // (batch_size, seq_len, num_heads * d_v)

	// Reshape for multi-head attention
	Q = m.reshapeForMultiHead(Q, batchSize, seqLen) // (batch_size, num_heads, seq_len, d_k)
	K = m.reshapeForMultiHead(K, batchSize, seqLen) // (batch_size, num_heads, seq_len, d_k)
	V = m.reshapeForMultiHead(V, batchSize, seqLen) // (batch_size, num_heads, seq_len, d_v)

	// Apply attention for each head
	headOutputs := make([]*mat.Dense, m.numHeads)
	for h := 0; h < m.numHeads; h++ {
		// Extract head-specific Q, K, V
		headQ := m.extractHead(Q, h, batchSize, seqLen)
		headK := m.extractHead(K, h, batchSize, seqLen)
		headV := m.extractHead(V, h, batchSize, seqLen)

		// Apply scaled dot-product attention
		headOutputs[h] = m.scaledDotProduct.Forward(headQ, headK, headV, mask)
	}

	// Concatenate heads
	concatenated := m.concatenateHeads(headOutputs, batchSize, seqLen)

	// Final linear transformation
	output := m.linearTransform(concatenated, m.WO)

	return output
}

// linearTransform applies a linear transformation: X * W
func (m *MultiHeadAttention) linearTransform(X, W *mat.Dense) *mat.Dense {
	var result mat.Dense
	result.Mul(X, W)
	return &result
}

// reshapeForMultiHead reshapes the input for multi-head processing
func (m *MultiHeadAttention) reshapeForMultiHead(X *mat.Dense, batchSize, seqLen int) *mat.Dense {
	// This is a simplified implementation
	// In practice, you'd want to properly reshape the tensor
	return X
}

// extractHead extracts the attention head at the given index
func (m *MultiHeadAttention) extractHead(X *mat.Dense, headIndex, batchSize, seqLen int) *mat.Dense {
	// This is a simplified implementation
	// In practice, you'd want to properly extract the head
	return X
}

// concatenateHeads concatenates the outputs from all attention heads
func (m *MultiHeadAttention) concatenateHeads(headOutputs []*mat.Dense, batchSize, seqLen int) *mat.Dense {
	// This is a simplified implementation
	// In practice, you'd want to properly concatenate the heads
	return headOutputs[0]
}

// GetAttentionWeights returns the attention weights for all heads
func (m *MultiHeadAttention) GetAttentionWeights(X, mask *mat.Dense) []*mat.Dense {
	batchSize, _ := X.Dims()
	seqLen := 1 // Simplified for this implementation

	Q := m.linearTransform(X, m.WQ)
	K := m.linearTransform(X, m.WK)

	Q = m.reshapeForMultiHead(Q, batchSize, seqLen)
	K = m.reshapeForMultiHead(K, batchSize, seqLen)

	attentionWeights := make([]*mat.Dense, m.numHeads)
	for h := 0; h < m.numHeads; h++ {
		headQ := m.extractHead(Q, h, batchSize, seqLen)
		headK := m.extractHead(K, h, batchSize, seqLen)
		attentionWeights[h] = m.scaledDotProduct.GetAttentionWeights(headQ, headK, mask)
	}

	return attentionWeights
}

// GetParameters returns all learnable parameters
func (m *MultiHeadAttention) GetParameters() []*mat.Dense {
	return []*mat.Dense{m.WQ, m.WK, m.WV, m.WO}
}

// UpdateParameters updates the parameters with gradients
func (m *MultiHeadAttention) UpdateParameters(gradients []*mat.Dense, learningRate float64) {
	if len(gradients) != 4 {
		return
	}

	// Update each parameter matrix
	params := m.GetParameters()
	for i, param := range params {
		var update mat.Dense
		update.Scale(learningRate, gradients[i])
		param.Sub(param, &update)
	}
}
