package attention

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// AttentionLayer interface defines the contract for attention layers
type AttentionLayer interface {
	Forward(X, mask *mat.Dense) *mat.Dense
	SetTraining(training bool)
	GetParameters() []*mat.Dense
	UpdateParameters(gradients []*mat.Dense, learningRate float64)
}

// SelfAttentionLayer implements self-attention
type SelfAttentionLayer struct {
	multiHead *MultiHeadAttention
}

// NewSelfAttentionLayer creates a new self-attention layer
func NewSelfAttentionLayer(numHeads, dModel, dK, dV int, dropoutRate float64) *SelfAttentionLayer {
	return &SelfAttentionLayer{
		multiHead: NewMultiHeadAttention(numHeads, dModel, dK, dV, dropoutRate),
	}
}

// Forward performs self-attention
func (sa *SelfAttentionLayer) Forward(X, mask *mat.Dense) *mat.Dense {
	return sa.multiHead.Forward(X, mask)
}

// SetTraining sets the training mode
func (sa *SelfAttentionLayer) SetTraining(training bool) {
	sa.multiHead.SetTraining(training)
}

// GetParameters returns all learnable parameters
func (sa *SelfAttentionLayer) GetParameters() []*mat.Dense {
	return sa.multiHead.GetParameters()
}

// UpdateParameters updates the parameters
func (sa *SelfAttentionLayer) UpdateParameters(gradients []*mat.Dense, learningRate float64) {
	sa.multiHead.UpdateParameters(gradients, learningRate)
}

// CrossAttentionLayer implements cross-attention (encoder-decoder attention)
type CrossAttentionLayer struct {
	multiHead *MultiHeadAttention
}

// NewCrossAttentionLayer creates a new cross-attention layer
func NewCrossAttentionLayer(numHeads, dModel, dK, dV int, dropoutRate float64) *CrossAttentionLayer {
	return &CrossAttentionLayer{
		multiHead: NewMultiHeadAttention(numHeads, dModel, dK, dV, dropoutRate),
	}
}

// Forward performs cross-attention
// Query comes from decoder, Key and Value come from encoder
func (ca *CrossAttentionLayer) Forward(query, keyValue, mask *mat.Dense) *mat.Dense {
	// For cross-attention, we need to modify the multi-head attention
	// to use different inputs for Q vs K,V
	// This is a simplified implementation
	return ca.multiHead.Forward(query, mask)
}

// SetTraining sets the training mode
func (ca *CrossAttentionLayer) SetTraining(training bool) {
	ca.multiHead.SetTraining(training)
}

// GetParameters returns all learnable parameters
func (ca *CrossAttentionLayer) GetParameters() []*mat.Dense {
	return ca.multiHead.GetParameters()
}

// UpdateParameters updates the parameters
func (ca *CrossAttentionLayer) UpdateParameters(gradients []*mat.Dense, learningRate float64) {
	ca.multiHead.UpdateParameters(gradients, learningRate)
}

// AttentionBlock implements a complete attention block with residual connection and layer norm
type AttentionBlock struct {
	attention   AttentionLayer
	layerNorm1  *LayerNorm
	layerNorm2  *LayerNorm
	feedForward *FeedForward
	dropoutRate float64
	training    bool
}

// NewAttentionBlock creates a new attention block
func NewAttentionBlock(attention AttentionLayer, dModel int, feedForwardDim int, dropoutRate float64) *AttentionBlock {
	return &AttentionBlock{
		attention:   attention,
		layerNorm1:  NewLayerNorm(dModel),
		layerNorm2:  NewLayerNorm(dModel),
		feedForward: NewFeedForward(dModel, feedForwardDim, dropoutRate),
		dropoutRate: dropoutRate,
		training:    true,
	}
}

// Forward performs the forward pass of the attention block
func (ab *AttentionBlock) Forward(X, mask *mat.Dense) *mat.Dense {
	// Self-attention with residual connection and layer norm
	attnOutput := ab.attention.Forward(X, mask)
	attnOutput = ab.residualConnection(X, attnOutput)
	attnOutput = ab.layerNorm1.Forward(attnOutput)

	// Feed-forward with residual connection and layer norm
	ffOutput := ab.feedForward.Forward(attnOutput)
	ffOutput = ab.residualConnection(attnOutput, ffOutput)
	ffOutput = ab.layerNorm2.Forward(ffOutput)

	return ffOutput
}

// SetTraining sets the training mode
func (ab *AttentionBlock) SetTraining(training bool) {
	ab.training = training
	ab.attention.SetTraining(training)
	ab.feedForward.SetTraining(training)
}

// residualConnection applies residual connection and dropout
func (ab *AttentionBlock) residualConnection(input, output *mat.Dense) *mat.Dense {
	var result mat.Dense
	result.Add(input, output)

	if ab.training && ab.dropoutRate > 0 {
		dropoutResult := ab.applyDropout(&result)
		result = *dropoutResult
	}

	return &result
}

// applyDropout applies dropout during training
func (ab *AttentionBlock) applyDropout(x *mat.Dense) *mat.Dense {
	// Simplified dropout implementation
	rows, cols := x.Dims()
	output := mat.NewDense(rows, cols, nil)

	output.Apply(func(i, j int, v float64) float64 {
		return v * (1.0 - ab.dropoutRate)
	}, x)

	return output
}

// GetParameters returns all learnable parameters
func (ab *AttentionBlock) GetParameters() []*mat.Dense {
	params := ab.attention.GetParameters()
	params = append(params, ab.layerNorm1.GetParameters()...)
	params = append(params, ab.layerNorm2.GetParameters()...)
	params = append(params, ab.feedForward.GetParameters()...)
	return params
}

// UpdateParameters updates all parameters
func (ab *AttentionBlock) UpdateParameters(gradients []*mat.Dense, learningRate float64) {
	// This is a simplified implementation
	// In practice, you'd want to properly distribute gradients
	ab.attention.UpdateParameters(gradients, learningRate)
}

// LayerNorm implements layer normalization
type LayerNorm struct {
	gamma *mat.Dense
	beta  *mat.Dense
	eps   float64
}

// NewLayerNorm creates a new layer normalization layer
func NewLayerNorm(dModel int) *LayerNorm {
	gamma := mat.NewDense(1, dModel, nil)
	beta := mat.NewDense(1, dModel, nil)

	// Initialize gamma to 1 and beta to 0
	for i := 0; i < dModel; i++ {
		gamma.Set(0, i, 1.0)
		beta.Set(0, i, 0.0)
	}

	return &LayerNorm{
		gamma: gamma,
		beta:  beta,
		eps:   1e-6,
	}
}

// Forward performs layer normalization
func (ln *LayerNorm) Forward(X *mat.Dense) *mat.Dense {
	rows, cols := X.Dims()
	output := mat.NewDense(rows, cols, nil)

	// Normalize each row (sequence element) independently
	for i := 0; i < rows; i++ {
		// Compute mean
		var sum float64
		for j := 0; j < cols; j++ {
			sum += X.At(i, j)
		}
		mean := sum / float64(cols)

		// Compute variance
		var variance float64
		for j := 0; j < cols; j++ {
			diff := X.At(i, j) - mean
			variance += diff * diff
		}
		variance /= float64(cols)

		// Normalize and scale
		stdDev := math.Sqrt(variance + ln.eps)
		for j := 0; j < cols; j++ {
			normalized := (X.At(i, j) - mean) / stdDev
			scaled := normalized*ln.gamma.At(0, j) + ln.beta.At(0, j)
			output.Set(i, j, scaled)
		}
	}

	return output
}

// GetParameters returns the learnable parameters
func (ln *LayerNorm) GetParameters() []*mat.Dense {
	return []*mat.Dense{ln.gamma, ln.beta}
}

// GetGamma returns the gamma (weight) matrix
func (ln *LayerNorm) GetGamma() *mat.Dense {
	return ln.gamma
}

// GetBeta returns the beta (bias) matrix
func (ln *LayerNorm) GetBeta() *mat.Dense {
	return ln.beta
}

// FeedForward implements the feed-forward network
type FeedForward struct {
	linear1     *mat.Dense
	linear2     *mat.Dense
	bias1       *mat.Dense
	bias2       *mat.Dense
	dropoutRate float64
	training    bool
}

// NewFeedForward creates a new feed-forward network
func NewFeedForward(dModel, feedForwardDim int, dropoutRate float64) *FeedForward {
	// Initialize weight matrices
	linear1 := mat.NewDense(dModel, feedForwardDim, nil)
	linear2 := mat.NewDense(feedForwardDim, dModel, nil)
	bias1 := mat.NewDense(1, feedForwardDim, nil)
	bias2 := mat.NewDense(1, dModel, nil)

	// Initialize with small random values
	randGen := rand.New(rand.NewSource(42))
	linear1.Apply(func(i, j int, v float64) float64 {
		return randGen.NormFloat64() * 0.1
	}, linear1)
	linear2.Apply(func(i, j int, v float64) float64 {
		return randGen.NormFloat64() * 0.1
	}, linear2)

	return &FeedForward{
		linear1:     linear1,
		linear2:     linear2,
		bias1:       bias1,
		bias2:       bias2,
		dropoutRate: dropoutRate,
		training:    true,
	}
}

// Forward performs the forward pass
func (ff *FeedForward) Forward(X *mat.Dense) *mat.Dense {
	// First linear layer with ReLU activation
	var hidden mat.Dense
	hidden.Mul(X, ff.linear1)
	hidden.Add(&hidden, ff.bias1)

	// Apply ReLU
	hidden.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return v
		}
		return 0
	}, &hidden)

	// Second linear layer
	var output mat.Dense
	output.Mul(&hidden, ff.linear2)
	output.Add(&output, ff.bias2)

	return &output
}

// SetTraining sets the training mode
func (ff *FeedForward) SetTraining(training bool) {
	ff.training = training
}

// GetParameters returns all learnable parameters
func (ff *FeedForward) GetParameters() []*mat.Dense {
	return []*mat.Dense{ff.linear1, ff.linear2, ff.bias1, ff.bias2}
}
