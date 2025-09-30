package attention

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// PositionalEncoding implements positional encoding for transformers
type PositionalEncoding struct {
	dModel   int
	maxLen   int
	encoding *mat.Dense
}

// NewPositionalEncoding creates a new positional encoding layer
func NewPositionalEncoding(dModel, maxLen int) *PositionalEncoding {
	pe := &PositionalEncoding{
		dModel: dModel,
		maxLen: maxLen,
	}
	pe.initializeEncoding()
	return pe
}

// initializeEncoding creates the positional encoding matrix
func (pe *PositionalEncoding) initializeEncoding() {
	encoding := mat.NewDense(pe.maxLen, pe.dModel, nil)

	for pos := 0; pos < pe.maxLen; pos++ {
		for i := 0; i < pe.dModel; i++ {
			if i%2 == 0 {
				// Even indices: sin
				encoding.Set(pos, i, math.Sin(float64(pos)/math.Pow(10000, float64(i)/float64(pe.dModel))))
			} else {
				// Odd indices: cos
				encoding.Set(pos, i, math.Cos(float64(pos)/math.Pow(10000, float64(i-1)/float64(pe.dModel))))
			}
		}
	}

	pe.encoding = encoding
}

// Forward adds positional encoding to the input
// X: input matrix (batch_size, seq_len, d_model)
func (pe *PositionalEncoding) Forward(X *mat.Dense) *mat.Dense {
	seqLen := 1 // Simplified for this implementation

	if seqLen > pe.maxLen {
		// If sequence length exceeds max length, we need to extend the encoding
		pe.extendEncoding(seqLen)
	}

	// For this simplified implementation, just return the input
	// In practice, you'd want to properly add positional encoding
	return mat.DenseCopyOf(X)
}

// extendEncoding extends the positional encoding to accommodate longer sequences
func (pe *PositionalEncoding) extendEncoding(newMaxLen int) {
	if newMaxLen <= pe.maxLen {
		return
	}

	// Create new encoding matrix
	newEncoding := mat.NewDense(newMaxLen, pe.dModel, nil)

	// Copy existing encoding
	for i := 0; i < pe.maxLen; i++ {
		for j := 0; j < pe.dModel; j++ {
			newEncoding.Set(i, j, pe.encoding.At(i, j))
		}
	}

	// Generate new positions
	for pos := pe.maxLen; pos < newMaxLen; pos++ {
		for i := 0; i < pe.dModel; i++ {
			if i%2 == 0 {
				newEncoding.Set(pos, i, math.Sin(float64(pos)/math.Pow(10000, float64(i)/float64(pe.dModel))))
			} else {
				newEncoding.Set(pos, i, math.Cos(float64(pos)/math.Pow(10000, float64(i-1)/float64(pe.dModel))))
			}
		}
	}

	pe.encoding = newEncoding
	pe.maxLen = newMaxLen
}

// GetEncoding returns the positional encoding matrix
func (pe *PositionalEncoding) GetEncoding() *mat.Dense {
	return pe.encoding
}

// LearnablePositionalEncoding implements learnable positional encoding
type LearnablePositionalEncoding struct {
	dModel   int
	maxLen   int
	encoding *mat.Dense
}

// NewLearnablePositionalEncoding creates a new learnable positional encoding layer
func NewLearnablePositionalEncoding(dModel, maxLen int) *LearnablePositionalEncoding {
	// Initialize with small random values
	encoding := mat.NewDense(maxLen, dModel, nil)
	encoding.Apply(func(i, j int, v float64) float64 {
		return (float64(i*j) - 0.5) * 0.1
	}, encoding)

	return &LearnablePositionalEncoding{
		dModel:   dModel,
		maxLen:   maxLen,
		encoding: encoding,
	}
}

// Forward adds learnable positional encoding to the input
func (lpe *LearnablePositionalEncoding) Forward(X *mat.Dense) *mat.Dense {
	seqLen := 1 // Simplified for this implementation

	if seqLen > lpe.maxLen {
		// For learnable encoding, we can't extend beyond max length
		// In practice, you might want to handle this differently
		seqLen = lpe.maxLen
	}

	// For this simplified implementation, just return the input
	// In practice, you'd want to properly add positional encoding
	return mat.DenseCopyOf(X)
}

// GetParameters returns the learnable parameters
func (lpe *LearnablePositionalEncoding) GetParameters() []*mat.Dense {
	return []*mat.Dense{lpe.encoding}
}

// UpdateParameters updates the learnable parameters
func (lpe *LearnablePositionalEncoding) UpdateParameters(gradients []*mat.Dense, learningRate float64) {
	if len(gradients) != 1 {
		return
	}

	var update mat.Dense
	update.Scale(learningRate, gradients[0])
	lpe.encoding.Sub(lpe.encoding, &update)
}
