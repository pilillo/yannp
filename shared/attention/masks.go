package attention

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// AttentionMask interface defines the contract for attention masks
type AttentionMask interface {
	GetMask(seqLen int) *mat.Dense
	GetName() string
}

// CausalMask implements causal (lower triangular) mask for autoregressive models
type CausalMask struct{}

// NewCausalMask creates a new causal mask
func NewCausalMask() *CausalMask {
	return &CausalMask{}
}

// GetMask returns a causal mask of the specified size
func (cm *CausalMask) GetMask(seqLen int) *mat.Dense {
	mask := mat.NewDense(seqLen, seqLen, nil)

	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j <= i {
				mask.Set(i, j, 1.0)
			} else {
				mask.Set(i, j, -math.Inf(1)) // -inf for masked positions
			}
		}
	}

	return mask
}

// GetName returns the mask name
func (cm *CausalMask) GetName() string {
	return "CausalMask"
}

// PaddingMask implements padding mask for variable-length sequences
type PaddingMask struct {
	lengths []int // Length of each sequence in the batch
}

// NewPaddingMask creates a new padding mask
func NewPaddingMask(lengths []int) *PaddingMask {
	return &PaddingMask{lengths: lengths}
}

// GetMask returns a padding mask
func (pm *PaddingMask) GetMask(seqLen int) *mat.Dense {
	maxLen := 0
	for _, length := range pm.lengths {
		if length > maxLen {
			maxLen = length
		}
	}

	if maxLen > seqLen {
		maxLen = seqLen
	}

	mask := mat.NewDense(len(pm.lengths), maxLen, nil)

	for i, length := range pm.lengths {
		for j := 0; j < maxLen; j++ {
			if j < length {
				mask.Set(i, j, 1.0)
			} else {
				mask.Set(i, j, -math.Inf(1)) // -inf for padded positions
			}
		}
	}

	return mask
}

// GetName returns the mask name
func (pm *PaddingMask) GetName() string {
	return "PaddingMask"
}

// LookAheadMask implements look-ahead mask for bidirectional models
type LookAheadMask struct {
	lookAheadSteps int
}

// NewLookAheadMask creates a new look-ahead mask
func NewLookAheadMask(lookAheadSteps int) *LookAheadMask {
	return &LookAheadMask{lookAheadSteps: lookAheadSteps}
}

// GetMask returns a look-ahead mask
func (lam *LookAheadMask) GetMask(seqLen int) *mat.Dense {
	mask := mat.NewDense(seqLen, seqLen, nil)

	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j <= i+lam.lookAheadSteps {
				mask.Set(i, j, 1.0)
			} else {
				mask.Set(i, j, -math.Inf(1)) // -inf for masked positions
			}
		}
	}

	return mask
}

// GetName returns the mask name
func (lam *LookAheadMask) GetName() string {
	return "LookAheadMask"
}

// CombinedMask combines multiple masks
type CombinedMask struct {
	masks []AttentionMask
}

// NewCombinedMask creates a new combined mask
func NewCombinedMask(masks ...AttentionMask) *CombinedMask {
	return &CombinedMask{masks: masks}
}

// GetMask returns the combined mask
func (cm *CombinedMask) GetMask(seqLen int) *mat.Dense {
	if len(cm.masks) == 0 {
		// Return identity mask if no masks provided
		return mat.NewDense(seqLen, seqLen, nil)
	}

	// Start with the first mask
	result := cm.masks[0].GetMask(seqLen)

	// Apply element-wise minimum with other masks
	for i := 1; i < len(cm.masks); i++ {
		mask := cm.masks[i].GetMask(seqLen)
		result.MulElem(result, mask)
	}

	return result
}

// GetName returns the mask name
func (cm *CombinedMask) GetName() string {
	return "CombinedMask"
}

// NoMask implements no masking (all positions are allowed)
type NoMask struct{}

// NewNoMask creates a new no-mask
func NewNoMask() *NoMask {
	return &NoMask{}
}

// GetMask returns an identity mask (no masking)
func (nm *NoMask) GetMask(seqLen int) *mat.Dense {
	mask := mat.NewDense(seqLen, seqLen, nil)

	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			mask.Set(i, j, 1.0)
		}
	}

	return mask
}

// GetName returns the mask name
func (nm *NoMask) GetName() string {
	return "NoMask"
}
