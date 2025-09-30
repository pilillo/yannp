package ann

import (
	"math"
)

type ActivationFunction interface {
	Forward(x float64) float64
	Backward(x float64) float64
}

type sigmoidActivation struct{}

func NewSigmoidActivation() sigmoidActivation {
	return sigmoidActivation{}
}

func (sigmoidActivation) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (s sigmoidActivation) Forward(x float64) float64 {
	return s.sigmoid(x)
}

func (s sigmoidActivation) Backward(x float64) float64 {
	return s.sigmoid(x) * (1.0 - s.sigmoid(x))
}

type reluActivation struct{}

func NewReluActivation() reluActivation {
	return reluActivation{}
}

func (r reluActivation) Forward(x float64) float64 {
	return math.Max(0, x)
}

func (r reluActivation) Backward(x float64) float64 {
	if x >= 0 {
		return 1.0
	} else {
		return 0.0
	}
}

type tanhActivation struct{}

func NewTanhActivation() tanhActivation {
	return tanhActivation{}
}

func (t tanhActivation) Forward(x float64) float64 {
	return math.Tanh(x)
}

func (t tanhActivation) Backward(x float64) float64 {
	tanhX := math.Tanh(x)
	return 1.0 - tanhX*tanhX
}

type leakyReluActivation struct {
	alpha float64
}

func NewLeakyReluActivation(alpha float64) leakyReluActivation {
	return leakyReluActivation{alpha: alpha}
}

func (l leakyReluActivation) Forward(x float64) float64 {
	if x >= 0 {
		return x
	} else {
		return l.alpha * x
	}
}

func (l leakyReluActivation) Backward(x float64) float64 {
	if x >= 0 {
		return 1.0
	} else {
		return l.alpha
	}
}

type eluActivation struct {
	alpha float64
}

func NewEluActivation(alpha float64) eluActivation {
	return eluActivation{alpha: alpha}
}

func (e eluActivation) Forward(x float64) float64 {
	if x >= 0 {
		return x
	} else {
		return e.alpha * (math.Exp(x) - 1.0)
	}
}

func (e eluActivation) Backward(x float64) float64 {
	if x >= 0 {
		return 1.0
	} else {
		return e.alpha * math.Exp(x)
	}
}

// Note: Softmax activation is not included here as it should be implemented
// at the layer level rather than as a per-element activation function.
// Softmax requires knowledge of all elements in the vector to compute
// the normalized probabilities correctly.
