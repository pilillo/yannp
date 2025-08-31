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
