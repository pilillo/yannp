package ann

import (
	"math/rand"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	Randomize(randGen *rand.Rand)
	Forward(input *mat.Dense) *mat.Dense
	ComputeError(outputError *mat.Dense, layerOutput *mat.Dense) (layerDerivative *mat.Dense, layerInputError *mat.Dense)
	Update(learningRate float64, layerInput *mat.Dense, dLayer *mat.Dense, layerError *mat.Dense)
}

type layer struct {
	name                  string
	inputSize, outputSize int
	weights               *mat.Dense
	bias                  *mat.Dense
	activationFunction    *ActivationFunction
}

func CreateLayer(name string, inputSize, outputSize int, activationFunction *ActivationFunction) Layer {
	weights := mat.NewDense(
		inputSize, outputSize,
		make([]float64, inputSize*outputSize),
	)

	bias := mat.NewDense(1, outputSize, make([]float64, outputSize))

	return &layer{
		name:               name,
		inputSize:          inputSize,
		outputSize:         outputSize,
		weights:            weights,
		bias:               bias,
		activationFunction: activationFunction,
	}
}

func (l *layer) Randomize(randGen *rand.Rand) {
	l.weights.Apply(func(i, j int, v float64) float64 {
		//return 2.0 * (rand.Float64() - 0.5) / 100
		return randGen.Float64()
	}, l.weights)
}

func (l *layer) Forward(input *mat.Dense) *mat.Dense {
	// compute layer input
	var z2 mat.Dense
	z2.Mul(input, l.weights)

	// compute layer activation by applying the activation function
	var a2 mat.Dense
	a2.Apply(func(r, c int, v float64) float64 {
		return (*l.activationFunction).Forward(
			v + l.bias.At(0, c),
		)
	}, &z2)

	return &a2
}

func (l *layer) ComputeError(outputError *mat.Dense, layerOutput *mat.Dense) (*mat.Dense, *mat.Dense) {

	var layerSlope mat.Dense
	layerSlope.Apply(func(_, _ int, v float64) float64 {
		return (*l.activationFunction).Backward(v)
	}, layerOutput)

	var dLayer mat.Dense
	dLayer.MulElem(outputError, &layerSlope)

	var layerError mat.Dense
	layerError.Mul(&dLayer, l.weights.T())

	return &dLayer, &layerError
}

func (l *layer) Update(learningRate float64, layerInput *mat.Dense, dLayer *mat.Dense, layerError *mat.Dense) {
	// 1. weight update
	var weightUpdate mat.Dense
	weightUpdate.Mul(layerInput.T(), dLayer)
	weightUpdate.Scale(learningRate, &weightUpdate)
	l.weights.Add(l.weights, &weightUpdate)

	// 2. bias update
	// sum over 0 axis
	_, numCols := dLayer.Dims()
	data := make([]float64, numCols)
	for i := 0; i < numCols; i++ {
		col := mat.Col(nil, i, dLayer)
		data[i] = floats.Sum(col)
	}
	bOutAdj := mat.NewDense(1, numCols, data)
	bOutAdj.Scale(learningRate, bOutAdj)
	l.bias.Add(l.bias, bOutAdj)
}