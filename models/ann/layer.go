package ann

import (
	"math/rand"

	"github.com/gonum/floats"
	"github.com/pilillo/yannp/shared/initialization"
	"github.com/pilillo/yannp/shared/optimizer"
	"github.com/pilillo/yannp/shared/regularization"
	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	Randomize(randGen *rand.Rand)
	Forward(input *mat.Dense) *mat.Dense
	ComputeError(outputError *mat.Dense, layerOutput *mat.Dense) (layerDerivative *mat.Dense, layerInputError *mat.Dense)
	Update(learningRate float64, layerInput *mat.Dense, dLayer *mat.Dense, layerError *mat.Dense)
	SetOptimizer(optimizer optimizer.Optimizer)
	SetWeightInitializer(initializer initialization.WeightInitializer)
	SetRegularizer(regularizer regularization.Regularizer)
	SetDropout(dropout *regularization.Dropout)
	SetBatchNorm(batchNorm *regularization.BatchNormalization)
	Initialize(randGen *rand.Rand)
	ForwardWithRegularization(input *mat.Dense, training bool) *mat.Dense
	UpdateWithOptimizer(learningRate float64, layerInput *mat.Dense, dLayer *mat.Dense, layerError *mat.Dense, iteration int)
}

type layer struct {
	name                  string
	inputSize, outputSize int
	weights               *mat.Dense
	bias                  *mat.Dense
	activationFunction    *ActivationFunction
	// Enhanced fields (nil by default for basic layers)
	optimizer         optimizer.Optimizer
	weightInitializer initialization.WeightInitializer
	regularizer       regularization.Regularizer
	dropout           *regularization.Dropout
	batchNorm         *regularization.BatchNormalization
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

// Enhanced method implementations

func (l *layer) SetOptimizer(opt optimizer.Optimizer) {
	l.optimizer = opt
}

func (l *layer) SetWeightInitializer(initializer initialization.WeightInitializer) {
	l.weightInitializer = initializer
}

func (l *layer) SetRegularizer(regularizer regularization.Regularizer) {
	l.regularizer = regularizer
}

func (l *layer) SetDropout(dropout *regularization.Dropout) {
	l.dropout = dropout
}

func (l *layer) SetBatchNorm(batchNorm *regularization.BatchNormalization) {
	l.batchNorm = batchNorm
}

func (l *layer) Initialize(randGen *rand.Rand) {
	if l.weightInitializer != nil {
		l.weightInitializer.Initialize(l.weights, l.inputSize, l.outputSize, randGen)
	} else {
		// Fallback to basic randomization
		l.Randomize(randGen)
	}
}

func (l *layer) ForwardWithRegularization(input *mat.Dense, training bool) *mat.Dense {
	// Apply batch normalization if present
	var normalizedInput *mat.Dense
	if l.batchNorm != nil {
		l.batchNorm.SetTraining(training)
		normalizedInput = l.batchNorm.Forward(input)
	} else {
		normalizedInput = input
	}

	// Compute layer output
	output := l.Forward(normalizedInput)

	// Apply dropout if present
	if l.dropout != nil {
		l.dropout.SetTraining(training)
		output = l.dropout.Apply(output, rand.New(rand.NewSource(rand.Int63())))
	}

	return output
}

func (l *layer) UpdateWithOptimizer(learningRate float64, layerInput *mat.Dense, dLayer *mat.Dense, layerError *mat.Dense, iteration int) {
	// Compute weight gradients
	var weightGradients mat.Dense
	weightGradients.Mul(layerInput.T(), dLayer)

	// Apply regularization if present
	if l.regularizer != nil {
		regularization := l.regularizer.Regularize(l.weights, learningRate)
		weightGradients.Add(&weightGradients, regularization)
	}

	// Update weights using optimizer
	if l.optimizer != nil {
		l.optimizer.Update(l.weights, &weightGradients, learningRate, iteration)
	} else {
		// Fallback to basic SGD
		weightGradients.Scale(learningRate, &weightGradients)
		l.weights.Add(l.weights, &weightGradients)
	}

	// Update bias (simple SGD for bias)
	_, numCols := dLayer.Dims()
	data := make([]float64, numCols)
	for i := 0; i < numCols; i++ {
		col := mat.Col(nil, i, dLayer)
		data[i] = floats.Sum(col)
	}
	biasGradients := mat.NewDense(1, numCols, data)
	biasGradients.Scale(learningRate, biasGradients)
	l.bias.Add(l.bias, biasGradients)
}
