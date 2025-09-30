package ann

import (
	"math/rand"

	"github.com/pilillo/yannp/shared/loss"
	"github.com/pilillo/yannp/shared/optimizer"
	"gonum.org/v1/gonum/mat"
)

type Ann interface {
	Train(x, y *mat.Dense, numEpochs int, learningRate float64)
	Predict(x *mat.Dense) *mat.Dense
	SetLossFunction(lossFunction loss.LossFunction)
	SetOptimizer(optimizer optimizer.Optimizer)
	TrainWithValidation(x, y, valX, valY *mat.Dense, numEpochs int, learningRate float64) []float64
	TrainWithEarlyStopping(x, y, valX, valY *mat.Dense, numEpochs int, learningRate float64, patience int) []float64
}

type ann struct {
	layers []Layer
}

func (ann *ann) Predict(x *mat.Dense) *mat.Dense {
	var layerOutput *mat.Dense
	layerInput := mat.DenseCopyOf(x)

	for l := 0; l < len(ann.layers); l++ {
		layerOutput = ann.layers[l].Forward(layerInput)
		layerInput = mat.DenseCopyOf(layerOutput)
	}

	return layerOutput
}

func (ann *ann) Train(x, y *mat.Dense, numEpochs int, learningRate float64) {
	for i := 0; i < numEpochs; i++ {
		// forward propagation
		var layerOutput *mat.Dense
		layerInput := mat.DenseCopyOf(x)
		layerOutputs := []*mat.Dense{}
		for l := 0; l < len(ann.layers); l++ {
			layerOutput = ann.layers[l].Forward(layerInput)
			layerOutputs = append(layerOutputs, layerOutput)
			layerInput = mat.DenseCopyOf(layerOutput)
		}

		// compute prediction error on the network output
		networkError := mat.DenseCopyOf(layerOutput)
		networkError.Sub(y, layerOutput)

		// compute error for each layer
		var layerError *mat.Dense
		var dLayer *mat.Dense
		nextLayerError := mat.DenseCopyOf(networkError)
		derivatives := []*mat.Dense{}
		errors := []*mat.Dense{}
		// iterate backward to collect errors
		for l := len(ann.layers) - 1; l >= 0; l-- {
			//println(fmt.Sprintf("Backpropagating error to layer #%d of %d layers", l, len(ann.layers)))
			dLayer, layerError = ann.layers[l].ComputeError(nextLayerError, layerOutputs[l])
			// append errors in reversed order
			//fmt.Printf("\nerror[%d] = % v\n\n", l, mat.Formatted(layerError, mat.Prefix("          ")))
			errors = append([]*mat.Dense{layerError}, errors...)
			//fmt.Printf("\ndLayer[%d] = % v\n\n", l, mat.Formatted(dLayer, mat.Prefix("          ")))
			derivatives = append([]*mat.Dense{dLayer}, derivatives...)
			nextLayerError = layerError
		}

		// compute and update gradients
		for l := 0; l < len(ann.layers); l++ {
			if l == 0 {
				layerInput = x
			} else {
				layerInput = layerOutputs[l-1]
			}
			ann.layers[l].Update(learningRate, layerInput, derivatives[l], errors[l])
		}
	}

}

// Enhanced method implementations for ann struct

func (ann *ann) SetLossFunction(lossFunction loss.LossFunction) {
	// Basic ann doesn't support custom loss functions
}

func (ann *ann) SetOptimizer(opt optimizer.Optimizer) {
	// Basic ann doesn't support custom optimizers
}

func (ann *ann) TrainWithValidation(x, y, valX, valY *mat.Dense, numEpochs int, learningRate float64) []float64 {
	// For basic ann, just train normally and return zeros for validation losses
	ann.Train(x, y, numEpochs, learningRate)
	return make([]float64, numEpochs)
}

func (ann *ann) TrainWithEarlyStopping(x, y, valX, valY *mat.Dense, numEpochs int, learningRate float64, patience int) []float64 {
	// For basic ann, just train normally and return zeros for validation losses
	ann.Train(x, y, numEpochs, learningRate)
	return make([]float64, numEpochs)
}

func (ann *ann) Initialize(randGen *rand.Rand) {
	// Initialize all layers
	for _, layer := range ann.layers {
		layer.Initialize(randGen)
	}
}
