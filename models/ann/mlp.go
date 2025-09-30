package ann

import (
	"github.com/pilillo/yannp/shared/loss"
	"github.com/pilillo/yannp/shared/optimizer"
	"gonum.org/v1/gonum/mat"
)

type mlp struct {
	ann
}

func NewMLP(inputNeurons, hiddenNeurons, outputNeurons int, activationFunction ActivationFunction) *mlp {
	return &mlp{
		ann: ann{
			layers: []Layer{
				CreateLayer("hidden layer", inputNeurons, hiddenNeurons, &activationFunction),
				CreateLayer("output layer", hiddenNeurons, outputNeurons, &activationFunction),
			},
		},
	}
}

// SetLossFunction is a no-op for basic MLP
func (m *mlp) SetLossFunction(lossFunction loss.LossFunction) {
	// Basic MLP doesn't support custom loss functions
}

// SetOptimizer is a no-op for basic MLP
func (m *mlp) SetOptimizer(opt optimizer.Optimizer) {
	// Basic MLP doesn't support custom optimizers
}

// TrainWithValidation falls back to basic training for basic MLP
func (m *mlp) TrainWithValidation(x, y, valX, valY *mat.Dense, numEpochs int, learningRate float64) []float64 {
	// For basic MLP, just train normally and return zeros for validation losses
	m.Train(x, y, numEpochs, learningRate)
	return make([]float64, numEpochs)
}

// TrainWithEarlyStopping falls back to basic training for basic MLP
func (m *mlp) TrainWithEarlyStopping(x, y, valX, valY *mat.Dense, numEpochs int, learningRate float64, patience int) []float64 {
	// For basic MLP, just train normally and return zeros for validation losses
	m.Train(x, y, numEpochs, learningRate)
	return make([]float64, numEpochs)
}
