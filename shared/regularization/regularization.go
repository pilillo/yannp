package regularization

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Regularizer interface defines the contract for different regularization techniques
type Regularizer interface {
	Regularize(weights *mat.Dense, learningRate float64) *mat.Dense
	GetName() string
}

// L1Regularization implements L1 (Lasso) regularization
type L1Regularization struct {
	lambda float64
}

func NewL1Regularization(lambda float64) *L1Regularization {
	return &L1Regularization{lambda: lambda}
}

func (l1 *L1Regularization) Regularize(weights *mat.Dense, learningRate float64) *mat.Dense {
	rows, cols := weights.Dims()
	regularization := mat.NewDense(rows, cols, nil)

	regularization.Apply(func(i, j int, v float64) float64 {
		weight := weights.At(i, j)
		if weight > 0 {
			return l1.lambda * learningRate
		} else if weight < 0 {
			return -l1.lambda * learningRate
		} else {
			return 0.0
		}
	}, weights)

	return regularization
}

func (l1 *L1Regularization) GetName() string {
	return "L1"
}

// L2Regularization implements L2 (Ridge) regularization
type L2Regularization struct {
	lambda float64
}

func NewL2Regularization(lambda float64) *L2Regularization {
	return &L2Regularization{lambda: lambda}
}

func (l2 *L2Regularization) Regularize(weights *mat.Dense, learningRate float64) *mat.Dense {
	rows, cols := weights.Dims()
	regularization := mat.NewDense(rows, cols, nil)

	regularization.Apply(func(i, j int, v float64) float64 {
		return l2.lambda * learningRate * weights.At(i, j)
	}, weights)

	return regularization
}

func (l2 *L2Regularization) GetName() string {
	return "L2"
}

// ElasticNet implements Elastic Net regularization (combination of L1 and L2)
type ElasticNet struct {
	l1Lambda, l2Lambda float64
}

func NewElasticNet(l1Lambda, l2Lambda float64) *ElasticNet {
	return &ElasticNet{l1Lambda: l1Lambda, l2Lambda: l2Lambda}
}

func (en *ElasticNet) Regularize(weights *mat.Dense, learningRate float64) *mat.Dense {
	rows, cols := weights.Dims()
	regularization := mat.NewDense(rows, cols, nil)

	regularization.Apply(func(i, j int, v float64) float64 {
		weight := weights.At(i, j)

		// L1 component
		var l1Grad float64
		if weight > 0 {
			l1Grad = en.l1Lambda * learningRate
		} else if weight < 0 {
			l1Grad = -en.l1Lambda * learningRate
		} else {
			l1Grad = 0.0
		}

		// L2 component
		l2Grad := en.l2Lambda * learningRate * weight

		return l1Grad + l2Grad
	}, weights)

	return regularization
}

func (en *ElasticNet) GetName() string {
	return "ElasticNet"
}

// Dropout implements dropout regularization
type Dropout struct {
	rate     float64
	training bool
}

func NewDropout(rate float64) *Dropout {
	return &Dropout{
		rate:     rate,
		training: true,
	}
}

func (d *Dropout) SetTraining(training bool) {
	d.training = training
}

func (d *Dropout) Apply(input *mat.Dense, randGen *rand.Rand) *mat.Dense {
	if !d.training {
		return input
	}

	rows, cols := input.Dims()
	output := mat.NewDense(rows, cols, nil)

	output.Apply(func(i, j int, v float64) float64 {
		if randGen.Float64() < d.rate {
			return 0.0
		}
		return v / (1.0 - d.rate) // Scale by inverse of dropout rate
	}, input)

	return output
}

func (d *Dropout) GetName() string {
	return "Dropout"
}

// BatchNormalization implements batch normalization
type BatchNormalization struct {
	gamma, beta *mat.Dense
	runningMean *mat.Dense
	runningVar  *mat.Dense
	momentum    float64
	epsilon     float64
	training    bool
}

func NewBatchNormalization(featureSize int, momentum, epsilon float64) *BatchNormalization {
	gamma := mat.NewDense(1, featureSize, nil)
	beta := mat.NewDense(1, featureSize, nil)
	runningMean := mat.NewDense(1, featureSize, nil)
	runningVar := mat.NewDense(1, featureSize, nil)

	// Initialize gamma to 1 and beta to 0
	for i := 0; i < featureSize; i++ {
		gamma.Set(0, i, 1.0)
		beta.Set(0, i, 0.0)
		runningMean.Set(0, i, 0.0)
		runningVar.Set(0, i, 1.0)
	}

	return &BatchNormalization{
		gamma:       gamma,
		beta:        beta,
		runningMean: runningMean,
		runningVar:  runningVar,
		momentum:    momentum,
		epsilon:     epsilon,
		training:    true,
	}
}

func (bn *BatchNormalization) SetTraining(training bool) {
	bn.training = training
}

func (bn *BatchNormalization) Forward(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	output := mat.NewDense(rows, cols, nil)

	if bn.training {
		// Training mode: compute batch statistics
		mean := bn.computeMean(input)
		variance := bn.computeVariance(input, mean)

		// Update running statistics
		bn.runningMean.Scale(bn.momentum, bn.runningMean)
		var temp1 mat.Dense
		temp1.Scale(1-bn.momentum, mean)
		bn.runningMean.Add(bn.runningMean, &temp1)

		bn.runningVar.Scale(bn.momentum, bn.runningVar)
		var temp2 mat.Dense
		temp2.Scale(1-bn.momentum, variance)
		bn.runningVar.Add(bn.runningVar, &temp2)

		// Normalize
		output.Apply(func(i, j int, v float64) float64 {
			normalized := (v - mean.At(0, j)) / math.Sqrt(variance.At(0, j)+bn.epsilon)
			return bn.gamma.At(0, j)*normalized + bn.beta.At(0, j)
		}, input)
	} else {
		// Inference mode: use running statistics
		output.Apply(func(i, j int, v float64) float64 {
			normalized := (v - bn.runningMean.At(0, j)) / math.Sqrt(bn.runningVar.At(0, j)+bn.epsilon)
			return bn.gamma.At(0, j)*normalized + bn.beta.At(0, j)
		}, input)
	}

	return output
}

func (bn *BatchNormalization) computeMean(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	mean := mat.NewDense(1, cols, nil)

	for j := 0; j < cols; j++ {
		sum := 0.0
		for i := 0; i < rows; i++ {
			sum += input.At(i, j)
		}
		mean.Set(0, j, sum/float64(rows))
	}

	return mean
}

func (bn *BatchNormalization) computeVariance(input *mat.Dense, mean *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	variance := mat.NewDense(1, cols, nil)

	for j := 0; j < cols; j++ {
		sum := 0.0
		meanVal := mean.At(0, j)
		for i := 0; i < rows; i++ {
			diff := input.At(i, j) - meanVal
			sum += diff * diff
		}
		variance.Set(0, j, sum/float64(rows))
	}

	return variance
}

func (bn *BatchNormalization) GetName() string {
	return "BatchNormalization"
}
