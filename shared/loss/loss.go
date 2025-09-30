package loss

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// LossFunction interface defines the contract for different loss functions
type LossFunction interface {
	Forward(predictions, targets *mat.Dense) float64
	Backward(predictions, targets *mat.Dense) *mat.Dense
	GetName() string
}

// MeanSquaredError implements MSE loss
type MeanSquaredError struct{}

func NewMeanSquaredError() *MeanSquaredError {
	return &MeanSquaredError{}
}

func (mse *MeanSquaredError) Forward(predictions, targets *mat.Dense) float64 {
	var diff mat.Dense
	diff.Sub(predictions, targets)

	var squared mat.Dense
	squared.MulElem(&diff, &diff)

	rows, cols := squared.Dims()
	sum := 0.0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			sum += squared.At(i, j)
		}
	}

	return sum / float64(rows*cols)
}

func (mse *MeanSquaredError) Backward(predictions, targets *mat.Dense) *mat.Dense {
	rows, cols := predictions.Dims()
	gradients := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			grad := 2.0 * (predictions.At(i, j) - targets.At(i, j)) / float64(rows*cols)
			gradients.Set(i, j, grad)
		}
	}

	return gradients
}

func (mse *MeanSquaredError) GetName() string {
	return "MSE"
}

// MeanAbsoluteError implements MAE loss
type MeanAbsoluteError struct{}

func NewMeanAbsoluteError() *MeanAbsoluteError {
	return &MeanAbsoluteError{}
}

func (mae *MeanAbsoluteError) Forward(predictions, targets *mat.Dense) float64 {
	rows, cols := predictions.Dims()
	sum := 0.0

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			sum += math.Abs(predictions.At(i, j) - targets.At(i, j))
		}
	}

	return sum / float64(rows*cols)
}

func (mae *MeanAbsoluteError) Backward(predictions, targets *mat.Dense) *mat.Dense {
	rows, cols := predictions.Dims()
	gradients := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			diff := predictions.At(i, j) - targets.At(i, j)
			var grad float64
			if diff > 0 {
				grad = 1.0 / float64(rows*cols)
			} else if diff < 0 {
				grad = -1.0 / float64(rows*cols)
			} else {
				grad = 0.0
			}
			gradients.Set(i, j, grad)
		}
	}

	return gradients
}

func (mae *MeanAbsoluteError) GetName() string {
	return "MAE"
}

// CrossEntropy implements cross-entropy loss for classification
type CrossEntropy struct{}

func NewCrossEntropy() *CrossEntropy {
	return &CrossEntropy{}
}

func (ce *CrossEntropy) Forward(predictions, targets *mat.Dense) float64 {
	rows, cols := predictions.Dims()
	sum := 0.0

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			pred := predictions.At(i, j)
			target := targets.At(i, j)

			// Add small epsilon to prevent log(0)
			pred = math.Max(pred, 1e-15)
			pred = math.Min(pred, 1-1e-15)

			sum += target * math.Log(pred)
		}
	}

	return -sum / float64(rows)
}

func (ce *CrossEntropy) Backward(predictions, targets *mat.Dense) *mat.Dense {
	rows, cols := predictions.Dims()
	gradients := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			pred := predictions.At(i, j)
			target := targets.At(i, j)

			// Add small epsilon to prevent division by zero
			pred = math.Max(pred, 1e-15)

			grad := -target / pred / float64(rows)
			gradients.Set(i, j, grad)
		}
	}

	return gradients
}

func (ce *CrossEntropy) GetName() string {
	return "CrossEntropy"
}

// BinaryCrossEntropy implements binary cross-entropy loss
type BinaryCrossEntropy struct{}

func NewBinaryCrossEntropy() *BinaryCrossEntropy {
	return &BinaryCrossEntropy{}
}

func (bce *BinaryCrossEntropy) Forward(predictions, targets *mat.Dense) float64 {
	rows, cols := predictions.Dims()
	sum := 0.0

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			pred := predictions.At(i, j)
			target := targets.At(i, j)

			// Clip predictions to prevent log(0)
			pred = math.Max(pred, 1e-15)
			pred = math.Min(pred, 1-1e-15)

			sum += target*math.Log(pred) + (1-target)*math.Log(1-pred)
		}
	}

	return -sum / float64(rows*cols)
}

func (bce *BinaryCrossEntropy) Backward(predictions, targets *mat.Dense) *mat.Dense {
	rows, cols := predictions.Dims()
	gradients := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			pred := predictions.At(i, j)
			target := targets.At(i, j)

			// Clip predictions to prevent division by zero
			pred = math.Max(pred, 1e-15)
			pred = math.Min(pred, 1-1e-15)

			grad := (pred - target) / (pred * (1 - pred)) / float64(rows*cols)
			gradients.Set(i, j, grad)
		}
	}

	return gradients
}

func (bce *BinaryCrossEntropy) GetName() string {
	return "BinaryCrossEntropy"
}

// HingeLoss implements hinge loss for SVM-like classification
type HingeLoss struct{}

func NewHingeLoss() *HingeLoss {
	return &HingeLoss{}
}

func (hl *HingeLoss) Forward(predictions, targets *mat.Dense) float64 {
	rows, cols := predictions.Dims()
	sum := 0.0

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			pred := predictions.At(i, j)
			target := targets.At(i, j)

			loss := math.Max(0, 1-target*pred)
			sum += loss
		}
	}

	return sum / float64(rows*cols)
}

func (hl *HingeLoss) Backward(predictions, targets *mat.Dense) *mat.Dense {
	rows, cols := predictions.Dims()
	gradients := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			pred := predictions.At(i, j)
			target := targets.At(i, j)

			var grad float64
			if target*pred < 1 {
				grad = -target / float64(rows*cols)
			} else {
				grad = 0.0
			}
			gradients.Set(i, j, grad)
		}
	}

	return gradients
}

func (hl *HingeLoss) GetName() string {
	return "HingeLoss"
}
