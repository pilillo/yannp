package validation

import (
	"math"

	"github.com/pilillo/yannp/shared/errors"
	"gonum.org/v1/gonum/mat"
)

// ValidationResult contains the results of input validation
type ValidationResult struct {
	IsValid bool
	Errors  []error
}

// ValidateInputs validates input matrices for training/prediction
func ValidateInputs(x, y *mat.Dense) *ValidationResult {
	result := &ValidationResult{
		IsValid: true,
		Errors:  make([]error, 0),
	}

	if x == nil {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidInputError("input matrix X cannot be nil"))
		return result
	}

	if y == nil {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidInputError("target matrix Y cannot be nil"))
		return result
	}

	xRows, xCols := x.Dims()
	yRows, yCols := y.Dims()

	// Check for empty matrices
	if xRows == 0 || xCols == 0 {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidInputError("input matrix X cannot be empty"))
	}

	if yRows == 0 || yCols == 0 {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidInputError("target matrix Y cannot be empty"))
	}

	// Check dimension consistency
	if xRows != yRows {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidDimensionsError(
			"input and target matrices must have the same number of rows"))
	}

	// Check for NaN or Inf values
	if hasNaNOrInf(x) {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidInputError("input matrix contains NaN or Inf values"))
	}

	if hasNaNOrInf(y) {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidInputError("target matrix contains NaN or Inf values"))
	}

	return result
}

// ValidatePredictionInputs validates input matrices for prediction
func ValidatePredictionInputs(x *mat.Dense) *ValidationResult {
	result := &ValidationResult{
		IsValid: true,
		Errors:  make([]error, 0),
	}

	if x == nil {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidInputError("input matrix X cannot be nil"))
		return result
	}

	xRows, xCols := x.Dims()

	// Check for empty matrices
	if xRows == 0 || xCols == 0 {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidInputError("input matrix X cannot be empty"))
	}

	// Check for NaN or Inf values
	if hasNaNOrInf(x) {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidInputError("input matrix contains NaN or Inf values"))
	}

	return result
}

// ValidateLayerConfiguration validates layer configuration parameters
func ValidateLayerConfiguration(inputSize, outputSize int, activationFunction interface{}) *ValidationResult {
	result := &ValidationResult{
		IsValid: true,
		Errors:  make([]error, 0),
	}

	if inputSize <= 0 {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidDimensionsError("input size must be positive"))
	}

	if outputSize <= 0 {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidDimensionsError("output size must be positive"))
	}

	if activationFunction == nil {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidActivationError("activation function cannot be nil"))
	}

	return result
}

// ValidateOptimizerConfiguration validates optimizer configuration
func ValidateOptimizerConfiguration(optimizer interface{}) *ValidationResult {
	result := &ValidationResult{
		IsValid: true,
		Errors:  make([]error, 0),
	}

	if optimizer == nil {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidOptimizerError("optimizer cannot be nil"))
	}

	return result
}

// ValidateLossFunctionConfiguration validates loss function configuration
func ValidateLossFunctionConfiguration(lossFunction interface{}) *ValidationResult {
	result := &ValidationResult{
		IsValid: true,
		Errors:  make([]error, 0),
	}

	if lossFunction == nil {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewInvalidLossFunctionError("loss function cannot be nil"))
	}

	return result
}

// ValidateRegularizerConfiguration validates regularizer configuration
func ValidateRegularizerConfiguration(regularizer interface{}) *ValidationResult {
	result := &ValidationResult{
		IsValid: true,
		Errors:  make([]error, 0),
	}

	// Regularizer can be nil (no regularization)
	// Note: Detailed parameter validation is handled within the regularization package
	// since the struct fields are private

	return result
}

// ValidateDropoutConfiguration validates dropout configuration
func ValidateDropoutConfiguration(dropout interface{}) *ValidationResult {
	result := &ValidationResult{
		IsValid: true,
		Errors:  make([]error, 0),
	}

	// Dropout can be nil (no dropout)
	// Note: Detailed parameter validation is handled within the regularization package
	// since the struct fields are private

	return result
}

// ValidateBatchNormConfiguration validates batch normalization configuration
func ValidateBatchNormConfiguration(batchNorm interface{}) *ValidationResult {
	result := &ValidationResult{
		IsValid: true,
		Errors:  make([]error, 0),
	}

	// Batch normalization can be nil (no batch normalization)
	// Note: Detailed parameter validation is handled within the regularization package
	// since the struct fields are private

	return result
}

// ValidateTrainingParameters validates training parameters
func ValidateTrainingParameters(numEpochs int, learningRate float64) *ValidationResult {
	result := &ValidationResult{
		IsValid: true,
		Errors:  make([]error, 0),
	}

	if numEpochs <= 0 {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewTrainingError("number of epochs must be positive"))
	}

	if learningRate <= 0 {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewTrainingError("learning rate must be positive"))
	}

	if learningRate > 1.0 {
		result.IsValid = false
		result.Errors = append(result.Errors, errors.NewTrainingError("learning rate is unusually high (>1.0)"))
	}

	return result
}

// Helper function to check for NaN or Inf values in a matrix
func hasNaNOrInf(m *mat.Dense) bool {
	rows, cols := m.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := m.At(i, j)
			if math.IsNaN(val) || math.IsInf(val, 0) {
				return true
			}
		}
	}
	return false
}
