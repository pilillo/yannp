package errors

import (
	"fmt"
)

// Custom error types for the neural network package
type NeuralNetworkError struct {
	Message string
	Type    ErrorType
}

type ErrorType int

const (
	ErrInvalidInput ErrorType = iota
	ErrInvalidDimensions
	ErrInvalidActivation
	ErrInvalidOptimizer
	ErrInvalidLossFunction
	ErrInvalidRegularizer
	ErrInvalidInitializer
	ErrTrainingError
	ErrPredictionError
	ErrConfigurationError
)

func (e *NeuralNetworkError) Error() string {
	return fmt.Sprintf("[%s] %s", e.Type.String(), e.Message)
}

func (et ErrorType) String() string {
	switch et {
	case ErrInvalidInput:
		return "InvalidInput"
	case ErrInvalidDimensions:
		return "InvalidDimensions"
	case ErrInvalidActivation:
		return "InvalidActivation"
	case ErrInvalidOptimizer:
		return "InvalidOptimizer"
	case ErrInvalidLossFunction:
		return "InvalidLossFunction"
	case ErrInvalidRegularizer:
		return "InvalidRegularizer"
	case ErrInvalidInitializer:
		return "InvalidInitializer"
	case ErrTrainingError:
		return "TrainingError"
	case ErrPredictionError:
		return "PredictionError"
	case ErrConfigurationError:
		return "ConfigurationError"
	default:
		return "UnknownError"
	}
}

// Helper functions to create specific errors
func NewInvalidInputError(message string) *NeuralNetworkError {
	return &NeuralNetworkError{
		Message: message,
		Type:    ErrInvalidInput,
	}
}

func NewInvalidDimensionsError(message string) *NeuralNetworkError {
	return &NeuralNetworkError{
		Message: message,
		Type:    ErrInvalidDimensions,
	}
}

func NewInvalidActivationError(message string) *NeuralNetworkError {
	return &NeuralNetworkError{
		Message: message,
		Type:    ErrInvalidActivation,
	}
}

func NewInvalidOptimizerError(message string) *NeuralNetworkError {
	return &NeuralNetworkError{
		Message: message,
		Type:    ErrInvalidOptimizer,
	}
}

func NewInvalidLossFunctionError(message string) *NeuralNetworkError {
	return &NeuralNetworkError{
		Message: message,
		Type:    ErrInvalidLossFunction,
	}
}

func NewInvalidRegularizerError(message string) *NeuralNetworkError {
	return &NeuralNetworkError{
		Message: message,
		Type:    ErrInvalidRegularizer,
	}
}

func NewInvalidInitializerError(message string) *NeuralNetworkError {
	return &NeuralNetworkError{
		Message: message,
		Type:    ErrInvalidInitializer,
	}
}

func NewTrainingError(message string) *NeuralNetworkError {
	return &NeuralNetworkError{
		Message: message,
		Type:    ErrTrainingError,
	}
}

func NewPredictionError(message string) *NeuralNetworkError {
	return &NeuralNetworkError{
		Message: message,
		Type:    ErrPredictionError,
	}
}

func NewConfigurationError(message string) *NeuralNetworkError {
	return &NeuralNetworkError{
		Message: message,
		Type:    ErrConfigurationError,
	}
}
